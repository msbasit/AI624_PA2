"""ResNet-18 pruning experiments. The main notebook contains these implementations."""

from pathlib import Path
import copy, gc, hashlib, io, json, math, os, platform, random, subprocess, sys, time
import threading, zipfile
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.datasets import CIFAR10
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ROOT.mkdir(exist_ok=True)
for folder in ["results", "figures", "checkpoints", "source"]:
    (ROOT / folder).mkdir(exist_ok=True)
SEED = 624
BATCH = 128
TARGET = 0.70
FT_EPOCHS = 12
SCRATCH_EPOCHS = 50
STRUCT_EPOCHS = 18
DEVICE = "cuda"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2471, 0.2435, 0.2616)
UPSTREAM_COMMIT = "641cac24371b17052b9bb6e56af1c83b5e97cd7f"
CHECKPOINT_SHA256 = "72d30ca70e7d54e24a26628113604c40bc872eb8dc7a44f50ec58c3a1400f1b0"


# Shared utilities and model construction
# ----------------------------------------------------------------------------
def seed_all(seed=SEED):
    """Set the Python, NumPy, and PyTorch random seeds used by this run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(name, data):
    """Save measured values or experiment metadata in the results directory."""
    (ROOT / "results" / name).write_text(
        json.dumps(data, indent=2, allow_nan=False), encoding="utf-8"
    )


def cpu_state(model):
    """Copy a model's complete state to CPU memory for a portable checkpoint."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def eligible(model):
    """Return named Conv2d and Linear layers. Only their weights are pruned."""
    return {
        n: m for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))
    }


def fresh(state=None):
    """Build the required upstream ResNet-18 and optionally load a saved state.

    BatchNorm parameters and the architecture remain those of the original model.
    """
    model = resnet18(pretrained=False)
    if state is not None:
        model.load_state_dict(state, strict=True)
    for mod in model.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    return model.to(DEVICE)


# Data preparation: the original GPU-cached batch iterator
# ----------------------------------------------------------------------------
class TensorBatches:
    """The existing GPU-cached alternative to a PyTorch DataLoader.

    x contains uint8 images; y contains class labels. The batches method applies
    normalization and, for training only, the recorded crop and flip operations.
    This implementation is retained to preserve the original experiment exactly.
    """

    def __init__(self, x, y, indices, augment=False):
        self.x, self.y = x, y
        self.indices = torch.as_tensor(indices, device=DEVICE)
        self.augment = augment

    def __len__(self):
        return math.ceil(len(self.indices) / BATCH)

    def batches(self, epoch=0):
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + epoch)
        ids = self.indices
        if self.augment:
            ids = ids[torch.randperm(len(ids), generator=gen, device=DEVICE)]
        mean = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)
        for pos in range(0, len(ids), BATCH):
            ix = ids[pos : pos + BATCH]
            x = self.x[ix].float().div_(255)
            if self.augment:
                b = len(x)
                padded = F.pad(x, (4, 4, 4, 4))
                oy = torch.randint(9, (b, 1, 1), generator=gen, device=DEVICE)
                ox = torch.randint(9, (b, 1, 1), generator=gen, device=DEVICE)
                yy = oy + torch.arange(32, device=DEVICE).view(1, 32, 1)
                xx = ox + torch.arange(32, device=DEVICE).view(1, 1, 32)
                x = (
                    padded.permute(0, 2, 3, 1)[
                        torch.arange(b, device=DEVICE)[:, None, None], yy, xx
                    ]
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                flip = torch.rand(b, generator=gen, device=DEVICE) < 0.5
                x[flip] = x[flip].flip(-1)
            yield (x - mean) / std, self.y[ix]


def setup():
    """Load the required architecture, checkpoint, data splits, and batch iterators.

    Creates shared module state: DENSE is the pretrained state; TRAIN is the
    optimization split; TRAIN_EVAL evaluates the full training set; VALID and
    TEST are evaluation splits; CALIB is the GraSP calibration subset.
    Calling this function can download files and requires a CUDA GPU.
    """
    global resnet18, DENSE, TRAIN, TRAIN_EVAL, VALID, TEST, CALIB, ENV
    assert torch.cuda.is_available(), "A CUDA-enabled GPU is required to run the experiments."
    torch.set_num_threads(2)
    seed_all()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repo = ROOT / "source" / "PyTorch_CIFAR10"
    if not repo.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/huyvnphan/PyTorch_CIFAR10",
                str(repo),
            ],
            check=True,
        )
        current = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != UPSTREAM_COMMIT:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    UPSTREAM_COMMIT,
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "checkout", UPSTREAM_COMMIT], check=True
            )
    sys.path.insert(0, str(repo))
    from cifar10_models.resnet import resnet18 as upstream_resnet18

    resnet18 = upstream_resnet18
    checkpoint = ROOT / "checkpoints" / "resnet18.pt"
    if not checkpoint.exists():
        import requests

        archive = ROOT / "weights.zip"
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )
        print("Downloading official weights archive...", flush=True)
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with archive.open("wb") as f:
                    for chunk in response.iter_content(2**20):
                        f.write(chunk)
        except requests.RequestException as error:
            print(
                "Box unavailable; using the official README Google Drive backup:",
                error,
                flush=True,
            )
            import gdown

            gdown.download(
                id="17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq", output=str(archive), quiet=False
            )
        with zipfile.ZipFile(archive) as z:
            names = [
                n
                for n in z.namelist()
                if n.endswith("/resnet18.pt") or n == "resnet18.pt"
            ]
            assert len(names) == 1, names
            checkpoint.write_bytes(z.read(names[0]))
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    ), "Unexpected pretrained checkpoint hash."
    DENSE = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = fresh(DENSE)
    assert set(DENSE) == set(model.state_dict())
    assert all(torch.equal(v.cpu(), DENSE[k]) for k, v in model.state_dict().items())
    del model
    # Byte-identical mirror; torchvision verifies the official archive and batch MD5s.
    CIFAR10.url = (
        "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
    )
    tr = CIFAR10(str(ROOT / "data"), train=True, download=True)
    te = CIFAR10(str(ROOT / "data"), train=False, download=True)
    xtr = torch.tensor(tr.data, device=DEVICE).permute(0, 3, 1, 2).contiguous()
    ytr = torch.tensor(tr.targets, device=DEVICE)
    xte = torch.tensor(te.data, device=DEVICE).permute(0, 3, 1, 2).contiguous()
    yte = torch.tensor(te.targets, device=DEVICE)
    rng = np.random.default_rng(SEED)
    train_ids, val_ids, cal_ids = [], [], []
    labels = np.asarray(tr.targets)
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(labels == c))
        val_ids.extend(ids[:500])
        train_ids.extend(ids[500:])
        cal_ids.extend(ids[500:520])
    TRAIN = TensorBatches(xtr, ytr, train_ids, True)
    TRAIN_EVAL = TensorBatches(xtr, ytr, np.arange(50000))
    VALID = TensorBatches(xtr, ytr, val_ids)
    TEST = TensorBatches(xte, yte, np.arange(10000))
    CALIB = TensorBatches(xtr, ytr, cal_ids)
    save_json(
        "split_indices.json",
        {
            "train": list(map(int, train_ids)),
            "validation": list(map(int, val_ids)),
            "calibration": list(map(int, cal_ids)),
        },
    )
    ENV = {
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_MB": torch.cuda.get_device_properties(0).total_memory / 1e6,
        "cpu": platform.processor(),
        "batch_size": BATCH,
        "input": [3, 32, 32],
        "seed": SEED,
        "mean": MEAN,
        "std": STD,
        "train_samples": 45000,
        "validation_samples": 5000,
        "test_samples": 10000,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "repository_commit": (
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            if (repo / ".git").exists()
            else UPSTREAM_COMMIT
        ),
        "architecture_sha256": hashlib.sha256(
            (repo / "cifar10_models" / "resnet.py").read_bytes()
        ).hexdigest(),
        "target_sparsity": TARGET,
        "finetuning_epochs": FT_EPOCHS,
        "scratch_epochs": SCRATCH_EPOCHS,
        "structured_epochs": STRUCT_EPOCHS,
    }
    save_json("environment.json", ENV)
    print(json.dumps(ENV, indent=2), flush=True)


# Classification metrics
# ----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate(model, loader):
    """Evaluate a model without updating weights. Return accuracy, F1, and loss.

    x and y are images and labels; out contains class scores; cm is the
    confusion matrix with true classes in rows and predictions in columns.
    tp contains the correctly classified count for each class.
    """
    model.eval()
    cm = torch.zeros(10, 10, dtype=torch.int64, device=DEVICE)
    total, top5, loss_sum = 0, 0, 0.0
    for x, y in loader.batches():
        out = model(x)
        pred = out.argmax(1)
        cm += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
        top5 += (out.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
        total += len(y)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
    cm = cm.cpu().numpy()
    tp = np.diag(cm)
    f1 = np.divide(
        2 * tp,
        cm.sum(0) + cm.sum(1),
        out=np.zeros(10, dtype=float),
        where=(cm.sum(0) + cm.sum(1)) != 0,
    )
    return {
        "n": total,
        "top1": float(100 * tp.sum() / total),
        "top5": float(100 * top5 / total),
        "macro_f1": float(f1.mean()),
        "loss": loss_sum / total,
    }


def full_metrics(model):
    """Evaluate the complete official training and test sets without augmentation."""
    return {
        p + "_" + k: v
        for p, loader in [("train", TRAIN_EVAL), ("test", TEST)]
        for k, v in evaluate(model, loader).items()
    }


# Unstructured pruning masks
# ----------------------------------------------------------------------------
def masks_for(model):
    """Create one all-True weight mask for every eligible layer. True means retain."""
    return {
        n: torch.ones_like(m.weight, dtype=torch.bool)
        for n, m in eligible(model).items()
    }


@torch.no_grad()
def apply_masks(model, masks, optimizer=None):
    """Keep removed weights at zero and also clear their SGD momentum entries.

    This is called after optimizer updates so earlier removals cannot regrow.
    """
    layers = eligible(model)
    for n, mask in masks.items():
        w = layers[n].weight
        w.mul_(mask)
        if optimizer is not None:
            state = optimizer.state.get(w, {})
            if "momentum_buffer" in state:
                state["momentum_buffer"].mul_(mask)


def verify_masks(model, masks):
    """Check every removed weight is zero and compare mask and tensor sparsity."""
    rows = []
    for n, m in eligible(model).items():
        mask = masks[n]
        masked_zero = bool(torch.all(m.weight.detach()[~mask] == 0).item())
        assert masked_zero, n
        zeros = int((m.weight.detach() == 0).sum())
        removed = int((~mask).sum())
        rows.append(
            {
                "layer": n,
                "weights": mask.numel(),
                "mask_sparsity": removed / mask.numel(),
                "actual_zero_fraction": zeros / mask.numel(),
                "extra_natural_zeros": zeros - removed,
                "remaining_ratio": float(mask.float().mean()),
                "masked_positions_zero": masked_zero,
            }
        )
    return rows


def prune_global(model, masks, sparsity, scores=None, largest=False):
    """Extend cumulative masks to the requested overall sparsity.

    By default, remove the smallest absolute weights. For GraSP, supplied
    signed scores are ranked with largest=True. Previously removed positions
    are excluded from selection; tied scores use stable flattened index order.
    """
    layers = eligible(model)
    total = sum(m.weight.numel() for m in layers.values())
    already = sum(int((~v).sum()) for v in masks.values())
    count = round(total * sparsity) - already
    assert count >= 0
    if count == 0:
        return masks
    all_scores = torch.cat(
        [
            (scores[n] if scores is not None else m.weight.detach().abs()).flatten()
            for n, m in layers.items()
        ]
    )
    active = torch.cat([m.flatten() for m in masks.values()])
    all_scores[~active] = -torch.inf if largest else torch.inf
    assert torch.isfinite(all_scores[active]).all()
    chosen = torch.argsort(all_scores, descending=largest, stable=True)[:count]
    active[chosen] = False
    offset = 0
    for n, m in layers.items():
        size = m.weight.numel()
        masks[n] = active[offset : offset + size].reshape_as(m.weight).clone()
        offset += size
    apply_masks(model, masks)
    return masks


# Sparse storage and inference wrappers
# ----------------------------------------------------------------------------
class COOConv(nn.Module):
    """Store convolution weights in COO form and densify on every forward pass.

    The conversion cost is included in the recorded COO inference measurements.
    """

    def __init__(self, m):
        super().__init__()
        self.register_buffer("sparse_weight", m.weight.detach().to_sparse().coalesce())
        self.register_buffer(
            "bias", None if m.bias is None else m.bias.detach().clone()
        )
        self.stride, self.padding, self.dilation, self.groups = (
            m.stride,
            m.padding,
            m.dilation,
            m.groups,
        )

    def forward(self, x):
        return F.conv2d(
            x,
            self.sparse_weight.to_dense(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class COOLinear(nn.Module):
    """Apply the linear layer with sparse matrix multiplication and optional bias."""

    def __init__(self, m):
        super().__init__()
        self.register_buffer("sparse_weight", m.weight.detach().to_sparse().coalesce())
        self.register_buffer(
            "bias", None if m.bias is None else m.bias.detach().clone()
        )

    def forward(self, x):
        y = torch.sparse.mm(self.sparse_weight, x.t()).t()
        return y if self.bias is None else y + self.bias


def coo_model(model):
    """Copy a model and replace eligible layers with the COO inference wrappers."""
    out = copy.deepcopy(model)
    for n, m in list(eligible(out).items()):
        parent_name, _, name = n.rpartition(".")
        parent = out.get_submodule(parent_name) if parent_name else out
        setattr(parent, name, COOConv(m) if isinstance(m, nn.Conv2d) else COOLinear(m))
    return out.eval()


def save_sparse(model, masks, name):
    """Save COO values, indices, shapes, masks, and the remaining dense state.

    Reload and reconstruct each sparse tensor to verify exact weight equality.
    """
    state = cpu_state(model)
    tensors = {}
    for n, m in eligible(model).items():
        key = n + ".weight"
        coo = state.pop(key).to_sparse().coalesce()
        tensors[key] = {
            "values": coo.values(),
            "indices": coo.indices(),
            "shape": tuple(coo.shape),
            "mask": masks[n].cpu(),
        }
    path = ROOT / "checkpoints" / f"{name}_coo.pt"
    torch.save({"sparse_tensors": tensors, "dense_state": state}, path)
    # Round trip checks all tensors, including shape and index accounting.
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    for n, m in eligible(model).items():
        record = loaded["sparse_tensors"][n + ".weight"]
        restored = torch.sparse_coo_tensor(
            record["indices"], record["values"], record["shape"]
        ).to_dense()
        assert torch.equal(restored, m.weight.detach().cpu())
    return path.stat().st_size / 1e6


# Operation counts and hardware profiling
# ----------------------------------------------------------------------------
@torch.inference_mode()
def operation_counts(model):
    """Count Conv/Linear MACs for one image and report FLOPs as twice MACs.

    The ideal nonzero count is theoretical; dense kernels still process zeros.
    """
    rows = []
    hooks = []

    def hook(name, m, x, y):
        if isinstance(m, nn.Conv2d):
            mac = y.numel() * m.weight[0].numel()
            nz = int((m.weight != 0).sum()) * y.shape[-2] * y.shape[-1]
        else:
            mac = m.in_features * m.out_features
            nz = int((m.weight != 0).sum())
        rows.append({"layer": name, "dense_MACs": mac, "ideal_nonzero_MACs": nz})

    for n, m in eligible(model).items():
        hooks.append(m.register_forward_hook(lambda m, x, y, n=n: hook(n, m, x, y)))
    model.eval()(torch.zeros(1, 3, 32, 32, device=DEVICE))
    for h in hooks:
        h.remove()
    return {
        "MACs": sum(r["dense_MACs"] for r in rows),
        "FLOPs": 2 * sum(r["dense_MACs"] for r in rows),
        "ideal_nonzero_MACs": sum(r["ideal_nonzero_MACs"] for r in rows),
    }


class PowerSampler:
    """Sample NVML whole-board GPU power in a background thread.

    Integrating the power samples estimates energy for a measured interval.
    Unavailable hardware readings are recorded as unavailable rather than zero.
    """

    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.handle = None
        self.error = None
        try:
            import pynvml

            self.nv = pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(self.handle)
        except Exception as e:
            self.error = str(e)

    def read(self):
        if self.handle is None or self.error:
            return
        try:
            self.samples.append(
                (
                    time.perf_counter(),
                    self.nv.nvmlDeviceGetPowerUsage(self.handle) / 1000,
                )
            )
        except Exception as e:
            self.error = str(e)

    def start(self):
        self.read()

        def worker():
            while not self.stop.wait(0.02):
                self.read()

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.read()
        if len(self.samples) < 2 or self.error:
            return None
        return float(
            np.trapezoid([x[1] for x in self.samples], x=[x[0] for x in self.samples])
        )


def rapl_read():
    """Read accessible CPU package-energy counters, returning an empty map if absent."""
    values = {}
    for path in Path("/sys/class/powercap").glob("intel-rapl:[0-9]*/energy_uj"):
        if path.parent.name.count(":") != 1:
            continue
        try:
            values[str(path)] = (
                int(path.read_text()),
                int((path.parent / "max_energy_range_uj").read_text()),
            )
        except OSError:
            pass
    return values


@torch.inference_mode()
def profile(model, name, counts):
    """Measure the unchanged latency, memory, and energy protocol for one model.

    Uses 20 warm-up forwards, 100 timed batches, and three energy windows.
    This function performs inference and writes profiling result files.
    """
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    xs = []
    for i, (x, y) in enumerate(TEST.batches()):
        if i == 10:
            break
        xs.append(x)
    for i in range(20):
        model(xs[i % 10])
    torch.cuda.synchronize()
    baseline_mem = torch.cuda.memory_allocated()
    samples = []
    mem = []
    peaks = []
    for i in range(100):
        torch.cuda.reset_peak_memory_stats()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        out = model(xs[i % 10])
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b))
        mem.append(torch.cuda.memory_allocated())
        peaks.append(torch.cuda.max_memory_allocated())
        del out
    # Three independent >=2-second windows improve board-energy integration.
    sampler = PowerSampler()
    energy_rows = []
    for repeat in range(3):
        sampler = PowerSampler()
        cpu_before = rapl_read()
        sampler.start()
        start = time.perf_counter()
        n = 0
        while time.perf_counter() - start < 2:
            model(xs[n % 10])
            torch.cuda.synchronize()
            n += 1
        elapsed = time.perf_counter() - start
        gpu_j = sampler.finish()
        cpu_after = rapl_read()
        cpu_j = None
        if cpu_before and cpu_before.keys() == cpu_after.keys():
            cpu_j = (
                sum((cpu_after[k][0] - v[0]) % v[1] for k, v in cpu_before.items())
                / 1e6
            )
        energy_rows.append(
            {
                "batches": n,
                "seconds": elapsed,
                "gpu_J": gpu_j,
                "cpu_J": cpu_j,
                "gpu_J_per_batch": None if gpu_j is None else gpu_j / n,
                "cpu_J_per_batch": None if cpu_j is None else cpu_j / n,
            }
        )
    pd.DataFrame(
        {
            "latency_ms": samples,
            "allocated_MB": np.array(mem) / 1e6,
            "peak_MB": np.array(peaks) / 1e6,
        }
    ).to_csv(ROOT / "results" / f"{name}_profile_batches.csv", index=False)
    save_json(f"{name}_energy.json", energy_rows)

    def avg(field):
        vals = [r[field] for r in energy_rows if r[field] is not None]
        return float(np.mean(vals)) if vals else None

    result = {
        **counts,
        "latency_ms": float(np.mean(samples)),
        "latency_std_ms": float(np.std(samples)),
        "peak_GPU_MB": max(peaks) / 1e6,
        "average_GPU_MB": float(np.mean(mem)) / 1e6,
        "incremental_peak_GPU_MB": (max(peaks) - baseline_mem) / 1e6,
        "gpu_J_per_batch": avg("gpu_J_per_batch"),
        "cpu_J_per_batch": avg("cpu_J_per_batch"),
        "energy_note": "GPU: integrated NVML board power, not process-isolated. CPU: RAPL package counter when exposed; otherwise unavailable.",
        "gpu_power_error": sampler.error,
        "profile_batch_size": BATCH,
        "profile_batches": 100,
    }
    return result


def assess(model, name, masks=None, sparse=True):
    """Evaluate, save, verify, and profile a final model and its COO variant.

    This writes measured result files. Final inference-only profiles are
    subsequently produced by final_validation using the same saved checkpoints.
    """
    print("Evaluating and profiling", name, flush=True)
    metrics = full_metrics(model)
    counts = operation_counts(model)
    metrics["parameters"] = sum(p.numel() for p in model.parameters())
    path = ROOT / "checkpoints" / f"{name}_dense.pt"
    torch.save(cpu_state(model), path)
    metrics["dense_MB"] = path.stat().st_size / 1e6
    if masks is not None:
        rows = verify_masks(model, masks)
        pd.DataFrame(rows).to_csv(ROOT / "results" / f"{name}_layers.csv", index=False)
        metrics["overall_sparsity"] = sum(
            int((~x).sum()) for x in masks.values()
        ) / sum(x.numel() for x in masks.values())
        metrics["sparse_MB"] = save_sparse(model, masks, name)
    else:
        metrics["overall_sparsity"] = 0.0
    metrics.update(profile(model, name, counts))
    if sparse and masks is not None:
        sparse_m = coo_model(model)
        x, _ = next(TEST.batches())
        with torch.inference_mode():
            a = model.eval()(x)
            b = sparse_m(x)
            error = float((a - b).abs().max())
            torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
        sparse_metrics = full_metrics(sparse_m)
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        sparse_metrics.update(profile(sparse_m, name + "_coo", counts))
        sparse_metrics["max_logit_difference"] = error
        save_json(name + "_coo_metrics.json", sparse_metrics)
        del sparse_m
        model.to(DEVICE)
    save_json(name + "_metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


# Training and fine-tuning
# ----------------------------------------------------------------------------
def optimizer_for(model, lr):
    """Construct the original SGD optimizer with the supplied learning rate."""
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )


def train_epoch(model, optimizer, epoch, masks=None):
    """Train one epoch, masking gradients and restoring masked weights after updates.

    Returns augmented-training loss and Top-1 accuracy for the learning curves.
    """
    model.train()
    n = 0
    correct = 0
    loss_sum = 0.0
    for x, y in TRAIN.batches(epoch):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        if masks is not None:
            for name, m in eligible(model).items():
                m.weight.grad.mul_(masks[name])
        optimizer.step()
        if masks is not None:
            apply_masks(model, masks, optimizer)
        n += len(y)
        correct += (out.argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
    return {"train_loss": loss_sum / n, "train_top1": 100 * correct / n}


def curves(history, name):
    """Save recorded epoch history and draw its loss and accuracy curves."""
    df = pd.DataFrame(history)
    df.to_csv(ROOT / "results" / f"{name}_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(
        axes, ["loss", "top1"], ["Cross-entropy loss", "Top-1 accuracy (%)"]
    ):
        marker = "o" if len(df) == 1 else None
        ax.plot(
            df.epoch, df["train_" + metric], label="Training (augmented)", marker=marker
        )
        ax.plot(df.epoch, df["val_" + metric], label="Validation", marker=marker)
        if len(df) == 1:
            ax.set_xticks(df.epoch)
        ax.set(xlabel="Epoch", ylabel=label, title=name.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{name}_curves.png", dpi=160)
    plt.close(fig)


def finetune(model, masks, name, epochs=FT_EPOCHS):
    """Run the original SGD fine-tuning schedule and save histories and checkpoints.

    Local and global pruning receive the same seed, schedule, and epoch budget.
    """
    seed_all()
    optimizer = optimizer_for(model, 0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.0001
    )
    history = []
    for epoch in range(epochs):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        row = train_epoch(model, optimizer, epoch, masks)
        val = evaluate(model, VALID)
        row.update(
            epoch=epoch + 1,
            lr=lr,
            val_loss=val["loss"],
            val_top1=val["top1"],
            seconds=time.perf_counter() - start,
        )
        history.append(row)
        scheduler.step()
        curves(history, name)
        torch.save(
            {
                "state": cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "masks": (
                    None if masks is None else {k: v.cpu() for k, v in masks.items()}
                ),
            },
            ROOT / "checkpoints" / f"{name}_resume.pt",
        )
        print(name, row, flush=True)
    if masks is not None:
        verify_masks(model, masks)
    return history


# Task 0: dense pretrained baseline
# ----------------------------------------------------------------------------
def task0():
    """Restore the pretrained baseline, evaluate and profile it, then release GPU state."""
    model = fresh(DENSE)
    result = assess(model, "baseline", sparse=False)
    assert (
        result["test_top1"] > 85
    ), "Checkpoint or preprocessing requires investigation."
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# Task 1: sensitivity and post-training magnitude pruning
# ----------------------------------------------------------------------------
def sensitivity():
    """Test one layer at a time at all seven required sparsity levels.

    Restore the complete pretrained state before every trial. Other layers
    remain dense, and no fine-tuning occurs during the 147 sensitivity trials.
    """
    model = fresh(DENSE)
    rows = []
    for name, layer in eligible(model).items():
        for sparsity in [0, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9]:
            model.load_state_dict(DENSE, strict=True)
            with torch.no_grad():
                w = layer.weight.flatten()
                k = round(w.numel() * sparsity)
                ids = torch.argsort(w.abs(), stable=True)[:k]
                w[ids] = 0
            metrics = evaluate(model, VALID)
            rows.append({"layer": name, "sparsity": sparsity, **metrics})
        print(
            "Sensitivity",
            name,
            [(r["sparsity"], round(r["top1"], 2)) for r in rows[-7:]],
            flush=True,
        )
        pd.DataFrame(rows).to_csv(ROOT / "results" / "sensitivity.csv", index=False)
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(7, 3, figsize=(14, 23), sharex=True, sharey=True)
    for ax, (name, group) in zip(axes.flat, df.groupby("layer", sort=False)):
        ax.plot(group.sparsity * 100, group.top1, "o-")
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.set(xlabel="Pruned weights (%)", ylabel="Validation Top-1 (%)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "sensitivity.png", dpi=130)
    plt.close(fig)
    del model
    return df


def local_masks(model, df):
    """Allocate the same overall weight-removal budget using sensitivity results.

    A larger validation drop at 70 percent makes a layer less prunable.
    The original bounded allocation and rounding rule are preserved exactly.
    """
    names = list(eligible(model))
    sizes = np.array([m.weight.numel() for m in eligible(model).values()])
    drops = []
    for name in names:
        part = df[df.layer == name]
        base = float(part[part.sparsity == 0].top1.iloc[0])
        drops.append(max(0.0, base - float(part[part.sparsity == 0.7].top1.iloc[0])))
    # Continuous sensitivity-weighted water filling, exact rounded global budget.
    importance = 1 / (1 + np.asarray(drops) / 5)
    low, high = 0.0, 100.0
    for _ in range(80):
        mid = (low + high) / 2
        r = np.clip(mid * importance, 0.05, 0.90)
        if (r * sizes).sum() / sizes.sum() < TARGET:
            low = mid
        else:
            high = mid
    desired = r * sizes
    counts = np.floor(desired).astype(int)
    remain = round(TARGET * int(sizes.sum())) - int(counts.sum())
    for i in np.argsort(-(desired - counts))[:remain]:
        counts[i] += 1
    assert counts.sum() == round(TARGET * int(sizes.sum()))
    masks = masks_for(model)
    for (name, layer), count in zip(eligible(model).items(), counts):
        flat = masks[name].flatten()
        flat[
            torch.argsort(layer.weight.detach().abs().flatten(), stable=True)[:count]
        ] = False
    apply_masks(model, masks)
    save_json(
        "local_allocation.json",
        [
            {"layer": n, "drop_at_70_pp": d, "chosen_sparsity": int(k) / int(z)}
            for n, d, k, z in zip(names, drops, counts, sizes)
        ],
    )
    return masks


def task1():
    """Run sensitivity analysis, then compare local and global magnitude pruning.

    Each method starts from the same pretrained checkpoint and is evaluated
    before and after identical fixed-mask fine-tuning.
    """
    df = sensitivity()
    results = {}
    for method in ["local", "global"]:
        model = fresh(DENSE)
        masks = (
            local_masks(model, df)
            if method == "local"
            else prune_global(model, masks_for(model), TARGET)
        )
        assert all(bool(mask.any()) for mask in masks.values()), "A layer collapsed."
        pre = full_metrics(model)
        save_json(method + "_before_finetune.json", pre)
        print(method, "before fine-tuning", pre, flush=True)
        finetune(model, masks, method)
        results[method] = assess(model, method, masks)
        del model, masks
        gc.collect()
        torch.cuda.empty_cache()
    assert (
        abs(
            results["local"]["overall_sparsity"] - results["global"]["overall_sparsity"]
        )
        < 1e-12
    )
    return results


# Task 2: iterative magnitude and signed GraSP pruning
# ----------------------------------------------------------------------------
def grasp_scores(model, masks):
    """Compute signed GraSP scores using two autograd passes, without forming a Hessian.

    gsum is the sample-weighted mean loss gradient; hv is its Hessian-vector
    product. The score is minus the weight times the corresponding HVP entry.
    Temperature, calibration weighting, BatchNorm mode, and ranking are unchanged.
    """
    model.eval()  # Fixed BatchNorm statistics; avoids changing state during criterion.
    params = list(model.parameters())
    total = sum(len(y) for x, y in CALIB.batches())
    gsum = [torch.zeros_like(p) for p in params]
    for x, y in CALIB.batches():
        loss = F.cross_entropy(model(x) / 200.0, y)
        grads = torch.autograd.grad(loss, params)
        for acc, g in zip(gsum, grads):
            acc.add_(g.detach(), alpha=len(y) / total)
    # In later stages the effective parameter is M*theta. Dead coordinates do
    # not contribute to the gradient norm or the HVP direction.
    mask_lookup = {id(m.weight): masks[n] for n, m in eligible(model).items()}
    for p, g in zip(params, gsum):
        if id(p) in mask_lookup:
            g.mul_(mask_lookup[id(p)])
    hv = [torch.zeros_like(p) for p in params]
    for x, y in CALIB.batches():
        loss = F.cross_entropy(model(x) / 200.0, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * v).sum() for g, v in zip(grads, gsum))
        hs = torch.autograd.grad(dot, params)
        for acc, h in zip(hv, hs):
            acc.add_(h.detach(), alpha=len(y) / total)
    lookup = {id(p): h for p, h in zip(params, hv)}
    scores = {
        n: -m.weight.detach() * lookup[id(m.weight)] for n, m in eligible(model).items()
    }
    assert all(torch.isfinite(v).all() for v in scores.values())
    return scores


def verify_grasp_math():
    """Check the HVP on a small analytical example and verify signed-score ranking.

    The model Hessian is never constructed; only this small check uses one.
    """
    theta = torch.tensor([0.4, -0.7, 0.2], dtype=torch.float64, requires_grad=True)

    def loss(t):
        return (t.sin().sum()) ** 2 + 0.3 * (t * t).sum()

    g = torch.autograd.grad(loss(theta), theta, create_graph=True)[0]
    hg = torch.autograd.grad((g * g.detach()).sum(), theta)[0]
    explicit = torch.autograd.functional.hessian(loss, theta) @ g.detach()
    torch.testing.assert_close(hg, explicit)
    # Signed scores: -3 survives while +4 is removed. No absolute value.
    test = torch.tensor([-3.0, -0.5, 1.0, 4.0])
    removed = torch.argsort(test, descending=True)[:2]
    assert set(removed.tolist()) == {2, 3}
    result = {
        "hvp_matches_toy_hessian": True,
        "largest_signed_scores_removed": True,
        "max_hvp_error": float((hg - explicit).abs().max()),
    }
    save_json("grasp_verification.json", result)
    print(result, flush=True)


def task2():
    """Compare iterative magnitude and GraSP from a common random initialization.

    Save a shared warm-up state, then run each method for the same total budget.
    Masks accumulate at the recorded 20/40/60 percent accuracy thresholds,
    with the original epoch-granularity behavior and fallback deadlines.
    """
    verify_grasp_math()
    seed_all()
    model = fresh()
    initial = cpu_state(model)
    torch.save(initial, ROOT / "checkpoints" / "random_initial.pt")
    optimizer = optimizer_for(model, 0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, SCRATCH_EPOCHS, eta_min=0.001
    )
    warm_history = []
    for epoch in range(5):
        row = train_epoch(model, optimizer, epoch)
        val = evaluate(model, VALID)
        row.update(
            epoch=epoch + 1,
            lr=optimizer.param_groups[0]["lr"],
            val_loss=val["loss"],
            val_top1=val["top1"],
        )
        warm_history.append(row)
        scheduler.step()
        print("Warm-up", row, flush=True)
        if val["top1"] >= 20:
            break
    warm = cpu_state(model)
    warm_opt = copy.deepcopy(optimizer.state_dict())
    warm_sched = copy.deepcopy(scheduler.state_dict())
    torch.save(
        {
            "initial": initial,
            "state": warm,
            "optimizer": warm_opt,
            "scheduler": warm_sched,
            "history": warm_history,
        },
        ROOT / "checkpoints" / "warmup.pt",
    )
    curves(warm_history, "warmup")
    del model, optimizer, scheduler
    stages = []
    results = {}
    for method in ["iterative_magnitude", "iterative_grasp"]:
        seed_all()
        model = fresh(warm)
        masks = masks_for(model)
        assert all(torch.equal(v.cpu(), warm[k]) for k, v in model.state_dict().items())
        optimizer = optimizer_for(model, 0.1)
        optimizer.load_state_dict(copy.deepcopy(warm_opt))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, SCRATCH_EPOCHS, eta_min=0.001
        )
        scheduler.load_state_dict(copy.deepcopy(warm_sched))
        history = copy.deepcopy(warm_history)
        stage = 0
        val = evaluate(model, VALID)
        for epoch in range(len(warm_history), SCRATCH_EPOCHS):
            thresholds = [20, 40, 60]
            deadlines = [5, 12, 25]
            if stage < 3 and (
                val["top1"] >= thresholds[stage] or epoch >= deadlines[stage]
            ):
                before = evaluate(model, VALID)
                before_test = evaluate(model, TEST)
                old = {k: v.clone() for k, v in masks.items()}
                torch.cuda.synchronize()
                start = time.perf_counter()
                scores = (
                    grasp_scores(model, masks) if method == "iterative_grasp" else None
                )
                torch.cuda.synchronize()
                criterion_seconds = time.perf_counter() - start
                # Magnitude criterion timing includes explicitly computing absolute values.
                if scores is None:
                    start = time.perf_counter()
                    scores = {
                        n: m.weight.detach().abs() for n, m in eligible(model).items()
                    }
                    torch.cuda.synchronize()
                    criterion_seconds = time.perf_counter() - start
                stage_target = TARGET * [0.5, 0.75, 1][stage]
                masks = prune_global(
                    model,
                    masks,
                    stage_target,
                    scores,
                    largest=method == "iterative_grasp",
                )
                apply_masks(model, masks, optimizer)
                assert all(not bool((masks[n] & ~old[n]).any()) for n in masks)
                after = evaluate(model, VALID)
                after_test = evaluate(model, TEST)
                row = {
                    "method": method,
                    "stage": stage + 1,
                    "epoch_before_training": epoch + 1,
                    "target": stage_target,
                    "before_val_top1": before["top1"],
                    "after_val_top1": after["top1"],
                    "before_test_top1": before_test["top1"],
                    "after_test_top1": after_test["top1"],
                    "sparsity": sum(int((~v).sum()) for v in masks.values())
                    / sum(v.numel() for v in masks.values()),
                    "criterion_seconds": criterion_seconds,
                    "trigger": (
                        "accuracy"
                        if before["top1"] >= thresholds[stage]
                        else "budget deadline"
                    ),
                    "collapsed_layers": [n for n, v in masks.items() if not v.any()],
                }
                stages.append(row)
                stage += 1
                print("PRUNING", row, flush=True)
                save_json("iterative_stages.json", stages)
                del scores, old
            start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            row = train_epoch(model, optimizer, epoch, masks)
            val = evaluate(model, VALID)
            row.update(
                epoch=epoch + 1,
                lr=lr,
                val_loss=val["loss"],
                val_top1=val["top1"],
                seconds=time.perf_counter() - start,
            )
            history.append(row)
            scheduler.step()
            curves(history, method)
            torch.save(
                {
                    "state": cpu_state(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "masks": {n: v.cpu() for n, v in masks.items()},
                    "stage": stage,
                    "epoch": epoch + 1,
                },
                ROOT / "checkpoints" / f"{method}_resume.pt",
            )
            print(method, row, flush=True)
        assert stage == 3
        results[method] = assess(model, method, masks)
        del model, masks, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    return results


# Task 3: regression-based channel pruning
# ----------------------------------------------------------------------------
def structured_plan(model):
    """Choose internal block widths that match the original overall removal target.

    Count both the preceding convolution's removed output filters and the
    next convolution's removed input filters. External residual widths stay fixed.
    """
    blocks = [
        (n, m)
        for n, m in model.named_modules()
        if hasattr(m, "conv2") and hasattr(m, "bn2")
    ][1:]
    total = sum(m.weight.numel() for m in eligible(model).values())

    def plan(retain):
        return {n: max(1, round(m.conv1.out_channels * retain)) for n, m in blocks}

    def removed(widths):
        return sum(
            (m.conv1.out_channels - widths[n])
            * (m.conv1.in_channels * 9 + m.conv2.out_channels * 9)
            for n, m in blocks
        )

    candidates = [plan(r) for r in np.linspace(0.05, 1, 10001)]
    widths = min(candidates, key=lambda p: abs(removed(p) / total - TARGET))
    return widths, removed(widths) / total


@torch.inference_mode()
def reconstruction_data(student, teacher, block_name):
    """Collect matched student-input and dense-teacher-output regression rows.

    X contains unfolded input patches. Y contains teacher outputs at the same
    image and spatial positions. The original sampling and row cap are unchanged.
    """
    block = student.get_submodule(block_name)
    reference = teacher.get_submodule(block_name)
    captured = {}
    xs = []
    ys = []
    h1 = block.conv2.register_forward_pre_hook(
        lambda m, x: captured.update(x=x[0].detach())
    )
    h2 = reference.conv2.register_forward_hook(
        lambda m, x, y: captured.update(y=y.detach())
    )
    splits = json.loads((ROOT / "results" / "split_indices.json").read_text())
    # 100 images per class, drawn only from the optimization split.
    ids = []
    labels = TRAIN.y.cpu().numpy()
    for c in range(10):
        pool = [i for i in splits["train"] if labels[i] == c]
        ids.extend(pool[:100])
    loader = TensorBatches(TRAIN.x, TRAIN.y, ids)
    generator = torch.Generator(device=DEVICE).manual_seed(
        SEED + sum(map(ord, block_name))
    )
    student.eval()
    teacher.eval()
    try:
        for images, _ in loader.batches():
            teacher(images)
            student(images)
            x = F.unfold(
                captured["x"],
                block.conv2.kernel_size,
                padding=block.conv2.padding,
                stride=block.conv2.stride,
                dilation=block.conv2.dilation,
            ).transpose(1, 2)
            y = captured["y"].flatten(2).transpose(1, 2)
            b, l, d = x.shape
            # At most eight spatial locations per image; identical rows in X and Y.
            loc = torch.randperm(l, generator=generator, device=DEVICE)[: min(8, l)]
            xs.append(x[:, loc].reshape(-1, d).cpu())
            ys.append(y[:, loc].reshape(-1, y.shape[-1]).cpu())
    finally:
        h1.remove()
        h2.remove()
    X = torch.cat(xs)
    Y = torch.cat(ys)
    # Bound CPU least-squares work without changing the sample pairing.
    order = torch.randperm(len(X), generator=torch.Generator().manual_seed(SEED))[:4096]
    return X[order], Y[order], ids


def lasso_channels(X, Y, weight, keep):
    """Select input channels with the original normalized LASSO path.

    Z stores per-channel output contributions. beta weights these contributions.
    Return the selected channel indices, coefficients, and solver diagnostics.
    """
    from sklearn.linear_model import Lasso
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    channels = weight.shape[1]
    kernel = weight.shape[2] * weight.shape[3]
    rng = np.random.default_rng(SEED)
    equations = min(16384, len(X) * Y.shape[1])
    rows = torch.from_numpy(rng.integers(len(X), size=equations))
    outputs = torch.from_numpy(rng.integers(Y.shape[1], size=equations))
    patches = X[rows].reshape(equations, channels, kernel)
    weights = weight.detach().cpu()[outputs].reshape(equations, channels, kernel)
    Z = (patches * weights).sum(-1).numpy().astype(np.float64)
    target = Y[rows, outputs].numpy().astype(np.float64)
    scale = np.sqrt(np.mean(Z * Z, axis=0))
    scale = np.maximum(scale, 1e-10)
    Z = np.asfortranarray(Z / scale)
    alpha_max = max(float(np.max(np.abs(Z.T @ target)) / len(target)), 1e-10)
    gram = Z.T @ Z
    solver = Lasso(
        fit_intercept=False,
        max_iter=3000,
        tol=1e-5,
        warm_start=True,
        selection="cyclic",
        precompute=gram,
    )
    records = []
    chosen = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for alpha in np.geomspace(alpha_max, alpha_max * 1e-6, 70):
            solver.alpha = alpha
            solver.fit(Z, target)
            count = int(np.count_nonzero(solver.coef_))
            records.append(
                {
                    "alpha": float(alpha),
                    "nonzero": count,
                    "iterations": int(solver.n_iter_),
                    "dual_gap": float(solver.dual_gap_),
                }
            )
            if count >= keep:
                chosen = solver.coef_.copy() / scale
                break
        warning_count = len(caught)
    if chosen is None:
        raise RuntimeError(
            "LASSO path did not provide enough nonzero channels; inspect calibration."
        )
    # Discrete support jumps can skip the exact requested width. Retain the largest
    # standardized coefficients at the first lambda with at least the target support.
    selected = np.sort(np.argsort(-np.abs(solver.coef_), kind="stable")[:keep])
    beta = chosen[selected]
    assert np.all(beta != 0)
    return (
        torch.tensor(selected, dtype=torch.long),
        torch.tensor(beta, dtype=torch.float64),
        {
            "lambda_path": records,
            "selected_lambda": records[-1]["alpha"],
            "lasso_support_before_exact_width": records[-1]["nonzero"],
            "selected_channels": selected.tolist(),
            "beta": beta.tolist(),
            "equations": equations,
            "convergence_warnings": warning_count,
        },
    )


def replace_hidden(block, selected, reconstructed):
    """Physically narrow a block's conv1, bn1, and conv2 internal connection.

    Copy the selected channels in consistent order and install the reconstructed
    conv2 weights. Both branches retain the same external residual dimensions.
    """
    device = block.conv1.weight.device
    selected = selected.to(device)
    old1, oldbn, old2 = block.conv1, block.bn1, block.conv2
    k = len(selected)
    conv1 = nn.Conv2d(
        old1.in_channels,
        k,
        old1.kernel_size,
        stride=old1.stride,
        padding=old1.padding,
        dilation=old1.dilation,
        bias=False,
    ).to(device)
    bn = nn.BatchNorm2d(k, eps=oldbn.eps, momentum=oldbn.momentum).to(device)
    conv2 = nn.Conv2d(
        k,
        old2.out_channels,
        old2.kernel_size,
        stride=old2.stride,
        padding=old2.padding,
        dilation=old2.dilation,
        bias=False,
    ).to(device)
    with torch.no_grad():
        conv1.weight.copy_(old1.weight[selected])
        conv2.weight.copy_(reconstructed.to(device))
        for attr in ["weight", "bias", "running_mean", "running_var"]:
            getattr(bn, attr).copy_(getattr(oldbn, attr)[selected])
        bn.num_batches_tracked.copy_(oldbn.num_batches_tracked)
    block.conv1, block.bn1, block.conv2 = conv1, bn, conv2
    # The branch and shortcut still have the same external width. No channel in
    # the residual addition is removed or permuted.
    assert block.conv2.out_channels == block.bn2.num_features


def task3():
    """Prune block interiors sequentially, reconstruct weights, and fine-tune.

    Xprime is the beta-scaled reduced design matrix. Wprime is the least-squares
    solution. folded absorbs beta into the convolution weights for dense inference.
    """
    seed_all()
    teacher = fresh(DENSE).eval()
    student = fresh(DENSE).eval()
    widths, expected = structured_plan(student)
    original_weights = sum(m.weight.numel() for m in eligible(student).values())
    rows = []
    details = {}
    start = time.perf_counter()
    for name, keep in widths.items():
        block = student.get_submodule(name)
        old_width = block.conv1.out_channels
        before = evaluate(student, VALID)["top1"]
        t = time.perf_counter()
        X, Y, ids = reconstruction_data(student, teacher, name)
        selected, beta, record = lasso_channels(X, Y, block.conv2.weight, keep)
        kernel = block.conv2.weight.shape[2] * block.conv2.weight.shape[3]
        Xselected = X.reshape(len(X), old_width, kernel)[:, selected].double()
        Xprime = (Xselected * beta[None, :, None]).reshape(len(X), -1)
        # Pivoted QR handles rank deficiency. The solve explicitly uses beta-scaled X.
        solution = torch.linalg.lstsq(Xprime, Y.double(), driver="gelsy", rcond=1e-8)
        Wprime = solution.solution
        folded = (Wprime.reshape(keep, kernel, -1) * beta[:, None, None]).reshape(
            keep * kernel, -1
        )
        before_mse = float(
            F.mse_loss(
                X.double() @ block.conv2.weight.detach().cpu().double().flatten(1).T,
                Y.double(),
            )
        )
        mse = float(F.mse_loss(Xselected.reshape(len(X), -1) @ folded, Y.double()))
        reconstructed = folded.T.reshape(
            block.conv2.out_channels, keep, *block.conv2.kernel_size
        ).float()
        replace_hidden(block, selected, reconstructed)
        student.eval()
        after = evaluate(student, VALID)["top1"]
        row = {
            "block": name,
            "old_hidden_channels": old_width,
            "new_hidden_channels": keep,
            "removed_hidden_fraction": 1 - keep / old_width,
            "calibration_images": len(ids),
            "regression_rows": len(X),
            "regression_columns": Xprime.shape[1],
            "regression_rank": int(solution.rank),
            "before_reconstruction_MSE": before_mse,
            "after_reconstruction_MSE": mse,
            "before_val_top1": before,
            "after_val_top1": after,
            "seconds": time.perf_counter() - t,
        }
        rows.append(row)
        details[name] = record
        save_json("structured_lasso.json", details)
        pd.DataFrame(rows).to_csv(
            ROOT / "results" / "structured_stages.csv", index=False
        )
        print("CHANNEL PRUNING", row, flush=True)
        del X, Y, Xselected, Xprime, solution, Wprime, folded, reconstructed
    retained_weights = sum(m.weight.numel() for m in eligible(student).values())
    actual = 1 - retained_weights / original_weights
    assert abs(actual - expected) < 1e-12 and abs(actual - TARGET) < 0.005
    save_json(
        "structured_architecture.json",
        {
            "hidden_widths": widths,
            "weight_reduction": actual,
            "target": TARGET,
            "original_eligible_weights": original_weights,
            "remaining_eligible_weights": retained_weights,
            "pruning_seconds": time.perf_counter() - start,
            "shortcut_policy": "Keep all external branch and shortcut channels unchanged.",
        },
    )
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    pre = full_metrics(student)
    save_json("structured_before_finetune.json", pre)
    print(pre, flush=True)
    finetune(student, None, "structured", epochs=STRUCT_EPOCHS)
    result = assess(student, "structured", sparse=False)
    result["overall_sparsity"] = actual
    result["sparsity_definition"] = (
        "Physical Conv/Linear weight-count reduction relative to original; retained dense tensors need not contain zeros."
    )
    save_json("structured_metrics.json", result)
    del student
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_structured():
    """Rebuild saved hidden widths and load the final structured checkpoint."""
    model = fresh(DENSE)
    spec = json.loads((ROOT / "results" / "structured_architecture.json").read_text())
    for name, k in spec["hidden_widths"].items():
        block = model.get_submodule(name)
        replace_hidden(
            block, torch.arange(k), torch.zeros(block.conv2.out_channels, k, 3, 3)
        )
    model.load_state_dict(
        torch.load(
            ROOT / "checkpoints" / "structured_dense.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    return model.eval()


# Final checkpoint validation and inference-only profiling
# ----------------------------------------------------------------------------
def final_validation():
    """Run the original checkpoint checks and clean inference-only profiling pass.

    This function performs inference measurements and writes the final profiling files.
    """
    names = [
        "baseline",
        "local",
        "global",
        "iterative_magnitude",
        "iterative_grasp",
        "structured",
    ]
    audit = {
        "checkpoint_roundtrips": {},
        "all_final_mask_checks": True,
        "profiling_state": "No live optimizer, gradients, or GPU pruning masks.",
    }
    for name in names:
        gc.collect()
        torch.cuda.empty_cache()
        if name == "structured":
            model = load_structured()
        else:
            state = torch.load(
                ROOT / "checkpoints" / f"{name}_dense.pt",
                map_location="cpu",
                weights_only=True,
            )
            model = fresh(state)
            assert all(
                torch.equal(v.detach().cpu(), state[k])
                for k, v in model.state_dict().items()
            )
            del state
        model.zero_grad(set_to_none=True)
        model.eval()
        assert all(p.grad is None for p in model.parameters())
        counts = operation_counts(model)
        record = json.loads((ROOT / "results" / f"{name}_metrics.json").read_text())
        record.update(profile(model, name, counts))
        record["profiling_state"] = (
            "Inference-only checkpoint reload; no optimizer, gradients, or GPU masks."
        )
        save_json(name + "_metrics.json", record)
        audit["checkpoint_roundtrips"][name] = True
        if name in ["local", "global", "iterative_magnitude", "iterative_grasp"]:
            sparse_record = torch.load(
                ROOT / "checkpoints" / f"{name}_coo.pt",
                map_location="cpu",
                weights_only=True,
            )
            for n, m in eligible(model).items():
                t = sparse_record["sparse_tensors"][n + ".weight"]
                restored = (
                    torch.sparse_coo_tensor(
                        t["indices"], t["values"], t["shape"], check_invariants=True
                    )
                    .coalesce()
                    .to_dense()
                )
                assert torch.equal(restored, m.weight.detach().cpu())
                assert (restored[~t["mask"]] == 0).all()
            del sparse_record, restored, t
            sparse_model = coo_model(model)
            sample, _ = next(TEST.batches())
            with torch.inference_mode():
                torch.testing.assert_close(
                    model(sample), sparse_model(sample), atol=2e-4, rtol=2e-4
                )
            del sample
            model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            sparse_metrics = json.loads(
                (ROOT / "results" / f"{name}_coo_metrics.json").read_text()
            )
            sparse_metrics.update(profile(sparse_model, name + "_coo", counts))
            sparse_metrics["profiling_state"] = (
                "COO inference-only reload; dense source held on CPU."
            )
            save_json(name + "_coo_metrics.json", sparse_metrics)
            del sparse_model
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(
            "FINAL VALIDATION",
            name,
            "passed;",
            round(record["latency_ms"], 4),
            "ms/batch;",
            round(record["peak_GPU_MB"], 3),
            "MB peak",
            flush=True,
        )
    a = json.loads((ROOT / "results" / "local_metrics.json").read_text())
    b = json.loads((ROOT / "results" / "global_metrics.json").read_text())
    assert a["overall_sparsity"] == b["overall_sparsity"]
    for name in ["iterative_magnitude", "iterative_grasp"]:
        item = json.loads((ROOT / "results" / f"{name}_metrics.json").read_text())
        assert abs(item["overall_sparsity"] - a["overall_sparsity"]) < 1e-12
    audit["equal_unstructured_budgets"] = True
    audit["sensitivity_trials"] = len(pd.read_csv(ROOT / "results" / "sensitivity.csv"))
    assert audit["sensitivity_trials"] == 21 * 7
    for name in ["local", "global", "iterative_magnitude", "iterative_grasp"]:
        layers = pd.read_csv(ROOT / "results" / f"{name}_layers.csv")
        assert layers.masked_positions_zero.all()
    save_json("final_validation.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    return audit
