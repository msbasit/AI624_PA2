"""Create the Markdown report and figures exclusively from recorded experiment files."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = [
    "baseline",
    "local",
    "global",
    "iterative_magnitude",
    "iterative_grasp",
    "structured",
]
LABELS = {
    "baseline": "Dense baseline",
    "local": "Local magnitude",
    "global": "Global magnitude",
    "iterative_magnitude": "Iterative magnitude",
    "iterative_grasp": "Iterative GraSP",
    "structured": "Structured channels",
}


def generate_report(root):
    root = Path(root)
    results = root / "results"
    figures = root / "figures"

    def read(name):
        return json.loads((results / name).read_text(encoding="utf-8"))

    def table(rows):
        return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")

    metrics = {n: read(n + "_metrics.json") for n in NAMES}
    coo = {n: read(n + "_coo_metrics.json") for n in NAMES[1:5]}
    env = read("environment.json")
    base = metrics["baseline"]
    parts = []

    def add(text):
        parts.append(text.strip())

    def image(name, caption):
        add(f"![{caption}](figures/{name})\n\n{caption}")

    add("# AI624 Assignment 2: Pruning ResNet-18 on CIFAR-10")
    add(
        "## Authorship and disclosure\n\nI have used OpenAI Codex to write original experiment code, explain pruning methods, execute the experiments, debug the implementation, create plots, and draft the report 1 time. Here, one time means one continuous assignment assistance session, including iterative tool calls and corrections. The measured results were produced by executing the supplied code. The model architecture and pretrained weights are attributed to Huy Phan's public repository; the pruning methods are attributed to their research papers. This is an AI-assisted submission, and does not claim that AI-generated work was independently handwritten."
    )
    add("## Experimental design and reproducibility")
    add(
        f"All final experiments use one the GPU environment {env['gpu']} session, FP32, batch size {env['batch_size']}, and 3 x 32 x 32 RGB inputs. The requested final pruning budget is 70% of Conv2d and Linear weights. Biases and normalization parameters are excluded from that denominator. Model parameter counts include all trainable tensors. Tasks 0, 1, and 3 start from exactly the same checkpoint; Task 2 starts from a separately saved random initialization. The random seed is {env['seed']}."
    )
    add(
        table(
            [
                {"Setting": k, "Value": str(env[k])}
                for k in [
                    "python",
                    "torch",
                    "torchvision",
                    "cuda",
                    "cudnn",
                    "gpu",
                    "cpu",
                    "gpu_total_MB",
                    "repository_commit",
                    "checkpoint_sha256",
                ]
            ]
        )
    )
    add(
        "The upstream architecture is loaded directly from `cifar10_models/resnet.py`, including its stem max-pooling operation. Loading uses strict key and shape checks and verifies every state tensor against the checkpoint. The Box weights URL returned 404 during setup, so the archive came from the official Google Drive backup linked in the repository README. CIFAR-10 came from a byte-identical mirror because the original host was unusually slow. Torchvision checks the official archive and extracted batch MD5 hashes before use. The expected archive MD5 is `c58f30108f718f92721af3b95e74349a`."
    )
    add(
        "### Data and preprocessing\n\nThe official 50,000 training images are divided by a seeded, class-stratified split into 45,000 optimization and 5,000 validation images. The official 10,000-image test set remains separate. The train metrics in final tables evaluate all 50,000 official training images without augmentation; the epoch curves evaluate the 45,000 augmented optimization images. Therefore these two training accuracy quantities need not match. The validation split was not used for gradient updates here, but the supplied pretrained checkpoint was originally trained on the original training set: validation results in Tasks 1 and 3 are not an unseen-pretraining generalization estimate. Final test results are the main generalization comparison."
    )
    add(
        "Each image is divided by 255 and normalized with channel means `(0.4914, 0.4822, 0.4465)` and standard deviations `(0.2471, 0.2435, 0.2616)`. Training uses four-pixel constant padding, a random 32 x 32 crop, and horizontal flipping with probability 0.5. Validation, calibration, profiling, and final evaluation use normalization only. Images are cached as uint8 tensors on the GPU; the deterministic batch iterator shuffles and augments with seed plus epoch, avoiding CPU transfer bottlenecks. The last incomplete evaluation batch is included. Profiling always uses complete batches of 128."
    )
    add(
        "### Metric definitions\n\nTop-1 and Top-5 are percentages. Macro-F1 is on a 0 to 1 scale and gives each class equal weight. With class confusion counts,\n\n$$F_{1,c}=\\frac{2TP_c}{2TP_c+FP_c+FN_c}, \\qquad F_{1,\\mathrm{macro}}=\\frac{1}{10}\\sum_{c=1}^{10}F_{1,c}.$$\n\nA convolution uses $H_oW_oC_o(C_i/g)k_hk_w$ MACs per image; a linear layer uses $d_{in}d_{out}$. We report $\\mathrm{FLOPs}=2\\,\\mathrm{MACs}$, counting the multiply and add separately. This is an analytical Conv/Linear count, excluding BatchNorm, activation, pooling, bias, data preparation, and sparse conversion operations. The `ideal_nonzero_MACs` field is only a theoretical zero-skipping count and is not an executed operation count. Serialized MB means decimal bytes divided by $10^6$."
    )
    add(
        "### Common latency, memory, and energy procedure\n\nAll models are evaluated in inference mode with the same ten normalized GPU-resident test batches. After 20 warm-up forwards and synchronization, 100 forward latencies are measured using CUDA events with synchronization at the end of each forward. Mean and standard deviation are saved. PyTorch allocated GPU memory is sampled after each forward; peak allocation is reset before each measured batch, and the largest batch peak is reported. Average memory is the arithmetic mean of post-forward allocations, not a continuous time average. Final profiling reloads each checkpoint after training has finished, with no live optimizer, parameter gradients, or GPU pruning masks. The common dataset cache contributes to absolute allocation. Incremental forward peak is also saved; these measurements are not whole-device NVML memory usage. All final metric files and batch CSVs contain this clean inference-only profiling pass."
    )
    add(
        "GPU energy integrates NVML whole-board power, sampled approximately every 20 ms, during three independent inference windows of at least two seconds. Each window is divided by its actual completed batch count; the table gives the mean joules per batch. These are board-energy estimates, include idle board power and synchronization overhead, and are not process-isolated or wall-socket measurements. CPU energy uses package-level RAPL counters if accessible; unavailable counters are recorded as `N/A`, never zero. The execution environment did not expose host CPU power counters to the guest. CPU energy cannot be measured reliably from these guest permissions; this requested metric is explicitly reported as unavailable rather than invented. Raw per-batch timing/memory CSVs and per-window energy JSONs are included."
    )
    add("## Task 0: Dense model baseline")
    accuracy = []
    resources = []
    energy = []
    for n, m in metrics.items():
        accuracy.append(
            {
                "Model": LABELS[n],
                "Sparsity %": m["overall_sparsity"] * 100,
                "Train Top-1 %": m["train_top1"],
                "Test Top-1 %": m["test_top1"],
                "Train Top-5 %": m["train_top5"],
                "Test Top-5 %": m["test_top5"],
                "Train macro-F1": m["train_macro_f1"],
                "Test macro-F1": m["test_macro_f1"],
                "Drop vs dense (pp)": base["test_top1"] - m["test_top1"],
            }
        )
        resources.append(
            {
                "Model": LABELS[n],
                "Parameters": m["parameters"],
                "Dense MB": m["dense_MB"],
                "COO MB": m.get("sparse_MB", "N/A"),
                "MACs/image": m["MACs"],
                "FLOPs/image": m["FLOPs"],
                "Latency ms/batch": m["latency_ms"],
                "Peak GPU MB": m["peak_GPU_MB"],
                "Average GPU MB": m["average_GPU_MB"],
            }
        )
        energy.append(
            {
                "Model": LABELS[n],
                "GPU J/batch": (
                    m.get("gpu_J_per_batch")
                    if m.get("gpu_J_per_batch") is not None
                    else "N/A"
                ),
                "CPU J/batch": (
                    m.get("cpu_J_per_batch")
                    if m.get("cpu_J_per_batch") is not None
                    else "N/A"
                ),
                "Latency SD ms": m["latency_std_ms"],
                "Incremental peak MB": m["incremental_peak_GPU_MB"],
            }
        )
    add(
        f"The verified pretrained model achieved {base['test_top1']:.2f}% test Top-1 accuracy and {base['train_top1']:.2f}% training Top-1 accuracy. The tables below contain the baseline and every final model, using the same metrics and measurement definitions. 'Sparsity' for structured pruning means physical reduction of eligible weight count; it does not mean the remaining dense tensors contain that fraction of zeros."
    )
    add("### Accuracy comparison\n\n" + table(accuracy))
    add("### Model size and inference resources\n\n" + table(resources))
    add("### Energy and measurement variability\n\n" + table(energy))
    image(
        "final_comparison.png",
        "Final accuracy, latency, and dense model size on the same T4 GPU.",
    )
    pd.DataFrame(accuracy).to_csv(results / "comparison_accuracy.csv", index=False)
    pd.DataFrame(resources).to_csv(results / "comparison_resources.csv", index=False)
    add("## Task 1: Unstructured post-training pruning")
    add(
        "### 1. Independent layer sensitivity\n\nEach of the 21 convolutional or linear weight tensors is pruned independently at 0%, 10%, 20%, 50%, 70%, 80%, and 90%. All state tensors are restored from the original checkpoint before each trial, and there is no fine-tuning in these trials. Accuracy is measured on the 5,000-image validation split. This prevents the test set from choosing layer allocations. Every curve is included below, and every numeric observation is in `results/sensitivity.csv`."
    )
    df = pd.read_csv(results / "sensitivity.csv")
    pivot = df.pivot(index="layer", columns="sparsity", values="top1")
    sensitivity = (pivot[0] - pivot[0.7]).sort_values(ascending=False)
    add(
        "Largest validation accuracy losses at 70% single-layer sparsity:\n\n"
        + table(
            [
                {"Layer": n, "Drop at 70% (pp)": v}
                for n, v in sensitivity.head(5).items()
            ]
        )
    )
    add(
        f"The most sensitive layer by this criterion is `{sensitivity.index[0]}`. Sensitivity measures the effect of deleting weights in isolation. It does not guarantee that combining several individually tolerable deletions will be harmless: downstream representations and residual branch interactions can amplify their joint effect."
    )
    image(
        "sensitivity.png",
        "Independent layer sensitivity; every trial restarts from the dense checkpoint.",
    )
    add(
        "### 2. Local allocation and 3. Global allocation\n\nFor layer $\\ell$, let $d_\\ell=\\max(0,a_{\\ell,0}-a_{\\ell,0.7})$ in percentage points. The local allocation uses\n\n$$q_\\ell=\\frac{1}{1+d_\\ell/5}, \\qquad s_\\ell=\\operatorname{clip}(\\alpha q_\\ell,0.05,0.90).$$\n\nA scalar water-filling search chooses $\\alpha$ so the parameter-weighted average is 70%. Integer counts use largest-remainder rounding to reach exactly the requested total. Sensitive layers consequently receive a lower pruning fraction. Each layer removes its smallest absolute weights. Global pruning ranks the concatenated absolute weights, removes the same exact total, and lets the individual layer fractions emerge. Stable flattened-index ordering breaks threshold ties deterministically. The global method therefore has one shared magnitude cutoff, apart from deterministic handling of equal values at that cutoff."
    )
    layer_frames = {n: pd.read_csv(results / f"{n}_layers.csv") for n in NAMES[1:5]}
    combined = []
    for i, r in layer_frames["local"].iterrows():
        gr = layer_frames["global"].iloc[i]
        combined.append(
            {
                "Layer": r.layer,
                "Weights": int(r.weights),
                "Local mask %": 100 * r.mask_sparsity,
                "Local actual zero %": 100 * r.actual_zero_fraction,
                "Global mask %": 100 * gr.mask_sparsity,
                "Global actual zero %": 100 * gr.actual_zero_fraction,
            }
        )
    add(table(combined))
    for n in ["local", "global"]:
        layers = layer_frames[n]
        assert layers.masked_positions_zero.all() and (layers.remaining_ratio > 0).all()
    ranked = layer_frames["global"].sort_values("mask_sparsity", ascending=False)
    add(
        f"Global pruning removed the largest fraction in `{ranked.iloc[0].layer}` ({100*ranked.iloc[0].mask_sparsity:.2f}%) and the smallest fraction in `{ranked.iloc[-1].layer}` ({100*ranked.iloc[-1].mask_sparsity:.2f}%). No layer was completely removed. This is a usable distribution because it preserves a nonempty path through every eligible layer and supports the measured recovery below. It is not a sensitivity guarantee: magnitude scales differ between layers and interact with normalization, so small absolute values alone do not prove low functional importance."
    )
    add(
        "### 4. Fine-tuning and 5. Mask verification\n\nBoth methods use SGD with Nesterov momentum 0.9, weight decay 0.0005, initial learning rate 0.01, 12 epochs, and cosine decay to 0.0001. Batch size, seed, batch order, and augmentations are identical. Masks remain fixed. Gradients and momentum buffers are also masked; after every optimizer step, $W \\leftarrow M\\odot W$ is enforced. Assertions check every masked coordinate, and the final layer table compares actual zeros with mask zeros. Extra natural zeros, if present, are recorded separately instead of being mistaken for an additional pruning decision."
    )
    recovery = []
    for n in ["local", "global"]:
        pre = read(n + "_before_finetune.json")
        post = metrics[n]
        recovery.append(
            {
                "Method": LABELS[n],
                "Before pruning test %": base["test_top1"],
                "After pruning test %": pre["test_top1"],
                "After FT test %": post["test_top1"],
                "Recovered (pp)": post["test_top1"] - pre["test_top1"],
                "Final drop (pp)": base["test_top1"] - post["test_top1"],
            }
        )
    add(table(recovery))
    for row in recovery:
        if row["Recovered (pp)"] < 0:
            add(
                f"For {row['Method'].lower()}, fine-tuning reduced test accuracy by {-row['Recovered (pp)']:.2f} percentage points relative to the immediate pruned checkpoint. Thus it did not recover accuracy in this case. The curves show the disruption and subsequent partial recovery under the shared learning-rate schedule. Updates to weights and BatchNorm statistics under augmentation can move an already accurate pretrained solution away from its useful operating point; the experiment does not isolate which factor caused the decrease. A smaller learning rate or longer schedule could be investigated on validation data, but the fixed protocol is retained here for a fair local/global comparison."
            )
    for n in ["local", "global"]:
        image(
            n + "_curves.png",
            LABELS[n] + ": training and validation loss and accuracy.",
        )
    add(
        "### 6. COO storage and 7. COO inference\n\nEach pruned tensor is converted with `to_sparse().coalesce()`. The saved file contains float32 nonzero values, int64 COO indices, original shapes, Boolean pruning masks, and unpruned state tensors. A load-and-reconstruct check verifies exact equality for every pruned tensor. COO convolutions explicitly call `to_dense()` during every forward because ordinary PyTorch Conv2d does not execute general sparse COO convolution. The final linear layer uses `torch.sparse.mm`. Output agreement with the dense masked representation is asserted before profiling, and full evaluation is repeated for COO execution."
    )
    add(
        table(
            [
                {
                    "Model": LABELS[n] + " COO",
                    "Test Top-1 %": m["test_top1"],
                    "Train Top-1 %": m["train_top1"],
                    "Test Top-5 %": m["test_top5"],
                    "Train Top-5 %": m["train_top5"],
                    "Test macro-F1": m["test_macro_f1"],
                    "Train macro-F1": m["train_macro_f1"],
                    "Latency ms": m["latency_ms"],
                    "Peak GPU MB": m["peak_GPU_MB"],
                    "Average GPU MB": m["average_GPU_MB"],
                    "GPU J/batch": (
                        m["gpu_J_per_batch"]
                        if m["gpu_J_per_batch"] is not None
                        else "N/A"
                    ),
                    "CPU J/batch": (
                        m["cpu_J_per_batch"]
                        if m["cpu_J_per_batch"] is not None
                        else "N/A"
                    ),
                }
                for n, m in coo.items()
            ]
        )
    )
    add(
        "For a four-dimensional convolution tensor, an entry uses approximately 4 bytes for its value and 4 x 8 bytes for its indices. At 70% sparsity, values and indices alone therefore cost about $0.3(4+32)=10.8$ bytes per original weight, versus 4 bytes in dense FP32; a stored Boolean mask adds about one more byte per original weight. The break-even sparsity is above $1-4/36=88.89\\%$ even before masks and serialization overhead. Consequently a 70% sparse, four-coordinate COO file can be larger than its dense file. The measured file sizes include that overhead. A flattened two-dimensional encoding, compressed indices, or a hardware-supported format would have different storage costs, but none is substituted silently here."
    )
    for n in ["local", "global"]:
        m = metrics[n]
        add(
            f"{LABELS[n]}: the dense state is {m['dense_MB']:.3f} MB and the actual COO package is {m['sparse_MB']:.3f} MB, a size ratio of {m['sparse_MB']/m['dense_MB']:.2f}. Dense masked latency is {m['latency_ms']:.3f} ms per batch versus {coo[n]['latency_ms']:.3f} ms for the COO execution path."
        )
    add(
        "Zero-valued weights do not reduce dense tensor dimensions, kernel launches, or the dense convolution loop bounds. The GPU executes the same dense kernels and nominal MAC count. COO-to-dense conversion adds reconstruction and allocation work, so sparse storage by itself is not an inference acceleration. Actual sparse speedups depend on kernel support, sparsity structure, batch size, and the cost of indexing and data movement."
    )
    add("## Task 2: Saliency-based iterative pruning")
    warm = pd.read_csv(results / "warmup_history.csv")
    stages = read("iterative_stages.json")
    add(
        f"### 1. Shared random initialization and warm-up\n\nThe model is created with `resnet18(pretrained=False)` and does not load the pretrained checkpoint. Its initial state is saved. One shared warm-up ran for {len(warm)} epoch(s) and reached {warm.iloc[-1].val_top1:.2f}% validation Top-1. Both methods restore byte-equal copies of this state, optimizer momentum, and scheduler position. The warm-up checkpoint contains the initial weights, warm-up weights, optimizer state, and curves. Each method receives 50 total training epochs including the shared warm-up; SGD uses initial learning rate 0.1, Nesterov momentum 0.9, weight decay 0.0005, and cosine decay to 0.001. This is a fixed-budget comparison, not a claim of full convergence."
    )
    image("warmup_curves.png", "Shared random-initialization warm-up.")
    add(
        "### 2. Calibration and 3. GraSP criterion\n\nCalibration uses 200 normalization-only optimization images, 20 per class. With batch size 128, there are two batches of 128 and 72 images. Batch means are weighted by their sample counts. BatchNorm statistics are fixed during criterion computation. A temperature of 200 follows the GraSP implementation convention and is used for scoring only; optimization uses ordinary cross-entropy without temperature scaling.\n\n$$L(\\theta)=\\sum_b\\frac{n_b}{200}L_b(\\theta),\\quad g=\\nabla L,\\quad Hg=\\nabla_\\theta\\left[(\\nabla L)^T\\operatorname{stopgrad}(g)\\right],\\quad S_i=-\\theta_i(Hg)_i.$$\n\nOne pass accumulates the detached mean gradient. A second pass accumulates Hessian-vector products with that common vector. This retains cross-batch terms, unlike averaging independent $H_bg_b$ products. Pruned coordinates are removed from the gradient-norm direction in later stages. Bias and normalization parameters participate in the loss derivatives but are never candidates for pruning. Autograd computes the HVP without materializing a model Hessian. A three-parameter toy test compares the HVP against an explicit Hessian solely as a numerical unit check; no network Hessian is constructed."
    )
    add(
        "The signed GraSP ranking removes the **largest** $S_i$ values. It does not remove the smallest values and does not take absolute values. For a deletion $\\Delta\\theta_i=-\\theta_i$, the first-order change in squared gradient norm is $2g^TH\\Delta\\theta=-2\\theta_i(Hg)_i=2S_i$. Preserving or increasing gradient flow therefore favors removing large scores under this convention. The direction is verified with a known signed-score example. This derivation also explains why an implementation that merely copies the magnitude-pruning sorting direction is wrong."
    )
    add(
        "### 4. Three stages and 5. Cumulative masks\n\nThe cumulative sparsities are 35%, 52.5%, and 70%, corresponding to $0.50s^\\star$, $0.75s^\\star$, and $s^\\star$. Stages trigger after epoch-level validation accuracy first reaches approximately 20%, 40%, and 60%; they can overshoot because validation is checked once per epoch. To guarantee the complete sparsity schedule within the fixed budget, fallback epoch deadlines are 5, 12, and 25 completed epochs. The table identifies the actual trigger; a deadline is not presented as reaching an accuracy threshold. Only currently retained weights can be removed, so the new mask is a subset of the old mask. The code asserts no regrowth and reapplies masks, including momentum masking, after every update."
    )
    add(
        table(
            [
                {
                    "Method": LABELS[r["method"]],
                    "Stage": r["stage"],
                    "Next epoch": r["epoch_before_training"],
                    "Before val %": r["before_val_top1"],
                    "After val %": r["after_val_top1"],
                    "Before test %": r["before_test_top1"],
                    "After test %": r["after_test_top1"],
                    "Sparsity %": 100 * r["sparsity"],
                    "Criterion seconds": r["criterion_seconds"],
                    "Trigger": r["trigger"],
                    "Collapsed layers": ", ".join(r["collapsed_layers"]) or "None",
                }
                for r in stages
            ]
        )
    )
    add(
        "Criterion time is synchronized wall time for score construction. It excludes the subsequent common sorting and mask-application operation, which is why it should not be read as total pruning time. Test accuracies immediately before and after stages are recorded for the requested comparison; stage decisions use validation accuracy only."
    )
    ratios = []
    for i, r in layer_frames["iterative_magnitude"].iterrows():
        ratios.append(
            {
                "Layer": r.layer,
                "Local remaining": layer_frames["local"].iloc[i].remaining_ratio,
                "Global remaining": layer_frames["global"].iloc[i].remaining_ratio,
                "Iterative magnitude remaining": r.remaining_ratio,
                "Iterative GraSP remaining": layer_frames["iterative_grasp"]
                .iloc[i]
                .remaining_ratio,
            }
        )
    add("### 6. Final remaining weights and model profiles\n\n" + table(ratios))
    image(
        "remaining_ratios.png",
        "Layer-wise remaining-weight ratios for all unstructured methods.",
    )
    for n in ["iterative_magnitude", "iterative_grasp"]:
        image(
            n + "_curves.png",
            LABELS[n] + ": full training budget including the common warm-up.",
        )
    im = metrics["iterative_magnitude"]["test_top1"]
    ig = metrics["iterative_grasp"]["test_top1"]
    gm = metrics["global"]["test_top1"]
    add(
        f"### Required comparison\n\nIterative magnitude achieved {im:.2f}% test Top-1, iterative GraSP achieved {ig:.2f}%, and pretrained global magnitude achieved {gm:.2f}%. GraSP minus iterative magnitude is {ig-im:+.2f} percentage points. Iterative magnitude minus pretrained global magnitude is {im-gm:+.2f} points. The first difference is a controlled comparison of the criterion under shared initialization and training budget. The second is descriptive only: Task 1 inherits a pretrained solution and unknown original training cost, whereas Task 2 must learn representations from scratch. It cannot isolate the benefit of iterative scheduling. A single seed also cannot establish statistical superiority."
    )
    collapse = [
        (r["method"], r["stage"], r["collapsed_layers"])
        for r in stages
        if r["collapsed_layers"]
    ]
    add(
        "Layer collapse observations: "
        + (
            "No eligible layer became completely pruned at any recorded stage."
            if not collapse
            else str(collapse)
            + ". A collapsed layer can destroy a feature path; residual bypasses may only partially mitigate that damage."
        )
    )
    chance_stages = [
        r
        for r in stages
        if r["method"] == "iterative_grasp" and r["after_test_top1"] <= 10.5
    ]
    if chance_stages:
        add(
            f"GraSP caused a separate functional failure immediately after {len(chance_stages)} pruning stage(s): test accuracy fell to approximately chance level even though no full layer was deleted. Thus 'no layer collapse' refers only to the structural mask check, not to preservation of useful predictions. The later learning curves demonstrate recovery with continued training. The Taylor score measures a local gradient-flow change, and removing a large fraction of weights at once can violate that local approximation; fixed pre-pruning BatchNorm statistics can also be poorly matched immediately after deletion. These are plausible mechanisms rather than experimentally isolated causes."
        )
    add(
        "A saliency criterion can preserve a local gradient-flow property while still losing eventual classification accuracy. Its finite calibration subset, fixed BatchNorm statistics, temperature, higher-order numerical sensitivity, and threshold overshoot all affect the selected support. The criterion costs and learning curves show whether a measured benefit compensates for its extra computation. The final results use the same dense masked and COO inference procedure as Task 1."
    )
    add("## Task 3: Regression-based structured channel pruning")
    spec = read("structured_architecture.json")
    structure = pd.read_csv(results / "structured_stages.csv")
    add(
        "### Sequential channel selection with valid residual connections\n\nThe dense pretrained checkpoint is restored. Starting from the second residual block (`layer1.1`), the hidden channels between each block's first and second convolutions are pruned sequentially. The first residual block, stem, external block widths, and projection shortcuts are preserved. For each selected input channel of `conv2`, the same output channel of the preceding `conv1` and corresponding `bn1` entries are retained, in the same sorted index order. Thus both branches of every residual addition still have identical external channel dimensions and ordering. Shortcuts require no modification under this internal-width design. This is an explicit architectural allocation choice; it does not claim that every external channel or every convolution input was pruned."
    )
    add(
        f"The hidden widths are chosen by a one-dimensional retention-ratio search that accounts for deleting both the producer's output filters and the consumer's input filters. It achieves {100*spec['weight_reduction']:.4f}% reduction in eligible weight count against the 70% target; the small difference is the unavoidable discrete channel granularity. The final linear layer remains dense. The removed count includes both affected convolutions, rather than counting the same requested input fraction as the overall network sparsity."
    )
    add(
        "### Calibration, unfolding, LASSO, and reconstruction\n\nCalibration uses 1,000 optimization images, 100 per class, with no augmentation. Hooks capture the current student input to each target convolution and the original dense teacher's output before BatchNorm for the same images and spatial coordinates. Using the teacher target lets later reconstructions compensate for error accumulated in earlier pruned blocks. This is an asymmetric reconstruction choice; the current input multiplied by the old weights is also evaluated and logged as the pre-reconstruction reference.\n\nFor a convolution with $c$ inputs, $n$ outputs, and kernel area $K=k_hk_w$, unfolding gives\n\n$$X_{\\mathrm{unf}}\\in\\mathbb{R}^{NL\\times cK},\\quad W_{\\mathrm{mat}}\\in\\mathbb{R}^{cK\\times n},\\quad Y_{\\mathrm{current}}=X_{\\mathrm{unf}}W_{\\mathrm{mat}}.$$\n\nAt most eight spatial locations per image are sampled, capped at 4,096 paired rows. Channel contributions are $Z_i=X_iW_i^T$. Up to 16,384 scalar row/output equations form the selection problem. Columns are divided by their RMS, and coefficients are mapped back to the original scale afterward. The LASSO objective on these normalized equations is\n\n$$\\widehat\\beta=\\arg\\min_\\beta\\frac{1}{2m}\\left\\|y-\\sum_i\\beta_i z_i\\right\\|_2^2+\\lambda\\|\\beta\\|_1.$$\n\nA descending logarithmic lambda path starts at the all-zero threshold and stops at the first support with at least the desired width. If a support jump skips that exact width, the largest standardized coefficients provide the exact support. The entire path, convergence diagnostics, beta values, and selected indices are saved in `structured_lasso.json`. This avoids presenting an arbitrary magnitude mask as LASSO channel selection."
    )
    add(
        "The reduced reconstruction design is\n\n$$X^\\prime=[\\beta_1X_1,\\ldots,\\beta_{c^\\prime}X_{c^\\prime}],\\qquad A^\\star=\\arg\\min_A\\|Y_{\\mathrm{teacher}}-X^\\prime A\\|_F^2.$$\n\n`torch.linalg.lstsq` solves this in CPU float64 with pivoted QR and a stated rank tolerance. Beta is folded into the corresponding input slices of the reconstructed convolution, so inference uses ordinary dense convolutions and needs no extra beta layer. The table reports regression dimensions, estimated rank, reconstruction errors, and validation accuracy at each physical replacement. These calibration MSEs are reconstruction diagnostics, not independent test losses. Rank deficiency, if present, is visible in the table."
    )
    add(
        structure.to_markdown(
            index=False,
            floatfmt=[".3e" if "MSE" in c else ".4f" for c in structure.columns],
        )
    )
    pre = read("structured_before_finetune.json")
    m = metrics["structured"]
    add(
        f"### Fine-tuning and final performance\n\nAfter all replacements, 18 epochs of SGD fine-tuning use initial learning rate 0.01, momentum 0.9, weight decay 0.0005, and cosine decay to 0.0001. No sparsity mask is required because removed channels no longer exist. Test Top-1 changed from {pre['test_top1']:.2f}% immediately after reconstruction to {m['test_top1']:.2f}% after fine-tuning, recovering {m['test_top1']-pre['test_top1']:.2f} percentage points. The final dense checkpoint and hidden-width metadata together reconstruct the modified network."
    )
    image(
        "structured_curves.png",
        "Structured channel pruning: recovery during final fine-tuning.",
    )
    speed = base["latency_ms"] / m["latency_ms"]
    memdelta = base["peak_GPU_MB"] - m["peak_GPU_MB"]
    drop = base["test_top1"] - m["test_top1"]
    add(
        f"Structured inference uses {m['MACs']:,} MACs per image versus {base['MACs']:,} for the baseline. Its measured latency ratio is {speed:.3f}x baseline speed, and its absolute peak allocated memory is {memdelta:+.3f} MB lower than baseline. Its test accuracy drop is {drop:.2f} percentage points. The broad assignment accuracy goal of at most 15 percentage points is {'met' if drop<=15 else 'not met'} in this run. Physical channel removal reduces dense operation dimensions and state size, but measured speedup need not equal parameter reduction: unchanged stages, small-channel kernel efficiency, launch overhead, and the common input cache limit the gain. The common batch size makes this a throughput-oriented comparison; it is not a claim about single-image edge-device latency."
    )
    add(
        "## Limitations and interpretation\n\nThis submission reports one seed and one T4 GPU session. Hardware scheduling and GPU clocks can vary; latency standard deviation describes repeated batches, not uncertainty across independent machines. The scratch-training comparison has a fixed 50-epoch budget and may not have converged. Sensitivity and reconstruction use finite samples; layerwise independent sensitivity does not fully describe joint pruning interactions. Structured and unstructured sparsity use different physical representations and therefore different compute consequences, even when their eligible weight-removal budgets match. CPU energy is unavailable where host counters are not exposed. The included raw files make these limitations and all reported values auditable."
    )
    add(
        "## References and code attribution\n\n1. Huy Phan, *PyTorch CIFAR-10 models*: https://github.com/huyvnphan/PyTorch_CIFAR10 . Architecture and pretrained checkpoint only; experiment, pruning, profiling, and reporting code in this submission were written for this assignment.\n2. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009), CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html .\n3. Wang, Zhang, and Grosse, *Picking Winning Tickets Before Training by Preserving Gradient Flow* (ICLR 2020): https://arxiv.org/abs/2002.07376 . Signed GraSP scoring and ranking.\n4. He, Zhang, and Sun, *Channel Pruning for Accelerating Very Deep Neural Networks* (ICCV 2017): https://arxiv.org/abs/1707.06168 . LASSO selection and least-squares reconstruction.\n5. Han, Mao, and Dally, *Deep Compression* (ICLR 2016): https://arxiv.org/abs/1510.00149 . Magnitude pruning background.\n6. PyTorch profiler documentation: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html .\n7. PyTorch sparse tensors: https://docs.pytorch.org/docs/stable/sparse.html .\n8. Official torchvision CIFAR-10 checksums: https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py . The mirror changes transport, not the dataset definition."
    )
    text = "\n\n".join(parts) + "\n"
    assert (
        text.isascii()
    ), "Report text must be ASCII (LaTeX handles mathematical symbols)."
    (root / "REPORT.md").write_text(text, encoding="utf-8")
    comparison_figures(root, metrics, layer_frames)
    print(
        "REPORT.md generated from verified result files. No result placeholders.",
        flush=True,
    )
    return text


def comparison_figures(root, metrics, frames):
    root = Path(root)
    names = list(metrics)
    labels = [LABELS[n] for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, field, title in zip(
        axes,
        ["test_top1", "latency_ms", "dense_MB"],
        ["Test Top-1 (%)", "Inference latency (ms/batch)", "Dense state size (MB)"],
    ):
        ax.bar(
            range(len(names)),
            [metrics[n][field] for n in names],
            color=["#526b7d", "#178a86", "#46b1a5", "#aa7a3f", "#b69b61", "#435cb8"],
        )
        ax.set_xticks(range(len(names)), labels, rotation=55, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "figures" / "final_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, df in frames.items():
        ax.plot(
            range(len(df)), df.remaining_ratio, "o-", label=LABELS[name], markersize=3
        )
    ax.set_xticks(range(len(df)), df.layer, rotation=65, ha="right")
    ax.set_ylabel("Fraction of weights retained")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "figures" / "remaining_ratios.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    generate_report(sys.argv[1] if len(sys.argv) > 1 else ".")
