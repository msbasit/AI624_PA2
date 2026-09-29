# AI624 Assignment 2: Pruning ResNet-18 on CIFAR-10

## Authorship and disclosure

I have used OpenAI Codex to write original experiment code, explain pruning methods, execute the experiments, debug the implementation, create plots, and draft the report 1 time. Here, one time means one continuous assignment assistance session, including iterative tool calls and corrections. The measured results were produced by executing the supplied code. The model architecture and pretrained weights are attributed to Huy Phan's public repository; the pruning methods are attributed to their research papers. This is an AI-assisted submission, and does not claim that AI-generated work was independently handwritten.

## Experimental design and reproducibility

All final experiments use one NVIDIA Tesla T4 GPU session, FP32, batch size 128, and 3 x 32 x 32 RGB inputs. The requested final pruning budget is 70% of Conv2d and Linear weights. Biases and normalization parameters are excluded from that denominator. Model parameter counts include all trainable tensors. Tasks 0, 1, and 3 start from exactly the same checkpoint; Task 2 starts from a separately saved random initialization. The random seed is 624.

| Setting           | Value                                                            |
|:------------------|:-----------------------------------------------------------------|
| python            | 3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]               |
| torch             | 2.11.0+cu128                                                     |
| torchvision       | 0.26.0+cu128                                                     |
| cuda              | 12.8                                                             |
| cudnn             | 91900                                                            |
| gpu               | Tesla T4                                                         |
| cpu               | x86_64                                                           |
| gpu_total_MB      | 15637.086208                                                     |
| repository_commit | 641cac24371b17052b9bb6e56af1c83b5e97cd7f                         |
| checkpoint_sha256 | 72d30ca70e7d54e24a26628113604c40bc872eb8dc7a44f50ec58c3a1400f1b0 |

The upstream architecture is loaded directly from `cifar10_models/resnet.py`, including its stem max-pooling operation. Loading uses strict key and shape checks and verifies every state tensor against the checkpoint. The Box weights URL returned 404 during setup, so the archive came from the official Google Drive backup linked in the repository README. CIFAR-10 came from a byte-identical mirror because the original host was unusually slow. Torchvision checks the official archive and extracted batch MD5 hashes before use. The expected archive MD5 is `c58f30108f718f92721af3b95e74349a`.

### Data and preprocessing

The official 50,000 training images are divided by a seeded, class-stratified split into 45,000 optimization and 5,000 validation images. The official 10,000-image test set remains separate. The train metrics in final tables evaluate all 50,000 official training images without augmentation; the epoch curves evaluate the 45,000 augmented optimization images. Therefore these two training accuracy quantities need not match. The validation split was not used for gradient updates here, but the supplied pretrained checkpoint was originally trained on the original training set: validation results in Tasks 1 and 3 are not an unseen-pretraining generalization estimate. Final test results are the main generalization comparison.

Each image is divided by 255 and normalized with channel means `(0.4914, 0.4822, 0.4465)` and standard deviations `(0.2471, 0.2435, 0.2616)`. Training uses four-pixel constant padding, a random 32 x 32 crop, and horizontal flipping with probability 0.5. Validation, calibration, profiling, and final evaluation use normalization only. Images are cached as uint8 tensors on the GPU; the deterministic batch iterator shuffles and augments with seed plus epoch, avoiding CPU transfer bottlenecks. The last incomplete evaluation batch is included. Profiling always uses complete batches of 128.

### Metric definitions

Top-1 and Top-5 are percentages. Macro-F1 is on a 0 to 1 scale and gives each class equal weight. With class confusion counts,

$$F_{1,c}=\frac{2TP_c}{2TP_c+FP_c+FN_c}, \qquad F_{1,\mathrm{macro}}=\frac{1}{10}\sum_{c=1}^{10}F_{1,c}.$$

A convolution uses $H_oW_oC_o(C_i/g)k_hk_w$ MACs per image; a linear layer uses $d_{in}d_{out}$. We report $\mathrm{FLOPs}=2\,\mathrm{MACs}$, counting the multiply and add separately. This is an analytical Conv/Linear count, excluding BatchNorm, activation, pooling, bias, data preparation, and sparse conversion operations. The `ideal_nonzero_MACs` field is only a theoretical zero-skipping count and is not an executed operation count. Serialized MB means decimal bytes divided by $10^6$.

### Common latency, memory, and energy procedure

All models are evaluated in inference mode with the same ten normalized GPU-resident test batches. After 20 warm-up forwards and synchronization, 100 forward latencies are measured using CUDA events with synchronization at the end of each forward. Mean and standard deviation are saved. PyTorch allocated GPU memory is sampled after each forward; peak allocation is reset before each measured batch, and the largest batch peak is reported. Average memory is the arithmetic mean of post-forward allocations, not a continuous time average. Final profiling reloads each checkpoint after training has finished, with no live optimizer, parameter gradients, or GPU pruning masks. The common dataset cache contributes to absolute allocation. Incremental forward peak is also saved; these measurements are not whole-device NVML memory usage. All final metric files and batch CSVs contain this clean inference-only profiling pass.

GPU energy integrates NVML whole-board power, sampled approximately every 20 ms, during three independent inference windows of at least two seconds. Each window is divided by its actual completed batch count; the table gives the mean joules per batch. These are board-energy estimates, include idle board power and synchronization overhead, and are not process-isolated or wall-socket measurements. CPU energy uses package-level RAPL counters if accessible; unavailable counters are recorded as `N/A`, never zero. The execution environment did not expose host CPU power counters to the guest. CPU energy cannot be measured reliably from these guest permissions; this requested metric is explicitly reported as unavailable rather than invented. Raw per-batch timing/memory CSVs and per-window energy JSONs are included.

## Task 0: Dense model baseline

The verified pretrained model achieved 93.07% test Top-1 accuracy and 99.87% training Top-1 accuracy. The tables below contain the baseline and every final model, using the same metrics and measurement definitions. 'Sparsity' for structured pruning means physical reduction of eligible weight count; it does not mean the remaining dense tensors contain that fraction of zeros.

### Baseline result

The table reports every Task 0 deliverable from the saved final baseline measurements in `results/baseline_metrics.json`. Both training and test values are included for Top-5 and macro-F1.

| Metric | Dense baseline result |
|:---|:---|
| Train Top-1 accuracy | 99.8660% |
| Test Top-1 accuracy | 93.0700% |
| Train Top-5 accuracy | 99.9960% |
| Test Top-5 accuracy | 99.7400% |
| Train macro-F1 | 0.998660 |
| Test macro-F1 | 0.930700 |
| Number of model parameters | 11,173,962 |
| Serialized dense model size | 44.7717 MB |
| MACs per image | 140,186,624 |
| FLOPs per image | 280,373,248 |
| Average inference latency | 12.2287 ms per batch |
| Peak GPU memory usage | 339.5205 MB |
| Average GPU memory usage | 267.6326 MB |
| CPU energy consumption | N/A - CPU energy counters were unavailable |
| GPU energy consumption | 0.8294 J per batch |

Measurements use FP32 on the Tesla T4, with batch size 128 and 3 x 32 x 32 inputs. Train metrics cover all 50,000 training images, and test metrics cover all 10,000 test images. MACs and FLOPs are per image; FLOPs count one multiply and one add as two operations. Model size and memory use decimal MB.

Latency is averaged over 100 batches after 20 warm-up forwards; its standard deviation is 0.1676 ms. Peak memory is the maximum allocated peak across the measured batches, while average memory is the mean post-forward allocation. Both include the common GPU dataset cache. GPU energy is the mean per-batch whole-board estimate over three inference windows of at least two seconds. CPU energy is unavailable, not zero, because the execution environment did not expose the required counters.

These are the saved final checkpoint-reload measurements. The raw task logs retain their original preliminary resource measurements; final metric files and comparison tables use the subsequent clean inference-only profiling pass. The tables below compare the baseline with all pruning methods.

### Accuracy comparison

| Model               |   Sparsity % |   Train Top-1 % |   Test Top-1 % |   Train Top-5 % |   Test Top-5 % |   Train macro-F1 |   Test macro-F1 |   Drop vs dense (pp) |
|:--------------------|-------------:|----------------:|---------------:|----------------:|---------------:|-----------------:|----------------:|---------------------:|
| Dense baseline      |       0.0000 |         99.8660 |        93.0700 |         99.9960 |        99.7400 |           0.9987 |          0.9307 |               0.0000 |
| Local magnitude     |      70.0000 |         98.1820 |        91.9700 |         99.9800 |        99.6900 |           0.9818 |          0.9196 |               1.1000 |
| Global magnitude    |      70.0000 |         96.4960 |        90.9300 |         99.9260 |        99.7300 |           0.9649 |          0.9092 |               2.1400 |
| Iterative magnitude |      70.0000 |         98.3560 |        90.5900 |         99.9700 |        99.6600 |           0.9835 |          0.9058 |               2.4800 |
| Iterative GraSP     |      70.0000 |         96.2300 |        88.5500 |         99.9240 |        99.5200 |           0.9622 |          0.8849 |               4.5200 |
| Structured channels |      70.0320 |         93.2280 |        88.9200 |         99.8340 |        99.5600 |           0.9322 |          0.8889 |               4.1500 |

### Model size and inference resources

| Model               |   Parameters |   Dense MB | COO MB     |   MACs/image |   FLOPs/image |   Latency ms/batch |   Peak GPU MB |   Average GPU MB |
|:--------------------|-------------:|-----------:|:-----------|-------------:|--------------:|-------------------:|--------------:|-----------------:|
| Dense baseline      |     11173962 |    44.7717 | N/A        |    140186624 |     280373248 |            12.2287 |      339.5205 |         267.6326 |
| Local magnitude     |     11173962 |    44.7713 | 131.840075 |    140186624 |     280373248 |            12.4733 |      339.5205 |         267.6326 |
| Global magnitude    |     11173962 |    44.7715 | 131.782517 |    140186624 |     280373248 |            12.3918 |      339.5215 |         267.6337 |
| Iterative magnitude |     11173962 |    44.7732 | 131.807127 |    140186624 |     280373248 |            12.1353 |      339.5215 |         267.6337 |
| Iterative GraSP     |     11173962 |    44.7726 | 131.847087 |    140186624 |     280373248 |            12.0313 |      339.5215 |         267.6337 |
| Structured channels |      3352678 |    13.4757 | N/A        |     55592960 |     111185920 |             7.7381 |      303.7957 |         236.6920 |

### Energy and measurement variability

| Model               |   GPU J/batch | CPU J/batch   |   Latency SD ms |   Incremental peak MB |
|:--------------------|--------------:|:--------------|----------------:|----------------------:|
| Dense baseline      |        0.8294 | N/A           |          0.1676 |               71.8930 |
| Local magnitude     |        0.8490 | N/A           |          0.1919 |               71.8930 |
| Global magnitude    |        0.8294 | N/A           |          0.1739 |               71.8930 |
| Iterative magnitude |        0.8192 | N/A           |          0.1412 |               71.8930 |
| Iterative GraSP     |        0.9518 | N/A           |          0.1600 |               71.8930 |
| Structured channels |        0.5403 | N/A           |          0.2127 |               67.1089 |

![Final accuracy, latency, and dense model size on the same T4 GPU.](figures/final_comparison.png)

Final accuracy, latency, and dense model size on the same T4 GPU.

## Task 1: Unstructured post-training pruning

### 1. Independent layer sensitivity

Each of the 21 convolutional or linear weight tensors is pruned independently at 0%, 10%, 20%, 50%, 70%, 80%, and 90%. All state tensors are restored from the original checkpoint before each trial, and there is no fine-tuning in these trials. Accuracy is measured on the 5,000-image validation split. This prevents the test set from choosing layer allocations. Every curve is included below, and every numeric observation is in `results/sensitivity.csv`.

Largest validation accuracy losses at 70% single-layer sparsity:

| Layer          |   Drop at 70% (pp) |
|:---------------|-------------------:|
| conv1          |            14.8600 |
| layer2.0.conv1 |             0.7000 |
| layer2.1.conv1 |             0.3800 |
| layer1.1.conv1 |             0.2600 |
| layer2.0.conv2 |             0.2600 |

The most sensitive layer by this criterion is `conv1`. Sensitivity measures the effect of deleting weights in isolation. It does not guarantee that combining several individually tolerable deletions will be harmless: downstream representations and residual branch interactions can amplify their joint effect.

![Independent layer sensitivity; every trial restarts from the dense checkpoint.](figures/sensitivity.png)

Independent layer sensitivity; every trial restarts from the dense checkpoint.

### 2. Local allocation and 3. Global allocation

For layer $\ell$, let $d_\ell=\max(0,a_{\ell,0}-a_{\ell,0.7})$ in percentage points. The local allocation uses

$$q_\ell=\frac{1}{1+d_\ell/5}, \qquad s_\ell=\operatorname{clip}(\alpha q_\ell,0.05,0.90).$$

A scalar water-filling search chooses $\alpha$ so the parameter-weighted average is 70%. Integer counts use largest-remainder rounding to reach exactly the requested total. Sensitive layers consequently receive a lower pruning fraction. Each layer removes its smallest absolute weights. Global pruning ranks the concatenated absolute weights, removes the same exact total, and lets the individual layer fractions emerge. Stable flattened-index ordering breaks threshold ties deterministically. The global method therefore has one shared magnitude cutoff, apart from deterministic handling of equal values at that cutoff.

| Layer                 |   Weights |   Local mask % |   Local actual zero % |   Global mask % |   Global actual zero % |
|:----------------------|----------:|---------------:|----------------------:|----------------:|-----------------------:|
| conv1                 |      1728 |        17.7083 |               17.7083 |         14.8727 |                14.8727 |
| layer1.0.conv1        |     36864 |        69.6696 |               69.6696 |         22.0215 |                22.0215 |
| layer1.0.conv2        |     36864 |        68.5791 |               68.5791 |          6.1252 |                 6.1252 |
| layer1.1.conv1        |     36864 |        66.7535 |               66.7535 |          6.7220 |                 6.7220 |
| layer1.1.conv2        |     36864 |        70.2257 |               70.2257 |          7.6253 |                 7.6253 |
| layer2.0.conv1        |     73728 |        61.6007 |               61.6007 |          6.4887 |                 6.4887 |
| layer2.0.conv2        |    147456 |        66.7542 |               66.7542 |          6.9722 |                 6.9722 |
| layer2.0.downsample.0 |      8192 |        68.5791 |               68.5791 |          4.2969 |                 4.2969 |
| layer2.1.conv1        |    147456 |        65.2656 |               65.2656 |          8.0526 |                 8.0526 |
| layer2.1.conv2        |    147456 |        70.2257 |               70.2257 |         10.2098 |                10.2098 |
| layer3.0.conv1        |    294912 |        69.1196 |               69.1196 |         11.4655 |                11.4655 |
| layer3.0.conv2        |    589824 |        70.2255 |               70.2255 |         17.9923 |                17.9923 |
| layer3.0.downsample.0 |     32768 |        70.2271 |               70.2271 |         11.7035 |                11.7035 |
| layer3.1.conv1        |    589824 |        70.2255 |               70.2255 |         29.6738 |                29.6738 |
| layer3.1.conv2        |    589824 |        70.2255 |               70.2255 |         44.3402 |                44.3402 |
| layer4.0.conv1        |   1179648 |        70.2256 |               70.2256 |         70.2994 |                70.2994 |
| layer4.0.conv2        |   2359296 |        70.2256 |               70.2256 |         82.3419 |                82.3419 |
| layer4.0.downsample.0 |    131072 |        70.2255 |               70.2255 |         17.4156 |                17.4156 |
| layer4.1.conv1        |   2359296 |        70.2256 |               70.2256 |         96.9073 |                96.9073 |
| layer4.1.conv2        |   2359296 |        70.2256 |               70.2256 |         88.8113 |                88.8113 |
| fc                    |      5120 |        70.2344 |               70.2344 |          0.0391 |                 0.0391 |

Global pruning removed the largest fraction in `layer4.1.conv1` (96.91%) and the smallest fraction in `fc` (0.04%). No layer was completely removed. This is a usable distribution because it preserves a nonempty path through every eligible layer and supports the measured recovery below. It is not a sensitivity guarantee: magnitude scales differ between layers and interact with normalization, so small absolute values alone do not prove low functional importance.

### 4. Fine-tuning and 5. Mask verification

Both methods use SGD with Nesterov momentum 0.9, weight decay 0.0005, initial learning rate 0.01, 12 epochs, and cosine decay to 0.0001. Batch size, seed, batch order, and augmentations are identical. Masks remain fixed. Gradients and momentum buffers are also masked; after every optimizer step, $W \leftarrow M\odot W$ is enforced. Assertions check every masked coordinate, and the final layer table compares actual zeros with mask zeros. Extra natural zeros, if present, are recorded separately instead of being mistaken for an additional pruning decision.

| Method           |   Before pruning test % |   After pruning test % |   After FT test % |   Recovered (pp) |   Final drop (pp) |
|:-----------------|------------------------:|-----------------------:|------------------:|-----------------:|------------------:|
| Local magnitude  |                 93.0700 |                88.0700 |           91.9700 |           3.9000 |            1.1000 |
| Global magnitude |                 93.0700 |                93.0000 |           90.9300 |          -2.0700 |            2.1400 |

For global magnitude, fine-tuning reduced test accuracy by 2.07 percentage points relative to the immediate pruned checkpoint. Thus it did not recover accuracy in this case. The curves show the disruption and subsequent partial recovery under the shared learning-rate schedule. Updates to weights and BatchNorm statistics under augmentation can move an already accurate pretrained solution away from its useful operating point; the experiment does not isolate which factor caused the decrease. A smaller learning rate or longer schedule could be investigated on validation data, but the fixed protocol is retained here for a fair local/global comparison.

![Local magnitude: training and validation loss and accuracy.](figures/local_curves.png)

Local magnitude: training and validation loss and accuracy.

![Global magnitude: training and validation loss and accuracy.](figures/global_curves.png)

Global magnitude: training and validation loss and accuracy.

### 6. COO storage and 7. COO inference

Each pruned tensor is converted with `to_sparse().coalesce()`. The saved file contains float32 nonzero values, int64 COO indices, original shapes, Boolean pruning masks, and unpruned state tensors. A load-and-reconstruct check verifies exact equality for every pruned tensor. COO convolutions explicitly call `to_dense()` during every forward because ordinary PyTorch Conv2d does not execute general sparse COO convolution. The final linear layer uses `torch.sparse.mm`. Output agreement with the dense masked representation is asserted before profiling, and full evaluation is repeated for COO execution.

| Model                   |   Test Top-1 % |   Train Top-1 % |   Test Top-5 % |   Train Top-5 % |   Test macro-F1 |   Train macro-F1 |   Latency ms |   Peak GPU MB |   Average GPU MB |   GPU J/batch | CPU J/batch   |
|:------------------------|---------------:|----------------:|---------------:|----------------:|----------------:|-----------------:|-------------:|--------------:|-----------------:|--------------:|:--------------|
| Local magnitude COO     |        91.9700 |         98.1820 |        99.6900 |         99.9800 |          0.9196 |           0.9818 |      13.5093 |      417.6543 |         345.6020 |        0.9117 | N/A           |
| Global magnitude COO    |        90.9300 |         96.4960 |        99.7300 |         99.9260 |          0.9092 |           0.9649 |      13.0739 |      418.8339 |         345.2503 |        0.8958 | N/A           |
| Iterative magnitude COO |        90.5900 |         98.3560 |        99.6600 |         99.9700 |          0.9058 |           0.9835 |      13.0901 |      417.4454 |         344.7547 |        0.8893 | N/A           |
| Iterative GraSP COO     |        88.5500 |         96.2300 |        99.5200 |         99.9240 |          0.8849 |           0.9622 |      13.0375 |      416.7086 |         344.6733 |        0.8828 | N/A           |

For a four-dimensional convolution tensor, an entry uses approximately 4 bytes for its value and 4 x 8 bytes for its indices. At 70% sparsity, values and indices alone therefore cost about $0.3(4+32)=10.8$ bytes per original weight, versus 4 bytes in dense FP32; a stored Boolean mask adds about one more byte per original weight. The break-even sparsity is above $1-4/36=88.89\%$ even before masks and serialization overhead. Consequently a 70% sparse, four-coordinate COO file can be larger than its dense file. The measured file sizes include that overhead. A flattened two-dimensional encoding, compressed indices, or a hardware-supported format would have different storage costs, but none is substituted silently here.

Local magnitude: the dense state is 44.771 MB and the actual COO package is 131.840 MB, a size ratio of 2.94. Dense masked latency is 12.473 ms per batch versus 13.509 ms for the COO execution path.

Global magnitude: the dense state is 44.771 MB and the actual COO package is 131.783 MB, a size ratio of 2.94. Dense masked latency is 12.392 ms per batch versus 13.074 ms for the COO execution path.

Zero-valued weights do not reduce dense tensor dimensions, kernel launches, or the dense convolution loop bounds. The GPU executes the same dense kernels and nominal MAC count. COO-to-dense conversion adds reconstruction and allocation work, so sparse storage by itself is not an inference acceleration. Actual sparse speedups depend on kernel support, sparsity structure, batch size, and the cost of indexing and data movement.

## Task 2: Saliency-based iterative pruning

### 1. Shared random initialization and warm-up

The model is created with `resnet18(pretrained=False)` and does not load the pretrained checkpoint. Its initial state is saved. One shared warm-up ran for 1 epoch(s) and reached 32.96% validation Top-1. Both methods restore byte-equal copies of this state, optimizer momentum, and scheduler position. The warm-up checkpoint contains the initial weights, warm-up weights, optimizer state, and curves. Each method receives 50 total training epochs including the shared warm-up; SGD uses initial learning rate 0.1, Nesterov momentum 0.9, weight decay 0.0005, and cosine decay to 0.001. This is a fixed-budget comparison, not a claim of full convergence.

![Shared random-initialization warm-up.](figures/warmup_curves.png)

Shared random-initialization warm-up.

### 2. Calibration and 3. GraSP criterion

Calibration uses 200 normalization-only optimization images, 20 per class. With batch size 128, there are two batches of 128 and 72 images. Batch means are weighted by their sample counts. BatchNorm statistics are fixed during criterion computation. A temperature of 200 follows the GraSP implementation convention and is used for scoring only; optimization uses ordinary cross-entropy without temperature scaling.

$$L(\theta)=\sum_b\frac{n_b}{200}L_b(\theta),\quad g=\nabla L,\quad Hg=\nabla_\theta\left[(\nabla L)^T\operatorname{stopgrad}(g)\right],\quad S_i=-\theta_i(Hg)_i.$$

One pass accumulates the detached mean gradient. A second pass accumulates Hessian-vector products with that common vector. This retains cross-batch terms, unlike averaging independent $H_bg_b$ products. Pruned coordinates are removed from the gradient-norm direction in later stages. Bias and normalization parameters participate in the loss derivatives but are never candidates for pruning. Autograd computes the HVP without materializing a model Hessian. A three-parameter toy test compares the HVP against an explicit Hessian solely as a numerical unit check; no network Hessian is constructed.

The signed GraSP ranking removes the **largest** $S_i$ values. It does not remove the smallest values and does not take absolute values. For a deletion $\Delta\theta_i=-\theta_i$, the first-order change in squared gradient norm is $2g^TH\Delta\theta=-2\theta_i(Hg)_i=2S_i$. Preserving or increasing gradient flow therefore favors removing large scores under this convention. The direction is verified with a known signed-score example. This derivation also explains why an implementation that merely copies the magnitude-pruning sorting direction is wrong.

### 4. Three stages and 5. Cumulative masks

The cumulative sparsities are 35%, 52.5%, and 70%, corresponding to $0.50s^\star$, $0.75s^\star$, and $s^\star$. Stages trigger after epoch-level validation accuracy first reaches approximately 20%, 40%, and 60%; they can overshoot because validation is checked once per epoch. To guarantee the complete sparsity schedule within the fixed budget, fallback epoch deadlines are 5, 12, and 25 completed epochs. The table identifies the actual trigger; a deadline is not presented as reaching an accuracy threshold. Only currently retained weights can be removed, so the new mask is a subset of the old mask. The code asserts no regrowth and reapplies masks, including momentum masking, after every update.

| Method              |   Stage |   Next epoch |   Before val % |   After val % |   Before test % |   After test % |   Sparsity % |   Criterion seconds | Trigger   | Collapsed layers   |
|:--------------------|--------:|-------------:|---------------:|--------------:|----------------:|---------------:|-------------:|--------------------:|:----------|:-------------------|
| Iterative magnitude |       1 |            2 |        32.9600 |       33.0000 |         32.4000 |        32.5000 |      35.0000 |              0.0006 | accuracy  | None               |
| Iterative magnitude |       2 |            3 |        50.5600 |       49.6400 |         49.7500 |        48.7900 |      52.5000 |              0.0007 | accuracy  | None               |
| Iterative magnitude |       3 |            5 |        60.0400 |       59.5400 |         59.6100 |        59.4200 |      70.0000 |              0.0006 | accuracy  | None               |
| Iterative GraSP     |       1 |            2 |        32.9600 |       10.0000 |         32.4000 |        10.0000 |      35.0000 |              0.4689 | accuracy  | None               |
| Iterative GraSP     |       2 |            4 |        42.1600 |       10.0000 |         41.9600 |        10.0000 |      52.5000 |              0.3521 | accuracy  | None               |
| Iterative GraSP     |       3 |            8 |        61.3200 |       10.0000 |         59.3100 |        10.0000 |      70.0000 |              0.3524 | accuracy  | None               |

Criterion time is synchronized wall time for score construction. It excludes the subsequent common sorting and mask-application operation, which is why it should not be read as total pruning time. Test accuracies immediately before and after stages are recorded for the requested comparison; stage decisions use validation accuracy only.

### 6. Final remaining weights and model profiles

| Layer                 |   Local remaining |   Global remaining |   Iterative magnitude remaining |   Iterative GraSP remaining |
|:----------------------|------------------:|-------------------:|--------------------------------:|----------------------------:|
| conv1                 |            0.8229 |             0.8513 |                          0.8970 |                      0.1366 |
| layer1.0.conv1        |            0.3033 |             0.7798 |                          0.7302 |                      0.1576 |
| layer1.0.conv2        |            0.3142 |             0.9387 |                          0.7043 |                      0.1959 |
| layer1.1.conv1        |            0.3325 |             0.9328 |                          0.6903 |                      0.1648 |
| layer1.1.conv2        |            0.2977 |             0.9237 |                          0.6751 |                      0.2054 |
| layer2.0.conv1        |            0.3840 |             0.9351 |                          0.5688 |                      0.3091 |
| layer2.0.conv2        |            0.3325 |             0.9303 |                          0.5335 |                      0.3142 |
| layer2.0.downsample.0 |            0.3142 |             0.9570 |                          0.7712 |                      0.2607 |
| layer2.1.conv1        |            0.3473 |             0.9195 |                          0.5312 |                      0.3013 |
| layer2.1.conv2        |            0.2977 |             0.8979 |                          0.5309 |                      0.2593 |
| layer3.0.conv1        |            0.3088 |             0.8853 |                          0.4066 |                      0.2818 |
| layer3.0.conv2        |            0.2977 |             0.8201 |                          0.4006 |                      0.2690 |
| layer3.0.downsample.0 |            0.2977 |             0.8830 |                          0.7274 |                      0.2595 |
| layer3.1.conv1        |            0.2977 |             0.7033 |                          0.4067 |                      0.2721 |
| layer3.1.conv2        |            0.2977 |             0.5566 |                          0.4100 |                      0.2369 |
| layer4.0.conv1        |            0.2977 |             0.2970 |                          0.2511 |                      0.2267 |
| layer4.0.conv2        |            0.2977 |             0.1766 |                          0.2452 |                      0.3383 |
| layer4.0.downsample.0 |            0.2977 |             0.8258 |                          0.6786 |                      0.1758 |
| layer4.1.conv1        |            0.2977 |             0.0309 |                          0.2368 |                      0.2790 |
| layer4.1.conv2        |            0.2977 |             0.1119 |                          0.2429 |                      0.3687 |
| fc                    |            0.2977 |             0.9996 |                          0.7314 |                      0.2355 |

![Layer-wise remaining-weight ratios for all unstructured methods.](figures/remaining_ratios.png)

Layer-wise remaining-weight ratios for all unstructured methods.

![Iterative magnitude: full training budget including the common warm-up.](figures/iterative_magnitude_curves.png)

Iterative magnitude: full training budget including the common warm-up.

![Iterative GraSP: full training budget including the common warm-up.](figures/iterative_grasp_curves.png)

Iterative GraSP: full training budget including the common warm-up.

### Required comparison

Iterative magnitude achieved 90.59% test Top-1, iterative GraSP achieved 88.55%, and pretrained global magnitude achieved 90.93%. GraSP minus iterative magnitude is -2.04 percentage points. Iterative magnitude minus pretrained global magnitude is -0.34 points. The first difference is a controlled comparison of the criterion under shared initialization and training budget. The second is descriptive only: Task 1 inherits a pretrained solution and unknown original training cost, whereas Task 2 must learn representations from scratch. It cannot isolate the benefit of iterative scheduling. A single seed also cannot establish statistical superiority.

Layer collapse observations: No eligible layer became completely pruned at any recorded stage.

GraSP caused a separate functional failure immediately after 3 pruning stage(s): test accuracy fell to approximately chance level even though no full layer was deleted. Thus 'no layer collapse' refers only to the structural mask check, not to preservation of useful predictions. The later learning curves demonstrate recovery with continued training. The Taylor score measures a local gradient-flow change, and removing a large fraction of weights at once can violate that local approximation; fixed pre-pruning BatchNorm statistics can also be poorly matched immediately after deletion. These are plausible mechanisms rather than experimentally isolated causes.

A saliency criterion can preserve a local gradient-flow property while still losing eventual classification accuracy. Its finite calibration subset, fixed BatchNorm statistics, temperature, higher-order numerical sensitivity, and threshold overshoot all affect the selected support. The criterion costs and learning curves show whether a measured benefit compensates for its extra computation. The final results use the same dense masked and COO inference procedure as Task 1.

## Task 3: Regression-based structured channel pruning

### Sequential channel selection with valid residual connections

The dense pretrained checkpoint is restored. Starting from the second residual block (`layer1.1`), the hidden channels between each block's first and second convolutions are pruned sequentially. The first residual block, stem, external block widths, and projection shortcuts are preserved. For each selected input channel of `conv2`, the same output channel of the preceding `conv1` and corresponding `bn1` entries are retained, in the same sorted index order. Thus both branches of every residual addition still have identical external channel dimensions and ordering. Shortcuts require no modification under this internal-width design. This is an explicit architectural allocation choice; it does not claim that every external channel or every convolution input was pruned.

The hidden widths are chosen by a one-dimensional retention-ratio search that accounts for deleting both the producer's output filters and the consumer's input filters. It achieves 70.0320% reduction in eligible weight count against the 70% target; the small difference is the unavoidable discrete channel granularity. The final linear layer remains dense. The removed count includes both affected convolutions, rather than counting the same requested input fraction as the overall network sparsity.

### Calibration, unfolding, LASSO, and reconstruction

Calibration uses 1,000 optimization images, 100 per class, with no augmentation. Hooks capture the current student input to each target convolution and the original dense teacher's output before BatchNorm for the same images and spatial coordinates. Using the teacher target lets later reconstructions compensate for error accumulated in earlier pruned blocks. This is an asymmetric reconstruction choice; the current input multiplied by the old weights is also evaluated and logged as the pre-reconstruction reference.

For a convolution with $c$ inputs, $n$ outputs, and kernel area $K=k_hk_w$, unfolding gives

$$X_{\mathrm{unf}}\in\mathbb{R}^{NL\times cK},\quad W_{\mathrm{mat}}\in\mathbb{R}^{cK\times n},\quad Y_{\mathrm{current}}=X_{\mathrm{unf}}W_{\mathrm{mat}}.$$

At most eight spatial locations per image are sampled, capped at 4,096 paired rows. Channel contributions are $Z_i=X_iW_i^T$. Up to 16,384 scalar row/output equations form the selection problem. Columns are divided by their RMS, and coefficients are mapped back to the original scale afterward. The LASSO objective on these normalized equations is

$$\widehat\beta=\arg\min_\beta\frac{1}{2m}\left\|y-\sum_i\beta_i z_i\right\|_2^2+\lambda\|\beta\|_1.$$

A descending logarithmic lambda path starts at the all-zero threshold and stops at the first support with at least the desired width. If a support jump skips that exact width, the largest standardized coefficients provide the exact support. The entire path, convergence diagnostics, beta values, and selected indices are saved in `structured_lasso.json`. This avoids presenting an arbitrary magnitude mask as LASSO channel selection.

The reduced reconstruction design is

$$X^\prime=[\beta_1X_1,\ldots,\beta_{c^\prime}X_{c^\prime}],\qquad A^\star=\arg\min_A\|Y_{\mathrm{teacher}}-X^\prime A\|_F^2.$$

`torch.linalg.lstsq` solves this in CPU float64 with pivoted QR and a stated rank tolerance. Beta is folded into the corresponding input slices of the reconstructed convolution, so inference uses ordinary dense convolutions and needs no extra beta layer. The table reports regression dimensions, estimated rank, reconstruction errors, and validation accuracy at each physical replacement. These calibration MSEs are reconstruction diagnostics, not independent test losses. Rank deficiency, if present, is visible in the table.

| block    | old_hidden_channels | new_hidden_channels | removed_hidden_fraction | calibration_images | regression_rows | regression_columns | regression_rank | before_reconstruction_MSE | after_reconstruction_MSE | before_val_top1 | after_val_top1 | seconds            |
| -------- | ------------------- | ------------------- | ----------------------- | ------------------ | --------------- | ------------------ | --------------- | ------------------------- | ------------------------ | --------------- | -------------- | ------------------ |
| layer1.1 | 64                  | 18                  | 0.71875                 | 1000               | 4096            | 162                | 162             | 1.723e-17                 | 6.288e-06                | 99.82           | 99.38          | 2.086981523999839  |
| layer2.0 | 128                 | 36                  | 0.71875                 | 1000               | 4096            | 324                | 324             | 3.491e-06                 | 1.377e-05                | 99.38           | 95.5           | 1.2272137109998766 |
| layer2.1 | 128                 | 36                  | 0.71875                 | 1000               | 4096            | 324                | 324             | 4.156e-06                 | 5.885e-06                | 95.5            | 84.86          | 1.2192918050000117 |
| layer3.0 | 256                 | 73                  | 0.71484375              | 1000               | 4096            | 657                | 657             | 6.768e-06                 | 3.583e-06                | 84.86           | 84.02          | 1.9081726979998166 |
| layer3.1 | 256                 | 73                  | 0.71484375              | 1000               | 4096            | 657                | 657             | 4.011e-06                 | 1.518e-06                | 84.02           | 85.08          | 2.900941209999928  |
| layer4.0 | 512                 | 145                 | 0.716796875             | 1000               | 4000            | 1305               | 1305            | 1.097e-05                 | 6.118e-06                | 85.08           | 85.72          | 5.870372724999925  |
| layer4.1 | 512                 | 145                 | 0.716796875             | 1000               | 4000            | 1305               | 1305            | 1.014e-05                 | 6.221e-06                | 85.72           | 85.56          | 8.260345251000217  |

### Fine-tuning and final performance

After all replacements, 18 epochs of SGD fine-tuning use initial learning rate 0.01, momentum 0.9, weight decay 0.0005, and cosine decay to 0.0001. No sparsity mask is required because removed channels no longer exist. Test Top-1 changed from 81.69% immediately after reconstruction to 88.92% after fine-tuning, recovering 7.23 percentage points. The final dense checkpoint and hidden-width metadata together reconstruct the modified network.

![Structured channel pruning: recovery during final fine-tuning.](figures/structured_curves.png)

Structured channel pruning: recovery during final fine-tuning.

Structured inference uses 55,592,960 MACs per image versus 140,186,624 for the baseline. Its measured latency ratio is 1.580x baseline speed, and its absolute peak allocated memory is +35.725 MB lower than baseline. Its test accuracy drop is 4.15 percentage points. The broad assignment accuracy goal of at most 15 percentage points is met in this run. Physical channel removal reduces dense operation dimensions and state size, but measured speedup need not equal parameter reduction: unchanged stages, small-channel kernel efficiency, launch overhead, and the common input cache limit the gain. The common batch size makes this a throughput-oriented comparison; it is not a claim about single-image edge-device latency.

## Limitations and interpretation

This submission reports one seed and one T4 GPU session. Hardware scheduling and GPU clocks can vary; latency standard deviation describes repeated batches, not uncertainty across independent machines. The scratch-training comparison has a fixed 50-epoch budget and may not have converged. Sensitivity and reconstruction use finite samples; layerwise independent sensitivity does not fully describe joint pruning interactions. Structured and unstructured sparsity use different physical representations and therefore different compute consequences, even when their eligible weight-removal budgets match. CPU energy is unavailable where host counters are not exposed. The included raw files make these limitations and all reported values auditable.

## References and code attribution

1. Huy Phan, *PyTorch CIFAR-10 models*: https://github.com/huyvnphan/PyTorch_CIFAR10 . Architecture and pretrained checkpoint only; experiment, pruning, profiling, and reporting code in this submission were written for this assignment.
2. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009), CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html .
3. Wang, Zhang, and Grosse, *Picking Winning Tickets Before Training by Preserving Gradient Flow* (ICLR 2020): https://arxiv.org/abs/2002.07376 . Signed GraSP scoring and ranking.
4. He, Zhang, and Sun, *Channel Pruning for Accelerating Very Deep Neural Networks* (ICCV 2017): https://arxiv.org/abs/1707.06168 . LASSO selection and least-squares reconstruction.
5. Han, Mao, and Dally, *Deep Compression* (ICLR 2016): https://arxiv.org/abs/1510.00149 . Magnitude pruning background.
6. PyTorch profiler documentation: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html .
7. PyTorch sparse tensors: https://docs.pytorch.org/docs/stable/sparse.html .
8. Official torchvision CIFAR-10 checksums: https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py . The mirror changes transport, not the dataset definition.
