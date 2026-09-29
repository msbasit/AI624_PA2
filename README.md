# AI624 Assignment 2: Pruning ResNet-18

## Main notebook

Open `AI624_PA2.ipynb` in Jupyter. It contains all experiment implementation code,
saved outputs, nine embedded graphs, result tables, equations, and discussions.
There are no notebook imports from the supporting experiment modules and no
review-mode switches. All custom numerical functions are defined in notebook cells.

The displayed results come from the completed experiment. Cells were reorganized
without being executed again, so execution counts are blank. Read the saved outputs
without running the notebook. Executing task cells would repeat the experiments.

## Files

- `AI624_PA2.ipynb`: complete main notebook.
- `REPORT.md`: required standalone Markdown report covering all tasks.
- `results/`: original measured values, training histories, stages, and verification records.
- `figures/`: the original plots, also embedded in the notebook.
- `checkpoints/`: original pretrained, initial, warm-up, dense, and COO model states.
- `source/`: supporting copies of the implementation and the credited upstream architecture.
- `SHA256SUMS.json`: integrity hashes for this package.

The main notebook implements setup, metrics, profiling, pruning masks, sparse
inference, training, magnitude pruning, GraSP, channel reconstruction, and final
verification in separate sections. `source/pa2_experiments.py` is a supporting copy
of the same functions. The report and packaging utilities are optional supporting
tools; neither is needed to read the notebook.

## Reproduction instructions

The notebook uses its current working directory as `ROOT`. Extract the ZIP into
one folder, start Jupyter from that folder, and open the notebook. For an intentional
new experiment, install the dependencies and execute the cells from top to bottom
on a CUDA-enabled GPU. No supporting experiment module needs to be imported.

The setup function loads the required upstream ResNet-18 and verified checkpoint,
downloads the dataset if missing, and prepares the original seeded data splits.
Tasks 0, 1, and 3 start from the same pretrained state; Task 2 uses a shared random
initialization and warm-up. Task functions write new measurements and checkpoints,
so reproduction should use a separate extracted copy if the original results are
to be retained. Training loops do not automatically resume interrupted tasks.

## Dependencies

Recorded environment: Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
CUDA 12.8, and a Tesla T4 GPU. Additional dependencies are NumPy, pandas,
matplotlib, scikit-learn, requests, gdown, nvidia-ml-py, tabulate, IPython, and
Jupyter. Exact package versions are saved in `results/package_versions.json`.
Git is needed only if the supplied upstream architecture directory is missing.
Viewing existing outputs requires no GPU or experiment execution.

## Results and limitations

All recorded measurements, training histories, graphs, and model files are
unchanged. CPU energy is unavailable because host energy counters were not exposed
in the execution environment. GPU energy is a whole-board estimate. The report
provides the complete measurement definitions and limitations.

The architecture, checkpoint, dataset, and papers are credited. The required
AI-use acknowledgment remains in the notebook and report. Submit the complete
ZIP after reviewing the work; it has not been submitted to the LMS.
