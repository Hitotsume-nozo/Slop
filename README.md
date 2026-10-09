# Slop

Machine-learning projects on the bundled datasets:

| Folder | Task | Model | Run it |
|---|---|---|---|
| [`diabetes_mlp/`](diabetes_mlp/) | 3-class diabetes-risk prediction (tabular, ~50k rows) | MLP (PyTorch) + LogReg/GB baselines | `cd diabetes_mlp && ./run.sh` |
| [`handwritten_cnn/`](handwritten_cnn/) | 10-class handwritten digits (28×28, tiny data) | ModernConvNet CNN w/ mixup, CV, Grad-CAM, t-SNE | `cd handwritten_cnn && ./run.sh` |
| [`catdog_audio_classifier/`](catdog_audio_classifier/) | Cat vs dog vocalization noise, single- & multi-channel spectrograms | Separation CNN (CE + triplet margin loss) + raw-wave control | `cd catdog_audio_classifier && ./run.sh` |

Each project contains a `DESIGN.md` (rigorous design rationale), a
`PROGRESS.md` (status log), source under `src/`, and a one-shot `run.sh` that
writes a **verbose timestamped log** to `logs/pipeline_<timestamp>.log` while
also printing to the console. Generated artifacts go to `artifacts/`
(gitignored). Send back the log file if a run fails.
