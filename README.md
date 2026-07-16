# ChACo: Channel-wise Adaptive Competitive Layer-wise Learning

Research code for the accepted tNNLS 2026 paper **“ChACo: Channel-wise Adaptive Competitive Layer-wise Learning.”**

## Abstract
Local layer-wise learning offers modular optimization, layer-level transparency, and training without end-to-end error transport. However, its scalability remains limited by three coupled difficulties: local objectives can be weak or poorly aligned with the final task, shallow layers are often forced into premature fine-class discrimination, and fully local optimization is sensitive to activation and variance drift across depth. In this work, we present \textbf{ChACo}, a channel-wise adaptive competitive framework for fully local learning in convolutional networks. Each layer is trained with a local discriminative objective, while the internal block-to-class association is allowed to vary with depth. Early layers can map fewer competitive blocks to fine-label logits through a learned association policy, whereas later layers can recover direct fine-class competition as a special case. This design reduces the shallow-layer channel burden in many-class settings while preserving a direct local path to the final label space. The framework is supported by an analysis of local optimization dynamics, showing why even-power goodness functions and the ordering of rectification, convolution, and normalization are important for stable activation conditioning. We further incorporate lightweight stabilization components, including block-wise normalization and modular training schedules. Experiments on standard and many-class image-classification benchmarks show that ChACo improves over prior local-learning baselines, transfers across WAN and ResNet convolutional architectures, and narrows the gap to matched BP references, while end-to-end BP remains stronger in the most depth-dependent settings.

## RELEASE NOTE: Work in Progress. The code will be cleaned up and tested to be published along with the paper publication.
This release keeps the code needed for the paper’s main ChACo experiments:

- direct fine-class and learned association-policy layers;
- WAN- and ResNet-style convolutional backbones;
- CIFAR-10, CIFAR-100, and Tiny-ImageNet-200;
- interleaved and progressive local schedules;
- squared-energy goodness, RCB ordering, block-wise L2 normalization, and Goodness-Margin Adaptive Dropout;
- repeated-seed aggregation.

## Method implemented

For layer \(\ell\), the input is detached and the final paper ordering is used:

\[
Y^{(\ell)} = \operatorname{BN}\!\left(W^{(\ell)} *
\operatorname{ReLU}(\operatorname{stopgrad}(Y^{(\ell-1)}))\right).
\]

The \(C_\ell\) output channels are divided into \(K_\ell\) equal blocks. The squared-energy goodness of block \(k\) is

\[
G^{(\ell)}_{n,k} =
\frac{1}{S_\ell H_\ell W_\ell}
\sum_{s,h,w}\left(Y^{(\ell)}_{n,k,s,h,w}\right)^2,
\qquad S_\ell=C_\ell/K_\ell.
\]

Fine-label logits are obtained through the column-stochastic association matrix \(A_\ell\):

\[
\widetilde G^{(\ell)} = G^{(\ell)}A_\ell.
\]

- A direct fine-class layer uses \(K_\ell=J\) and \(A_\ell=I_J\).
- An association-policy layer uses \(K_\ell<J\) and learns \(A_\ell\) from class-conditioned local block responses.

Only the local fine-label cross-entropy updates \(W^{(\ell)}\). The association scores \(Q_\ell\) are buffers, not gradient-updated parameters, and follow the paper’s slower activity-driven update:

\[
Q_{\ell,kj}\leftarrow(1-\lambda_q)Q_{\ell,kj}
+\eta_q\left(E_{\ell,kj}-\gamma\Delta^{\mathrm{bal}}_{\ell,kj}\right),
\qquad
A_{\ell,:,j}=\operatorname{softmax}(Q_{\ell,:,j}/\tau_\ell).
\]

The activation passed to the next layer is block-wise L2 normalized and detached. In the ResNet-style model, a detached shortcut is added before the local head is evaluated, so the local loss updates only the current block parameters.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the implementation without downloading a dataset:

```bash
python train.py --dry-run --device cpu
python -m unittest discover -s tests -v
```

## Data

### CIFAR-10 and CIFAR-100

`torchvision` downloads these datasets automatically under `--data-root`.

### Tiny-ImageNet-200

Place the dataset at:

```text
data/
└── tiny-imagenet-200/
    ├── train/
    │   ├── n01443537/
    │   └── ...
    └── val/
        ├── n01443537/
        └── ...
```

The official validation images must be reorganized into class directories before using `ImageFolder`. The training split is stratified into optimization and validation subsets; Tiny-ImageNet’s official validation split is used only for the final test evaluation.

## Main experiment commands

All commands below use local training, block-wise L2 normalization, adaptive dropout, and the interleaved schedule by default. `--seeds` accepts a comma-separated list.

### CIFAR-10

Direct fine-class competition is the relevant default because the task has only ten classes.

```bash
python train.py \
  --dataset cifar10 \
  --backbone wan \
  --num-association-layers 0 \
  --seeds 22,13,52 \
  --data-root ./data
```

Use `--backbone resnet` for the matched ResNet-style run.

### Association-policy ablation

Fine-only, one association-policy layer, and two association-policy layers correspond to the three settings in Table IV.

```bash
python train.py --dataset [cifar100/tiny-imagenet-20] --backbone wan --num-association-layers 0 --seeds 22,13,52 --data-root ./data
python train.py --dataset [cifar100/tiny-imagenet-20] --backbone wan --num-association-layers 1 --seeds 22,13,52 --data-root ./data
python train.py --dataset [cifar100/tiny-imagenet-20] --backbone wan --num-association-layers 2 --seeds 22,13,52 --data-root ./data
```

Repeat with `--backbone resnet` for the ResNet rows.

## Outputs and evaluation protocol

Each seed produces a run directory containing:

```text
runs/<experiment>_seed<seed>_<timestamp>/
├── config.json
├── metrics.jsonl
├── best.pt
└── summary.json
```

The code follows this evaluation order:

1. optimize layer-local objectives on the training subset;
2. use only the fixed validation subset for adaptive dropout and best-epoch selection;
3. restore the best final-head validation checkpoint;
4. evaluate the official test split once;
5. report the final local head’s test error.

Aggregate completed seeds with:

```bash
python summarize.py --runs-root ./runs
```

The output is CSV text with mean and sample standard deviation of test error in percent.

## Repository structure

```text
.
├── chaco/
│   ├── data.py       # datasets and stratified validation split
│   ├── layers.py     # local objective, association update, BwL2
│   ├── models.py     # WAN/ResNet topology and local schedules
│   └── training.py   # local training, validation, AdaDrop, checkpoints
├── tests/
│   └── test_chaco.py
├── train.py
├── summarize.py
├── requirements.txt
└── PAPER_CODE_AUDIT.md
```

## Citation

```bibtex
@article{papachristodoulou2025chaco,
  title   = {ChACo: Channel-wise Adaptive Competitive Layer-wise Learning},
  author  = {Papachristodoulou, Andreas and Kyrkou, Christos and
             Timotheou, Stelios and Theocharides, Theocharis},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2026},
  note    = {Accepted manuscript}
}
```
