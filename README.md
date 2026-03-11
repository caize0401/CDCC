# UACAN: A Unified Adaptive Alignment Network for Universal Domain Adaptation in Crimp Quality Curve Diagnosis

<div align="justify">
Crimp connections are solderless joints formed by mechanical compression in electrical systems, and their quality is critically important. In industrial settings, the domain distribution disparities caused by multiple specifications of conductor-terminal pairs, coupled with the scarcity of fault samples, severely limit the cross-domain generalization capabilities of diagnostic models. To address the crimp quality diagnosis problem under three typical cross-domain scenarios—closed-set, partial-set, and open-set—this paper proposes a Unified Adaptive Crimp Alignment Network (UACAN). The model employs a dual-input feature encoding structure for joint modeling and introduces a class-conditional domain adversarial mechanism coupled with a category-weighted maximum mean discrepancy alignment strategy to achieve fine-grained, class-conditional feature alignment between the source and target domains. Furthermore, an energy-based unknown category detection mechanism is incorporated to identify potential unknown classes in the target domain, preventing their erroneous alignment with known classes and thereby effectively mitigating negative transfer. A dynamic alignment scheduling strategy is also designed, which adaptively adjusts the alignment strength based on the prediction entropy of the target domain, the proportion of unknown samples, and category distribution characteristics, enabling the model to maintain stable performance across various domain adaptation scenarios. Experimental results demonstrate that, compared to other state-of-the-art methods, the proposed UACAN achieves superior diagnostic performance in various cross-domain tasks, validating its effectiveness and versatility for intelligent crimp quality detection in industrial applications.
</div>

# UACAN — Unified Adaptation for Closed, Partial, and Open-Set Domain Adaptation

A dual-input (curve + hand-crafted features) domain adaptation framework that unifies **closed-set**, **partial-set**, and **open-set** scenarios via multi-branch optimization and dynamic loss scheduling. It learns a shared representation from both modalities, aligns source and target with conditional domain adversarial training and class-weighted MMD, and detects unknown classes with an energy-based criterion.

## Features

- **Dual-input encoding**: Curve encoder \(E_c\) and feature encoder \(E_f\) produce \(F_c\), \(F_f\); fused representation \(F = \phi(F_c, F_f)\).
- **Classifier**: \(P(y|x) = C(F)\) for known classes.
- **Conditional domain adversarial**: \(z = F \oplus P(y|x)\) with gradient reversal; domain discriminator loss \(L_{\text{domain}}\).
- **Class-weighted MMD**: Soft target centers \(\mu_t^k\), class weights \(w_k\); \(L_{\text{cwMMD}} = \sum_k w_k L_{\text{MMD}}^k\).
- **Energy-based unknown detection**: \(E(x) = -T \log \sum_k e^{f_k/T}\); samples with \(E(x) > \delta\) are predicted as unknown; \(L_{\text{unk}}\) encourages high energy for unknown-like target samples.
- **Dynamic scheduling**: \(\lambda_{\text{cwMMD}} = \lambda_0(1 - \rho)\), \(\lambda_{\text{domain}} = \lambda_0(1 - \text{Var}(w_k))\) to balance alignment and discrimination.

## Project structure

```
UACAN/
├── config.py       # Source/target domains, class sets, λ₀, energy T/δ/margin
├── data.py         # Load data1 (curves) + data2 (35-d features), align by CrimpID
├── model.py       # UACAN: encoders, fusion, classifier, domain disc, energy, cwMMD
├── train.py       # Training loop, evaluation, H-score, confusion matrix export
├── requirements.txt
└── experiments/    # training_log.csv, uacan_best.pt, target_confusion_matrix.xlsx
```

## Data

- **data1**: Raw crimp force curves  
  - `crimp_force_curves_dataset_05.pkl`, `crimp_force_curves_dataset_035.pkl`
- **data2**: Hand-crafted features (35-D per sample)  
  - `features_05.pkl`, `features_035.pkl`
- Default paths: `datasets/data1` and `datasets/data2` under the project parent (override with `--data_dir`).
- **Domains**: `05` (0.5) and `035` (0.35) — two wire/process conditions.
- **Labels 1–5**: OK, one missing strand, two missing strands, three missing strands, crimped insulation.  
  Source and target class sets are configurable; common classes = known, target-private = unknown at test time.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `numpy`, `pandas`, `scikit-learn`, `openpyxl` (for Excel export).

## Usage

From the project root (e.g. `2026/task1/UACAN`):

**Default** (source 05, target 035; source classes 1,2,3,5; target classes 1,3,5):

```bash
python train.py
```

**Custom domains and classes:**

```bash
python train.py --source_domain 05 --target_domain 035 --source_classes 1,2,3,5 --target_classes 1,3,5
```

**Custom data root** (must contain `data1/` and `data2/`):

```bash
python train.py --data_dir /path/to/datasets
```

**Other common options:**

```bash
python train.py --epochs 50 --batch_size 64 --lr 0.001 \
  --lambda_0 1.0 --energy_delta 0 --energy_margin 1.0 --lambda_unk 0.1 \
  --out_dir experiments
```

## Outputs

- **During training**: Per-epoch source ACC, target ACC, H-SCORE, train loss.
- **After training**:
  - `experiments/training_log.csv`: Epoch-wise source/target ACC, H-SCORE, train loss.
  - `experiments/uacan_best.pt`: Model state and config (dims, num_known, etc.).
  - `experiments/target_confusion_matrix.xlsx`: Target-domain confusion matrix (known classes + unknown). Falls back to CSV if `openpyxl` is unavailable.

## License

See repository license. This implementation is for research and reproducibility.

