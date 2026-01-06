# CDCC: Cross-Domain Crimp Quality Classification

A comprehensive deep learning framework for crimp quality classification with cross-domain adaptation capabilities. This project addresses the challenge of classifying crimp quality across different wire cross-section sizes (0.35 and 0.5) using both raw force curves and handcrafted features.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Models](#models)
- [Results](#results)

## 🎯 Overview

This project implements a complete pipeline for crimp quality classification, including:

1. **Data Analysis and Feature Extraction**: Comprehensive analysis of force curves and extraction of discriminative features
2. **Single-Domain Experiments**: Baseline comparisons and proposed IHGNet model
3. **Cross-Domain Experiments**: Baseline comparisons and proposed DAHDANet model for domain adaptation
4. **Explainability Analysis**: Visualization and interpretation of model decisions

The dataset consists of crimp force curves with two wire cross-section sizes (0.35 and 0.5), and the goal is to classify crimp quality into multiple categories (e.g., OK, crimped insulation, one missing strand, two missing strands, etc.).

## 📁 Project Structure

```
CDCC/
├── 01-Data-Analysis-And-Feature-Extraction/
│   └── task1/                    # Data visualization, feature extraction, and analysis
│       ├── curve_visualization*.py
│       ├── feature_extraction.py
│       ├── feature_selection_analysis.py
│       └── visualization_analysis*.py
│
├── 02-Single-Domain-Experiments/
│   ├── task2/                     # Baseline models for single-domain experiments
│   │   ├── task2_2/               # Single-path input experiments
│   │   └── task2_3/               # Dual-path input experiments
│   └── task3/                     # Proposed IHGNet model
│       ├── models/
│       │   ├── single_model_035_parameter.py
│       │   └── single_model_05_parameter.py
│       ├── train_test_035_parameter.py
│       └── train_test_05_parameter.py
│
├── 03-Cross-Domain-Experiments/
│   ├── task4/                     # Baseline models for cross-domain experiments
│   │   ├── 035→05/                # Train on 0.35, test on 0.5
│   │   ├── 05→035/                # Train on 0.5, test on 0.35
│   │   └── 05↔035/                # Bidirectional experiments
│   └── task5/                     # Proposed DAHDANet model
│       ├── models/
│       │   ├── domain_adversarial_fusion_035→05_parameter.py
│       │   └── domain_adversarial_fusion_05→035_parameter.py
│       ├── train_035→05_parameter.py
│       └── train_05→035_parameter.py
│
└── 04-Explainability-Analysis/
    ├── task6/                     # Feature visualization and class alignment analysis
    │   ├── visualize_tsne.py
    │   ├── plot_class_alignment*.py
    │   └── plot_class_distance_scatter.py
    └── task7/                     # SHAP attribution analysis
        └── shap_attribution_waterfall.py
```

## ✨ Features

### Data Processing
- **Force Curve Visualization**: High-quality visualization of crimp force curves
- **Feature Extraction**: Automatic extraction of 35 handcrafted features from force curves
- **Feature Selection**: Statistical analysis and selection of discriminative features
- **Distribution Analysis**: Comprehensive analysis of feature distributions across domains

### Single-Domain Models
- **Baseline Models** (Task 2):
  - Random Forest
  - Multi-Layer Perceptron (MLP)
  - XGBoost
  - H2O AutoML
  - 1D CNN
- **Proposed Model** (Task 3): **IHGNet** (Interactive Hybrid Gated Network)
  - Dual-path architecture (raw curves + handcrafted features)
  - Gated residual blocks for stable training
  - Multi-head feature interaction modules
  - Progressive fusion strategy

### Cross-Domain Models
- **Baseline Models** (Task 4):
  - Traditional ML models (RF, MLP, XGBoost, AutoML)
  - Deep learning models (CNN1D, Transformer)
  - Hybrid fusion models
- **Proposed Model** (Task 5): **DAHDANet** (Domain Adversarial Hybrid Dual-path Attention Network)
  - Domain adversarial training
  - Maximum Mean Discrepancy (MMD) loss
  - Exponential Moving Average (EMA) mechanism
  - Dual-path feature fusion

### Explainability
- **t-SNE Visualization**: Low-dimensional embedding visualization
- **Class Alignment Analysis**: Analysis of class distributions across domains
- **SHAP Attribution**: Feature importance analysis using SHAP values

## 🚀 Installation
conda create -n myenv python=3.9
conda activate CDCC
pip install -r requirements.txt

## 💻 Usage

### 1. Data Analysis and Feature Extraction

```bash
# Visualize force curves
python 01-Data-Analysis-And-Feature-Extraction/task1/curve_visualization_real_data.py

# Extract features
python 01-Data-Analysis-And-Feature-Extraction/task1/feature_extraction.py

# Feature selection analysis
python 01-Data-Analysis-And-Feature-Extraction/task1/feature_selection_analysis.py
```

### 2. Single-Domain Experiments

#### Proposed IHGNet Model (Task 3)

```bash
# Train on 0.35 dataset
python 02-Single-Domain-Experiments/task3/train_test_035_parameter.py --size 035 --epochs 100 --lr 0.001 --out experiments_single/035

# Train on 0.5 dataset
python 02-Single-Domain-Experiments/task3/train_test_05_parameter.py --size 05 --model hybrid_fusion_v13 --epochs 100 --lr 0.001 --out experiments_single/05
```

### 3. Cross-Domain Experiments

#### Proposed DAHDANet Model (Task 5)

```bash
# Train on 0.35, test on 0.5
python 03-Cross-Domain-Experiments/task5/train_035→05_parameter.py --source 035 --target 05 --epochs 100 --batch_size 64 --lr 0.001 --lambda_mmd 0.3 --lambda_aux 1.7 --ema_decay 0.95 --out experiments/035→05

# Train on 0.5, test on 0.35
python 03-Cross-Domain-Experiments/task5/train_05→035_parameter.py --source 05 --target 035 --epochs 100 --batch_size 64 --lr 0.001 --lambda_mmd 1.0 --lambda_aux 0.3 --out experiments/05→035
```

### 4. Explainability Analysis

```bash
# t-SNE visualization
python 04-Explainability-Analysis/task6/visualize_tsne.py

# Class alignment analysis
python 04-Explainability-Analysis/task6/plot_class_alignment_improved.py

# SHAP attribution
python 04-Explainability-Analysis/task7/shap_attribution_waterfall.py
```

## 🔬 Experiments

### Single-Domain Experiments

**Task 2**: Baseline model comparisons
- **Input Types**: 
  - Single-path: Raw curves OR handcrafted features OR concatenated
  - Dual-path: Raw curves + handcrafted features (separate paths)
- **Models**: Random Forest, MLP, XGBoost, H2O AutoML, CNN1D
- **Datasets**: 0.35 and 0.5 wire cross-section sizes

**Task 3**: Proposed IHGNet
- **Architecture**: Dual-path hybrid network with gated residual blocks
- **Key Components**:
  - Enhanced curve processor with multi-scale feature extraction
  - Advanced feature processor for handcrafted features
  - Progressive fusion module with multi-stage fusion
  - Multi-head feature interaction

### Cross-Domain Experiments

**Task 4**: Baseline model comparisons
- **Transfer Directions**: 
  - 0.35 → 0.5 (train on 0.35, test on 0.5)
  - 0.5 → 0.35 (train on 0.5, test on 0.35)
  - Bidirectional (05↔035)
- **Models**: RF, MLP, XGBoost, AutoML, CNN1D, Transformer, Hybrid models

**Task 5**: Proposed DAHDANet
- **Architecture**: Domain adversarial network with dual-path attention
- **Key Components**:
  - Domain adversarial training with gradient reversal
  - MMD loss for domain alignment
  - EMA mechanism for model stability
  - Progressive fusion with attention mechanisms

## 🏗️ Models

### IHGNet (Task 3)

The Interactive Hybrid Gated Network (IHGNet) is designed for single-domain crimp quality classification:

- **Dual-Path Architecture**: 
  - Path 1: Raw force curve (500 points) → Enhanced Curve Processor
  - Path 2: Handcrafted features (35 features) → Advanced Feature Processor
  
- **Key Modules**:
  - `GatedResidualBlock`: Provides stable gradient flow and feature selection
  - `MultiHeadFeatureInteraction`: Enhances feature representation through multi-head attention
  - `ProgressiveFusionModule`: Multi-stage fusion strategy for optimal feature combination
  - `EnhancedCurveProcessor`: Multi-scale feature extraction from raw curves

### DAHDANet (Task 5)

The Domain Adversarial Hybrid Dual-path Attention Network (DAHDANet) addresses cross-domain adaptation:

- **Domain Adaptation Mechanisms**:
  - Domain adversarial training with gradient reversal layer
  - Maximum Mean Discrepancy (MMD) loss for distribution alignment
  - Exponential Moving Average (EMA) for model stability
  
- **Architecture**:
  - Shared feature extractors for both domains
  - Domain classifier for adversarial training
  - Label classifier for task-specific learning
  - Auxiliary classifiers for multi-task learning

## 📊 Results

The project includes comprehensive experimental results:

- **Single-Domain Performance**: Comparison of baseline models vs. IHGNet
- **Cross-Domain Performance**: Comparison of baseline models vs. DAHDANet
- **Visualization Results**: 
  - Force curve visualizations
  - Feature distribution comparisons
  - t-SNE embeddings
  - Class alignment plots
  - SHAP attribution waterfall charts
  - Confusion matrices
  - Radar charts for transfer performance

All results are saved in respective experiment directories with detailed metrics (accuracy, precision, recall, F1-score).

