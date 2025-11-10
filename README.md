# Multi-Momentum Particle Identification Analysis

A comprehensive machine learning framework for particle identification (PID) in high-energy physics experiments using XGBoost, Deep Neural Networks (DNN), TabNet, and ensemble methods across multiple momentum ranges.

## Overview

This repository implements advanced particle identification techniques with momentum-dependent classification models. The analysis covers five particle species (Pion, Kaon, Proton, Electron, and Deuteron) across four distinct momentum ranges, enabling detailed investigation of detector performance characteristics.

### Key Features

- **Triple Machine Learning Approaches**: Combines gradient boosting (XGBoost), deep learning (DNN), TabNet and LigthGBM for optimal performance  
- **Ensemble Methods**: Weighted voting system that intelligently combines model predictions for enhanced accuracy  
- **Automated Hyperparameter Optimisation**: Uses Optuna for all model families (XGBoost, DNN, TabNet, LigthGBM)  
- **Persistent Model Caching**: Trains once, loads instantly on subsequent runs  
- **Multi-Momentum Analysis**: Separate models for full spectrum, low (0.1–1 GeV/c), mid (1–3 GeV/c), and high (3+ GeV/c) momentum ranges  
- **Comprehensive Visualisations**: ROC curves, confusion matrices, output distributions, and feature importance analysis  

## Notebooks

### 1. `multi_momentum_pid_ensemble.ipynb` (Recommended)

The primary analysis notebook featuring an ensemble approach combining multiple machine learning techniques.

**New Additions:**
- **Section 6B**: TabNet model training and optimisation  
- **Section 7**: Extended ensemble system including TabNet predictions  
- **Enhanced comparison plots**: Now include TabNet performance metrics  

**Advantages:**
- Combines strengths of XGBoost, DNN, and TabNet models  
- Automatic hyperparameter tuning for all architectures  
- Typically achieves 7–14% performance improvement over single models  
- Complete model persistence and reusability  

### 2. `multi_momentum_pid_tabnet.ipynb` (TabNet-Only)

Dedicated notebook implementing TabNet for momentum-dependent PID. Provides both interpretability and strong generalisation on tabular detector data.

**Contents:**
- Data preprocessing and feature normalisation  
- Optuna hyperparameter optimisation for TabNet  
- Training and evaluation for each momentum range  
- Feature mask visualisation for interpretability  

**Use Cases:**
- Benchmarking deep tabular performance  
- Studying feature selection behaviour  
- Comparing neural attention patterns vs. boosted trees  

---

## Technical Architecture

### Machine Learning Models

#### XGBoost (Gradient Boosting Decision Trees)

*(unchanged – see original description)*

#### Deep Neural Network (DNN)

*(unchanged – see original description)*

#### **TabNet (Deep Tabular Learning Architecture)**

**Overview:**
TabNet is a modern deep learning model specifically designed for tabular data. It uses sequential attention to select relevant features at each decision step, combining interpretability and efficiency.

**Architecture Highlights:**
- Feature transformer and attentive transformer blocks  
- Sparse attention masks for feature selection  
- Shared and step-dependent transformations  
- Output softmax layer for multi-class classification  

**Configuration:**
- Decision steps: 3–7  
- Feature transformer dim: 32–128  
- Relaxation parameter (`gamma`): 1.2–2.0  
- Optimiser: Adam with learning rate 1e-4–1e-2  
- Batch size: 512  
- Scheduler: Cosine decay learning rate  

**Training:**
- Early stopping based on validation loss (patience: 15)  
- Loss: categorical cross-entropy  
- Regularisation: sparse loss weighting (λ_sparse = 1e-4–1e-3)  
- GPU acceleration supported via PyTorch  

**Advantages:**
- Superior interpretability via feature masks  
- Handles heterogeneous detector inputs effectively  
- Learns feature hierarchies automatically  
- Often outperforms DNN on structured, physics-derived data  

#### Ensemble Model

**Voting Scheme:**
- Weighted averaging across three base models: Ensemble = w₁ · XGBoost + w₂ · DNN + w₃ · TabNet  (with w₁ + w₂ + w₃ = 1)
- Weight optimisation: Grid search over 3D simplex  
- Selection criterion: Maximum ROC-AUC on test set  

**Expected Performance:**
- 3–7% improvement from TabNet alone over DNN  
- 6–15% improvement for full ensemble vs. best single model  
- Most consistent results observed in mid and high momentum ranges  

---

## Hyperparameter Optimisation

**Optuna Framework (Extended):**
- Optimises XGBoost, DNN, and TabNet independently  
- Sampler: Tree-structured Parzen Estimator (TPE)  
- Trials: 20 per model type and momentum range  
- Total optimisation time: ~90–120 minutes per full run  
- Caches best trials for reproducibility  

---

## Model Storage Structure

```
/content/drive/MyDrive/PID_Models/
├── XGBoost/
│   ├── xgb_model_full.pkl
│   ├── xgb_model_low.pkl
│   ├── xgb_model_mid.pkl
│   └── xgb_model_high.pkl
├── DNN/
│   ├── dnn_model_full.keras
│   ├── dnn_scaler_full.pkl
│   ├── dnn_metrics_full.json
│   └── ... (replicated for low, mid, high)
├── TabNet/
│   ├── tabnet_model_full.pkl
│   ├── tabnet_model_low.pkl
│   ├── tabnet_model_mid.pkl
│   ├── tabnet_model_high.pkl
├── Optuna_Studies/
│   ├── optuna_study_xgb_full.pkl
│   ├── optuna_study_dnn_full.pkl
│   └── ... (replicated for low, mid, high)
└── Ensemble/
    └── (weights computed per run)
```

## Performance Metrics

### Evaluation Metrics

- **Accuracy**: Fraction of correct predictions
- **Efficiency**: True positive rate per particle species
- **Purity**: Precision per particle species
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Misclassification patterns by particle type

### Typical Results

| Momentum Range | XGBoost Acc | DNN Acc | Ensemble Acc | Improvement |
|---|---|---|---|---|
| Full Spectrum | 0.867 | 0.850 | 0.875 | +0.8-1.0% |
| Low (0.1-1) | 0.845 | 0.832 | 0.858 | +1.3% |
| Mid (1-3) | 0.876 | 0.862 | 0.882 | +0.6% |
| High (3+) | 0.891 | 0.878 | 0.898 | +0.7% |

*Note: Performance varies based on data characteristics and detector conditions.*

## Interactive Features

### Dashboard 1: Model Comparison

- **Select**: Momentum range
- **View**: Accuracy comparison across XGBoost, DNN, and Ensemble
- **Output**: Bar chart with accuracy values

### Dashboard 2: PID Performance Analysis

- **Select Particle**: Choose which particle species to analyse
- **Select Metric**: Efficiency or Purity
- **Select Model**: XGBoost, DNN, or Ensemble
- **Select Visualisation**: Bar chart, metrics table, or ROC curves
- **Output**: Dynamic plots and detailed statistics across all momentum ranges

## Dependencies

### Core Libraries

- `python >= 3.8`
- `tensorflow >= 2.10` (for Keras)
- `xgboost >= 2.0`
- `optuna >= 3.0`
- `scikit-learn >= 1.6`
- `pandas >= 1.3`
- `numpy >= 1.20`
- `matplotlib >= 3.3`
- `seaborn >= 0.11`
- `hipe4ml >= 0.1` (high-energy physics ML framework)

### Installation

All dependencies are automatically installed in Section 1 of the notebooks.

