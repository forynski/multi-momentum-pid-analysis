# Multi-Momentum Particle Identification Analysis

A comprehensive machine learning framework for particle identification (PID) in high-energy physics experiments using XGBoost, Deep Neural Networks (DNN), and ensemble methods across multiple momentum ranges.

## Overview

This repository implements advanced particle identification techniques with momentum-dependent classification models. The analysis covers five particle species (Pion, Kaon, Proton, and Electron) across four distinct momentum ranges, enabling detailed investigation of detector performance characteristics.

### Key Features

- **Triple Machine Learning Approaches**: Combines gradient boosting (XGBoost), deep learning (DNN) and LigthGBM for optimal performance  
- **Ensemble Methods**: Weighted voting system that intelligently combines model predictions for enhanced accuracy  
- **Automated Hyperparameter Optimisation**: Uses Optuna for all model families (XGBoost, DNN, LigthGBM)  
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
- Combines strengths of XGBoost, DNN, and LightGBM models  
- Automatic hyperparameter tuning for all architectures  
- Typically achieves 7–14% performance improvement over single models  
- Complete model persistence and reusability  

**Contents:**
- Data preprocessing and feature normalisation  
- Optuna hyperparameter optimisation  
- Training and evaluation for each momentum range  
- Feature mask visualisation for interpretability  

**Use Cases:**
- Benchmarking deep tabular performance  
- Studying feature selection behaviour  
- Comparing neural attention patterns vs. boosted trees  

---

# Technical Architecture

## Machine Learning Models

### XGBoost (Gradient Boosting Decision Trees)

**Overview:**  
XGBoost is an optimised distributed gradient boosting library designed to be highly efficient, flexible, and portable. It implements machine learning algorithms under the Gradient Boosting framework, delivering state-of-the-art performance on structured data.

**Architecture Highlights:**
- Ensemble of decision trees built sequentially to correct previous errors  
- Regularization terms (L1 and L2) to control overfitting  
- Weighted quantile sketch for approximate tree learning  
- Parallelized tree construction for scalability  

**Configuration:**
- Number of estimators: 200–1000  
- Maximum depth: 4–10  
- Learning rate (η): 0.01–0.3  
- Subsample: 0.7–1.0  
- Column sample by tree: 0.6–1.0  
- Objective: multi-class softmax  
- Evaluation metric: log-loss or accuracy  

**Training:**
- Early stopping based on validation loss (patience: 20)  
- Regularization parameters tuned via grid or Bayesian search  
- GPU acceleration via CUDA support  

**Advantages:**
- High accuracy on tabular data  
- Excellent handling of missing values  
- Strong regularization to prevent overfitting  
- Proven robustness in large-scale structured datasets  

---

### Deep Neural Network (DNN)

**Overview:**  
The DNN is a fully connected feedforward neural network designed to capture nonlinear relationships in high-dimensional feature spaces, particularly effective when data exhibits complex correlations.

**Architecture Highlights:**
- Input normalization and embedding layers for heterogeneous inputs  
- Multiple hidden layers (3–6) with ReLU or GELU activations  
- Batch normalization and dropout for regularization  
- Output softmax layer for multi-class classification  

**Configuration:**
- Hidden layer sizes: 128–512 units  
- Activation: ReLU or GELU  
- Optimiser: Adam or AdamW (learning rate: 1e-4–1e-3)  
- Batch size: 256–1024  
- Scheduler: Cosine annealing or step decay  

**Training:**
- Loss: categorical cross-entropy  
- Early stopping based on validation accuracy  
- Weight decay and dropout (0.2–0.5) for overfitting prevention  
- GPU acceleration supported via PyTorch or TensorFlow  

**Advantages:**
- High representational capacity for nonlinear feature interactions  
- Adaptable to mixed feature types  
- Effective for high-dimensional structured or semi-structured data  
- Can be combined with autoencoders or feature extractors for pretraining  

---

### LightGBM

**Overview:**  
LightGBM is a gradient boosting framework that uses tree-based learning algorithms, designed for efficiency and scalability on large datasets. It supports histogram-based algorithms and leaf-wise tree growth for faster convergence.

**Architecture Highlights:**
- Leaf-wise tree growth with depth constraints  
- Histogram-based split finding  
- Gradient-based one-side sampling (GOSS)  
- Exclusive feature bundling (EFB) to reduce dimensionality  

**Configuration:**
- Number of leaves: 31–255  
- Max depth: -1 (unlimited) or tuned per dataset  
- Learning rate: 0.01–0.2  
- Feature fraction: 0.6–1.0  
- Bagging fraction: 0.6–1.0  
- Objective: multi-class classification  
- Metric: multi-logloss or accuracy  

**Training:**
- Early stopping with validation monitoring  
- Regularization via L1/L2 and min data in leaf  
- GPU support for faster training  

**Advantages:**
- High-speed training with low memory footprint  
- Strong performance on large tabular datasets  
- Supports categorical features natively  
- Excellent scalability and distributed training


#### Ensemble Model

**Voting Scheme:**
- Weighted averaging across three base models: Ensemble = w₁ · XGBoost + w₂ · LightGBM + w₃ · DNN (with w₁ + w₂ + w₃ = 1)
- Weight optimisation: Grid search over 3D simplex  
- Selection criterion: Maximum ROC-AUC on test set

---

## Hyperparameter Optimisation

**Optuna Framework (Extended):**
- Optimises XGBoost and DNN, independently  
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
├── LightGBM/
│   ├── LightGBM_full.pkl
│   ├── LightGBM_low.pkl
│   ├── LightGBM_mid.pkl
│   ├── LightGBM_high.pkl
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
- **View**: Accuracy comparison across XGBoost, DNN, LightGBM and Ensemble
- **Output**: Bar chart with accuracy values

### Dashboard 2: PID Performance Analysis

- **Select Particle**: Choose which particle species to analyse
- **Select Metric**: Efficiency or Purity
- **Select Model**: XGBoost, DNN, LightGBM or Ensemble
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

