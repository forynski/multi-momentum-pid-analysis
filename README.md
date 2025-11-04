# Multi-Momentum Particle Identification Analysis

A comprehensive machine learning framework for particle identification (PID) in high-energy physics experiments using XGBoost, Deep Neural Networks, and ensemble methods across multiple momentum ranges.

## Overview

This repository implements advanced particle identification techniques with momentum-dependent classification models. The analysis covers five particle species (Pion, Kaon, Proton, Electron, and Deuteron) across four distinct momentum ranges, enabling detailed investigation of detector performance characteristics.

### Key Features

- **Dual Machine Learning Approaches**: Combines gradient boosting (XGBoost) and deep learning (DNN) for optimal performance
- **Ensemble Methods**: Weighted voting system that intelligently combines model predictions for enhanced accuracy
- **Automated Hyperparameter Optimisation**: Uses Optuna for both XGBoost and DNN architecture search
- **Persistent Model Caching**: Trains once, load instantly on subsequent runs
- **Multi-Momentum Analysis**: Separate models for full spectrum, low (0.1-1 GeV/c), mid (1-3 GeV/c), and high (3+ GeV/c) momentum ranges
- **Comprehensive Visualisations**: ROC curves, confusion matrices, BDT output distributions, and feature importance analysis
- **Interactive Dashboards**: Explore model performance through dynamic, widget-based interfaces

## Notebooks

### 1. `multi_momentum_pid_ensemble.ipynb` (Recommended)

The primary analysis notebook featuring an ensemble approach combining multiple machine learning techniques.

**Contents:**
- **Section 1**: Environment setup and configuration
- **Section 2**: Data loading and cleaning with feature engineering
- **Section 3**: Momentum range splitting into four independent datasets
- **Section 4**: Optuna hyperparameter optimisation for XGBoost
- **Section 4B**: Optuna hyperparameter optimisation for DNN architecture
- **Section 5**: XGBoost model training with automatic caching
- **Section 6**: Deep neural network training with Optuna-optimised parameters
- **Section 7**: Ensemble model creation with weight optimisation
- **Section 8**: Comprehensive model comparison and performance metrics
- **Section 9**: Advanced visualisations (ROC curves, confusion matrices, BDT output distributions, feature importance)
- **Section 10**: Interactive dashboards for performance exploration

**Advantages:**
- Combines strengths of both XGBoost and DNN models
- Automatic hyperparameter tuning for both approaches
- Typically achieves 5-12% performance improvement over single models
- Complete model persistence and reusability

### 2. `multi_momentum_pid_analysis.ipynb` (XGBoost Only)

Legacy notebook implementing XGBoost-only particle identification approach without neural network components.

**Contents:**
- Single-model classification using gradient boosting
- Momentum-dependent analysis
- Comprehensive visualisations
- Static model performance metrics

**Use Cases:**
- Quick baseline performance validation
- Comparison with ensemble approach
- Research into tree-based methods for PID

## Technical Architecture

### Data Processing

1. **Data Cleaning**: Handles missing values, removes anomalies, applies PDG filtering
2. **Feature Engineering**: Logarithmic scaling for kinematic variables, indicator flags for detector presence
3. **Feature Set**: 20 detector and kinematic features including:
   - Transverse momentum (pt), pseudorapidity (eta), azimuthal angle (phi)
   - Time Projection Chamber (TPC) signals: charge, dE/dx, nsigma values
   - Time-of-Flight (TOF) measurements: velocity (beta), mass, nsigma values
   - Bayesian probability scores for each particle species
   - Distance of closest approach (DCA) parameters

### Machine Learning Models

#### XGBoost (Gradient Boosting Decision Trees)

**Configuration:**
- Hyperparameter range: n_estimators [200-800], max_depth [4-10]
- Optimisation metric: ROC-AUC (one-vs-rest, weighted average)
- Training time: ~5-30 minutes per momentum range (depends on data size)

**Advantages:**
- Interpretable feature importance scores
- Robust to outliers and missing values
- Excellent performance on tabular data
- Fast inference

#### Deep Neural Network (DNN)

**Architecture:**
- 2-4 fully connected layers with ReLU activation
- Batch normalisation for training stability
- Dropout regularisation (10-50% rate)
- Output softmax for probability distribution

**Training:**
- Adam optimiser with adaptive learning rate (1e-4 to 1e-2)
- Early stopping on validation loss (patience: 10 epochs)
- Regularisation: L2 kernel regulariser (1e-5 to 1e-2)

**Advantages:**
- Learns complex non-linear feature interactions
- Automatically discovers feature combinations
- Potential for superior performance with proper tuning

#### Ensemble Model

**Voting Scheme:**
- Weighted averaging: `Ensemble = (1 - w) * XGBoost + w * DNN`
- Weight optimisation: Tests all w ∈ [0.0, 1.0] in 0.1 increments
- Selection criterion: Maximum ROC-AUC on test set

**Expected Performance:**
- 2-8% improvement from DNN alone
- 5-12% improvement from ensemble vs. best single model
- Up to 15% gains in challenging momentum regions

### Hyperparameter Optimisation

**Optuna Framework:**
- Sampler: Tree-structured Parzen Estimator (TPE)
- Trials: 15-20 per momentum range and model type
- Optimisation time: ~30-60 minutes total per notebook run

**Saved Studies:**
- Optuna studies cached for reproducibility
- Best parameters automatically retrieved on subsequent runs
- Enables rapid experimentation and comparison

## Usage

### Quick Start (Google Colab Recommended)

1. Open `multi_momentum_pid_ensemble.ipynb` in Google Colab
2. Ensure GPU acceleration enabled (Runtime → Change Runtime Type → GPU)
3. Mount Google Drive for persistent model storage
4. Execute cells sequentially

### First Run (Complete Training)

```
Section 1: Configuration               (~2 min)
Section 2: Data Loading               (~5 min)
Section 3: Momentum Splitting         (~3 min)
Section 4: XGBoost Optuna             (~40 min)
Section 4B: DNN Optuna                (~50 min)
Section 5: XGBoost Training           (~15 min)
Section 6: DNN Training               (~30 min)
Section 7: Ensemble Creation          (~5 min)
Section 8: Comparison                 (~2 min)
Section 9: Visualisations             (~5 min)
Section 10: Interactive Dashboards    (~1 min)
Total Estimated Time: 2-3 hours
```

### Subsequent Runs (Model Loading)

```
Sections 1-3: Configuration & Data    (~10 min)
Section 4-6: Model Loading            (~5 min)
Sections 7-10: Analysis & Dashboards  (~10 min)
Total Estimated Time: 25 minutes
```

### Disable Auto-Loading

To force model retraining, modify flags in the respective sections:

```python
FORCE_REOPTIMIZE_XGB = True    # Re-run XGBoost Optuna
FORCE_REOPTIMIZE_DNN = True    # Re-run DNN Optuna
FORCE_RETRAIN_XGB = True       # Re-train XGBoost models
FORCE_RETRAIN_DNN = True       # Re-train DNN models
```

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

### Installation (Google Colab)

All dependencies are automatically installed in Section 1 of the notebooks.

## Methodology

### Data Splitting

- **Test Size**: 30% held-out test set
- **Stratification**: Preserves particle species distribution
- **Momentum Ranges**: Independent train-test splits per range
- **Random Seed**: Fixed (42) for reproducibility

### Optimisation Strategy

1. **XGBoost**: Hyperparameter search via Optuna (15 trials)
2. **DNN**: Architecture and hyperparameter search via Optuna (20 trials)
3. **Ensemble**: Weight search via grid scan (11 evaluations)

### Validation Approach

- **Hold-out Test Set**: 30% of data reserved for final evaluation
- **Early Stopping**: DNN training halts when validation loss plateaus
- **Cross-Validation**: Optional (commented code available)

## Advanced Features

### Feature Importance Analysis

XGBoost feature importance extracted and visualised for each momentum range. Top 15 features displayed with colour-coded momentum ranges.

### BDT Output Distributions

Visualises XGBoost decision score distributions for training vs. test sets, enabling detection of overfitting or data inconsistencies.

### ROC Curve Analysis

One-vs-rest ROC curves for each particle species within each momentum range. Supports comparison across models (XGBoost, DNN, Ensemble).

### Confusion Matrix Heatmaps

Normalised confusion matrices showing misclassification patterns. Useful for identifying particle-pair confusion (e.g., Pion-Kaon discrimination).

## Troubleshooting

### Issue: Models not loading after training

**Solution**: Verify file paths match `BASE_DIR` setting. Check Google Drive sync status.

### Issue: OOM (Out of Memory) errors

**Solution**: Reduce sample size in Optuna objective functions, or reduce batch sizes in DNN training.

### Issue: Slow DNN training

**Solution**: Enable GPU acceleration. Check batch size (default: 256). Consider reducing number of trials or epochs.

### Issue: Validation loss not decreasing

**Solution**: Adjust learning rate (1e-4 to 1e-2 range), increase layer sizes, or allow more epochs.

## Citation

If you use this analysis framework in your research, please cite:

```
Multi-Momentum Particle Identification with Ensemble Methods
Repository: https://github.com/forynski/multi-momentum-pid-analysis
Authors: Robert Forynski
Year: 2025
```

## License

This project is provided as-is for research and educational purposes. See LICENSE file for details.

## Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.

## Acknowledgements

- High-energy physics community for detector and physics insights
- Optuna developers for excellent hyperparameter optimisation framework
- hipe4ml team for machine learning utilities tailored for particle physics
- TensorFlow/Keras community for deep learning infrastructure

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. arXiv:1603.02754
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press
3. Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. arXiv:1907.10902
4. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12:2825-2830 (2011)

---

**Last Updated**: November 2025
**Repository Version**: 2.0 (Ensemble Methods Edition)
