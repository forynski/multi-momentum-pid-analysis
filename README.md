# Multi-Momentum Particle Identification (PID) Analysis

This repository contains a **fully-featured Jupyter Notebook** implementing advanced particle identification (PID) analysis across multiple momentum ranges, using machine learning techniques optimized with Optuna and featuring model caching for efficient repeated analysis.

## Overview

Particle identification is a core task in high-energy physics experiments, requiring sophisticated algorithms to distinguish particle species based on detector signals. This work provides:

- PID classification for **five particle types**: Pion, Kaon, Proton, Electron, and Deuteron  
- Analysis performed over **four momentum ranges**, including the full spectrum and three specific bins (0.1–1, 1–3, and 3+ GeV/c)  
- **Gradient Boosting Models (XGBoost)** trained independently per momentum range  
- **Automatic hyperparameter optimization via Optuna** for robust model tuning  
- **Model caching** to save and reload trained models and optimization studies, dramatically reducing rerun times  
- Extensive **visualization suite** including BDT output distributions, ROC curves, confusion matrices, and feature importances  
- An interactive dashboard for exploring PID performance across momentum ranges and particles using widgets  
- Quantitative PID performance metrics such as efficiency, purity, and contamination calculated and displayed

## Features

### Multi-Range Analysis

The data is split into four momentum ranges:

- Full Spectrum: all data combined  
- Low Momentum: 0.1–1 GeV/c  
- Mid Momentum: 1–3 GeV/c  
- High Momentum: 3+ GeV/c  

Separate models are trained for each range, enabling detailed performance comparison and momentum-dependent PID insight.

### Optuna Hyperparameter Optimization

- Automatic tuning of key XGBoost hyperparameters including number of trees, depth, learning rate, and sampling fractions  
- Optimization objective is ROC-AUC using a one-vs-one multi-class strategy  
- Completed Optuna studies are saved locally and reused automatically for subsequent runs, avoiding redundant computation  

### Model Caching and Auto-Loading

- Trained models for each momentum range are serialized and saved to Google Drive  
- On subsequent notebook executions, saved models and Optuna studies are loaded automatically to skip expensive training and optimization steps  
- Flags allow forcing re-training or re-optimization if desired

### Visualizations and Dashboard

- Plot distributions of classifier output scores for training and testing data  
- Multi-class ROC curves and confusion matrices per momentum range  
- Feature importance plots to interpret model decisions  
- Interactive widgets let users select particles and PID metrics, toggling between bar charts, tables, and ROC comparisons across ranges  

### Data Handling

- Efficient chunked loading of large CSV datasets  
- Cleaning and handling of missing detector signals and flags for detector hits  
- Mapping of MC truth PDG codes to particle species labels  
- Log transformation and feature scaling for robust model input  

## Usage

1. Mount Google Drive for persistent storage  
2. Install dependencies via `pip`  
3. Run cells sequentially to load data, optimize hyperparameters (or load pre-optimized), train models (or load cached), and generate visualizations  
4. Explore PID performance interactively via the dashboard  

## Advantages

- Modular design allowing easy extension to other particles or features  
- Drastically reduces analysis runtime with caching mechanisms  
- Provides clear insights into momentum-dependent PID performance  
- Combines the latest machine learning practices with robust data preparation  

---

This notebook and repository serve as a ready-to-use, extendable framework for multi-range PID analysis leveraging machine learning, ideal for physicists and data scientists working on particle detector datasets.

