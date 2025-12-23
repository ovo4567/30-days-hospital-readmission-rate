# Hospital Readmission Prediction: A Multi-Modal Deep Learning Approach

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project addresses the critical healthcare challenge of predicting 30-day hospital readmissions using multi-modal clinical data. We develop and evaluate several machine learning approaches, from traditional ML baselines to deep learning ensemble models.

### Key Features
- **Multi-Modal Data Integration**: Combines Electronic Health Records (EHR), clinical notes, and chest X-ray images
- **Deep Learning Ensemble**: XGBoost + GRU ensemble for EHR-only predictions
- **Multi-Modal Fusion**: Bidirectional GRU + TF-IDF + Image features fusion model
- **Comprehensive Analysis**: Full EDA, feature engineering, and model comparison

## Project Structure

```
├── Group_14.ipynb              # Main analysis notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── 2025-fall-stat-3612-group-project/  # Data directory (not tracked)
    ├── train.csv
    ├── valid.csv
    ├── test.csv
    ├── notes.csv
    ├── ehr_preprocessed_seq_by_day_cat_embedding.pkl
    └── image_features/
        └── cxr_features/
```

## Methods

### Track 1: EHR-Only Model
- **XGBoost**: Gradient boosting on last-day EHR features with class imbalance handling
- **GRU**: Recurrent neural network for full sequence modeling
- **Ensemble**: Simple averaging of both models' predictions

### Track 2: Multi-Modal Model
- **EHR Branch**: Bidirectional GRU (hidden_dim=128)
- **Notes Branch**: TF-IDF (5000 features) + MLP compression
- **Image Branch**: Pre-extracted CXR features + MLP compression
- **Fusion**: Concatenation of all modality representations → MLP classifier

## Results

| Track | Model | Validation AUC |
|-------|-------|----------------|
| Track 1 | XGBoost + GRU Ensemble | ~0.78 |
| Track 2 | Multi-Modal Fusion | ~0.80 |

## Setup

1. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**
   - Place the data files in the `2025-fall-stat-3612-group-project/` directory
   - Data is not included in this repository for privacy reasons

4. **Run the notebook**
   ```bash
   jupyter lab Group_14.ipynb
   ```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- XGBoost
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

See `requirements.txt` for full list.

## Usage

Execute the notebook cells in order. The notebook is organized into four main sections:

1. **Data Loading & EDA**: Load data and explore distributions
2. **Baseline Models**: Train and evaluate traditional ML models
3. **Track 1 Model**: XGBoost + GRU ensemble (EHR-only)
4. **Track 2 Model**: Multi-modal fusion model

## Output Files

- `track1_ensemble_xgb_gru.csv`: Track 1 predictions
- `multimodal_123.csv`: Track 2 predictions

## Team

STAT 3612 Group 14

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Course: STAT 3612 - Statistical Machine Learning
- Data: MIMIC-derived clinical dataset (de-identified)