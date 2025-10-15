# BioBERT-based Biomedical Relation Classification

A machine learning project for classifying biomedical relationships using BioBERT embeddings and k-Nearest Neighbors (k-NN) classification. This project focuses on identifying relationships between biological entities in the Gene Ontology (GO) dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔬 Overview

This project implements and compares different approaches for biomedical relation classification using pre-trained BioBERT embeddings. The main goal is to classify relationships between biological entities into predefined categories such as "positively regulates", "part of", "has part", etc.

### Key Features:
- **BioBERT Integration**: Uses `dmis-lab/biobert-v1.1` for biomedical text embeddings
- **Multiple Model Architectures**: Implements both simple and enhanced k-NN classifiers
- **Class Imbalance Handling**: Addresses dataset imbalance using class weights and SMOTE
- **Ensemble Methods**: Combines multiple prediction strategies for improved performance
- **Comprehensive Evaluation**: Detailed classification reports and F1-score analysis

## 📊 Dataset

The project uses the Gene Ontology (GO) relation dataset with the following characteristics:

- **Total Samples**: 17,772 relation pairs
- **Training Set**: 10,538 samples
- **Test Set**: 7,234 samples
- **Relation Types**: 7 different relationship categories
- 
### Dataset Source

- This dataset was obtained from the **LLMs4OL Challenge 2024** - Task C: Non-Taxonomic Relation Extraction, specifically SubTask C.2 (Few-Shot) for Gene Ontology relations. The dataset is part of a research competition focused on Large Language Models for Ontology Learning.

**Source**: [LLMs4OL Challenge 2024 - SubTask C.2(FS) - GO](https://github.com/sciknoworg/LLMs4OL-Challenge/tree/main/2024/TaskC-Non-Taxonomic%20Relation%20Extraction/SubTask%20C.2(FS)%20-%20GO)
### Relation Distribution:

- `part of`: 7,637 samples (43.0%)
- `regulates`: 3,329 samples (18.7%)
- `positively regulates`: 2,903 samples (16.3%)
- `negatively regulates`: 2,901 samples (16.3%)
- `has part`: 792 samples (4.5%)
- `occurs in`: 197 samples (1.1%)
- `happens during`: 13 samples (0.1%)

![Go Distribution](Go_Distribution.png?raw=true "Go Distribution")
## 🚀 Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Required Packages

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install sentence-transformers
pip install scikit-learn
pip install pandas numpy
pip install matplotlib seaborn
pip install imbalanced-learn
```

## 🤖 Models

### 1. BioBERTkNN
- **Architecture**: k-Nearest Neighbors with BioBERT embeddings
- **Features**: Cosine similarity-based classification
- **Best Performance**: Macro F1: 0.4874, Micro F1: 0.5089

### 2. ImprovedRelationClassifier
- **Architecture**: Enhanced k-NN with multiple strategies
- **Features**: 
  - Multi-strategy text representation
  - Class weight balancing
  - SMOTE oversampling
  - Ensemble voting
- **Best Performance**: Macro F1: 0.4562, Micro F1: 0.5030

### 3. ImprovedBioBERTkNN
- **Architecture**: Optimized BioBERT k-NN with ensemble methods
- **Features**:
  - Enhanced text representations
  - Multi-k ensemble voting
  - Weighted similarity scoring
- **Best Performance**: Macro F1: 0.5101, Micro F1: 0.5347

## 📈 Results

### Performance Summary

| Model | Macro F1 | Micro F1 | Accuracy |
|-------|----------|----------|----------|
| Basic BioBERT k-NN | 0.4874 | 0.5089 | 0.49 |
| Enhanced (Class Weights) | 0.4567 | 0.5541 | 0.55 |
| Enhanced (SMOTE) | 0.4300 | 0.4166 | 0.42 |
| Ensemble | 0.4562 | 0.5030 | 0.50 |
| **Improved BioBERT k-NN** | **0.5101** | **0.5347** | **0.53** |


## 🔧 Key Features Implementation

### Text Representation Strategies
1. **BioBERT Format**: `{head} [SEP] {tail}`
2. **Enhanced Context**: `biological relation: {head} [SEP] {tail}`
3. **Template-based**: Multiple domain-specific templates
4. **Ensemble Embeddings**: Combination of different strategies

### Class Imbalance Handling
- **Class Weights**: Computed using `sklearn.utils.class_weight.compute_class_weight`
- **SMOTE Oversampling**: Synthetic minority oversampling technique (Not a good idea for this project)
- **Stratified Splitting**: Maintains class distribution in train/dev splits

### Ensemble Methods
- **Multi-k Voting**: Combines predictions from different k values (3, 5, 7)
- **Confidence-based Weighting**: Uses prediction confidence for ensemble decisions
- **Similarity Weighting**: Exponential similarity weighting for better discrimination

## 🎯 Key Insights

1. **Best Performing Model**: ImprovedBioBERT k-NN with ensemble methods achieved the highest performance
2. **Class Imbalance Impact**: The dataset shows significant class imbalance, affecting minority class performance
3. **BioBERT Effectiveness**: Pre-trained biomedical embeddings provide strong baseline performance
4. **Ensemble Benefits**: Multi-strategy approaches consistently outperform single-method approaches

## 📞 Contact

For questions or suggestions, please open an issue.

---

**Note**: This project is part of a Knowledge Engineering course final project. The dataset and evaluation metrics are specific to the Gene Ontology relation classification task.
