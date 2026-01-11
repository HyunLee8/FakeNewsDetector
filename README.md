# LSTM Fake News Detector

A deep learning-based binary classifier that detects fake news articles using Long Short-Term Memory (LSTM) networks with word embeddings.

## Overview

This project implements a recurrent neural network architecture to classify news articles as real or fake. The model leverages sequential text processing through stacked LSTM layers and trained word embeddings to capture semantic patterns indicative of misinformation.

## Model Architecture

**Network Structure:**
- **Embedding Layer**: Converts tokenized text into dense vectors (40 dimensions)
- **Stacked LSTM Layers**: 
  - First LSTM: 100 units with sequence return for temporal feature extraction
  - Second LSTM: 100 units for final sequence encoding
- **Dropout Regularization**: 30% dropout to prevent overfitting
- **Output Layer**: Sigmoid activation for binary classification

**Training Configuration:**
- Loss Function: Binary cross-entropy
- Optimizer: Adam
- Metrics: Accuracy

## Key Features

### 1. **Sequence Padding**
Normalizes variable-length text inputs to fixed `max_length` for batch processing.

### 2. **Threshold Optimization**
Implements custom threshold tuning to maximize classification accuracy beyond default 0.5 cutoff:

```python
def best_threshold_value(model, thresholds, X_test, y_test):
    # Tests multiple classification thresholds
    # Returns accuracy metrics for each threshold
    # Enables optimal decision boundary selection
```

### 3. **Model Callbacks**
- **EarlyStopping**: Prevents overfitting by monitoring validation performance
- **ModelCheckpoint**: Saves best model weights during training

## Technical Specifications

- **Embedding Dimension**: 40 features per word
- **LSTM Hidden Units**: 100 per layer
- **Dropout Rate**: 0.3
- **Vocabulary Size**: Variable (based on dataset)
- **Sequence Length**: Variable (padded to `max_length`)

## Installation

```bash
pip install tensorflow scikit-learn pandas numpy
```

## Usage

```python
# Build the model
model = build_model(vocab_size=10000, max_length=500)

# Train with callbacks
model.fit(X_train, y_train, 
          validation_data=(X_val, y_val),
          epochs=20,
          callbacks=[EarlyStopping(patience=3), 
                     ModelCheckpoint('best_model.h5')])

# Optimize classification threshold
thresholds = np.arange(0.3, 0.8, 0.05)
results = best_threshold_value(model, thresholds, X_test, y_test)
```

## Model Performance

The threshold optimization function enables fine-tuning of the decision boundary, typically improving accuracy over the default 0.5 threshold by accounting for class imbalance or misclassification costs.

## Future Improvements

- Implement attention mechanisms for better interpretability
- Add bidirectional LSTM layers for context from both directions
- Experiment with pre-trained embeddings (GloVe, Word2Vec)
- Incorporate metadata features (source, publication date)

## Dependencies

- TensorFlow/Keras
- NumPy
- Pandas
- Scikit-learn

---

**Note**: This implementation focuses on binary classification (real vs. fake) and can be extended for multi-class fake news categorization.