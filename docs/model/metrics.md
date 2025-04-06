# Model Evaluation Metrics

This document explains all metrics used to evaluate our recommendation model's performance.

## Loss Metrics

### Mean Squared Error (MSE)
- **Definition**: Average squared difference between predicted and actual ratings
- **Formula**: \( MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 \)
- **Interpretation**:
  - Lower values indicate better prediction accuracy
  - Values closer to 0 are better
  - Sensitive to outliers (large errors are penalized more)
- **Range**: 0 to ∞ (unbounded)
- **Good Values**: Typically < 1.0 for rating prediction

## Classification Metrics

### Precision
- **Definition**: Proportion of recommended items that are actually relevant
- **Formula**: \( Precision = \frac{True\ Positives}{True\ Positives + False\ Positives} \)
- **Interpretation**:
  - Measures recommendation accuracy
  - Higher values mean fewer irrelevant recommendations
  - Important when user satisfaction depends on recommendation quality
- **Range**: 0 to 1
- **Good Values**: > 0.7 for high-quality recommendations

### Recall
- **Definition**: Proportion of relevant items that are recommended
- **Formula**: \( Recall = \frac{True\ Positives}{True\ Positives + False\ Negatives} \)
- **Interpretation**:
  - Measures recommendation coverage
  - Higher values mean fewer missed relevant items
  - Important when completeness of recommendations matters
- **Range**: 0 to 1
- **Good Values**: > 0.5 for good coverage

### F1 Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: \( F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \)
- **Interpretation**:
  - Balances precision and recall
  - Useful when both accuracy and coverage are important
  - Higher values indicate better overall performance
- **Range**: 0 to 1
- **Good Values**: > 0.6 for balanced performance

## Training vs Validation Metrics

### Training Metrics
- **Purpose**: Measure model's performance on training data
- **Interpretation**:
  - High training precision/recall with low validation metrics indicates overfitting
  - Consistently low values may indicate underfitting
  - Should be monitored alongside validation metrics

### Validation Metrics
- **Purpose**: Measure model's generalization to unseen data
- **Interpretation**:
  - More reliable indicator of real-world performance
  - Should be close to training metrics for good generalization
  - Used to select best model version

## Metric Trade-offs

### Precision vs Recall
- **High Precision, Low Recall**:
  - Few but highly relevant recommendations
  - Good for users who value accuracy over coverage
- **Low Precision, High Recall**:
  - Many recommendations with some irrelevant ones
  - Good for users who want to see all relevant items
- **Balanced Performance**:
  - F1 score helps identify good balance
  - Target depends on application requirements

## Best Practices

### Monitoring Metrics
1. **During Training**:
   - Plot metrics over epochs
   - Look for convergence patterns
   - Identify overfitting early

2. **Model Selection**:
   - Use validation metrics for model selection
   - Consider both precision and recall
   - Balance based on application needs

3. **Threshold Selection**:
   - Default threshold: 3.5 (like/dislike boundary)
   - Can be adjusted based on requirements
   - Affects precision-recall trade-off

### Interpreting Results
1. **Good Model**:
   - Validation metrics close to training metrics
   - Both precision and recall above 0.5
   - F1 score above 0.6

2. **Overfitting**:
   - Large gap between training and validation metrics
   - High training precision/recall but low validation
   - Solution: Regularization, early stopping

3. **Underfitting**:
   - Consistently low metrics on both sets
   - Solution: Increase model capacity, adjust learning rate

## Example Output Interpretation

```
BEST MODEL SUMMARY
==================================================
Best Epoch: 15
Validation Loss: 0.8567
Training Loss: 0.8321
Validation Precision: 0.7234
Validation Recall: 0.6123
Training Precision: 0.7456
Training Recall: 0.6289
Validation F1 Score: 0.6632
Training F1 Score: 0.6821

Model Configuration:
Embedding Dimension: 50
Learning Rate: 0.001
Batch Size: 1024
Number of Users: 10000
Number of Movies: 5000
==================================================
```

**Interpretation**:
1. **Loss**: Good prediction accuracy (MSE < 1.0)
2. **Precision**: 72% of recommendations are relevant
3. **Recall**: 61% of relevant items are recommended
4. **F1 Score**: 0.66 indicates reasonable balance
5. **Overfitting**: Small gap between training/validation metrics
6. **Configuration**: Standard settings with good performance 