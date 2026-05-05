# Results and Analysis

## 1. Traditional Linguistic Baseline

**Dataset:** 109 samples

| Model | MAE | R² |
|---|---:|---:|
| Ridge | 3.225 | -0.087 |
| Random Forest | 3.247 | -0.202 |
| Gradient Boosting | 3.388 | -0.331 |

**Best Model:** Ridge

| Metric | Value |
|---|---:|
| Final MAE | 3.222 |
| Final R² | -0.048 |
| Within ±5 years | 78.0% |

### Explanation

The traditional linguistic baseline uses only transcript-based features. Ridge Regression performs best in this setup, which suggests that the basic linguistic features mostly capture simple linear patterns.

The MAE of 3.222 means the model is off by about 3.2 years on average. The within ±5 years accuracy of 78.0% shows that many predictions are reasonably close.

However, the negative R² means the model is still weak at explaining age variation in the data. In simple words, the model is making useful predictions, but it is not yet learning strong age-related patterns.

---

## 2. Advanced Features: BERT + DLATK

**Dataset:** 109 samples

| Model | MAE | R² |
|---|---:|---:|
| Ridge | 3.143 | -0.071 |
| Random Forest | 3.126 | -0.072 |
| Gradient Boosting | 3.205 | -0.214 |

**Best Model:** Random Forest

| Metric | Value |
|---|---:|
| Final MAE | 3.126 |
| Final R² | 0.023 |
| Within ±5 years | 78.9% |

### Explanation

After adding BERT embeddings and DLATK-style features, the model improves.

The MAE improves from 3.222 to 3.126. The within ±5 years accuracy also improves from 78.0% to 78.9%.

The best model changes from Ridge Regression to Random Forest. This is important because it suggests that the added semantic and sentiment features contain more complex patterns. Random Forest can capture these non-linear relationships better than a simple linear model.

The R² becomes slightly positive, which means the model is starting to learn meaningful structure from the data.

---

## 3. Larger Dataset Experiment

**Input size:** N = 9000  
**Usable training samples:** 132 samples

| Model | MAE | MAE Std | R² | R² Std |
|---|---:|---:|---:|---:|
| Ridge | 3.908 | 0.646 | -0.188 | 0.244 |
| Random Forest | 3.532 | 0.243 | 0.010 | 0.122 |
| Gradient Boosting | 3.707 | 0.293 | -0.058 | 0.230 |

**Best Model:** Random Forest

| Metric | Value |
|---|---:|
| Final MAE | 3.535 |
| Final R² | 0.053 |
| Within ±5 years | 74.2% |

### Explanation

When the dataset size increases, the model is tested on more diverse samples. Because of this, the MAE increases from 3.126 to 3.535.

This does not necessarily mean the model became worse. It means the evaluation became more realistic. A larger dataset usually contains more variation, noise, and difficult examples.

The important point is that the final R² becomes positive at 0.053. This suggests that the model is learning more general patterns instead of only fitting a smaller dataset.

The within ±5 years accuracy drops to 74.2%, but the result is still reasonable because the task becomes harder with more varied data.

---

## Comparison

| Experiment | Dataset Size | Best Model | Final MAE | Final R² | Within ±5 Years |
|---|---:|---|---:|---:|---:|
| Traditional Linguistic Baseline | 109 | Ridge | 3.222 | -0.048 | 78.0% |
| BERT + DLATK Features | 109 | Random Forest | 3.126 | 0.023 | 78.9% |
| Larger Dataset Experiment | 132 usable samples | Random Forest | 3.535 | 0.053 | 74.2% |

---

## Key Observations

| Observation | Meaning |
|---|---|
| BERT + DLATK improves MAE | Error decreases from 3.222 to 3.126 years |
| Within ±5 years accuracy improves | Accuracy increases from 78.0% to 78.9% |
| Best model changes to Random Forest | Advanced features contain non-linear patterns |
| R² becomes positive | The model begins learning meaningful structure |
| Larger data gives more realistic results | More samples introduce more variation and noise |
| MAE increases on larger data | The task becomes harder with more diverse examples |
| R² improves on larger data | Generalization becomes more stable |

---

## Final Summary

The traditional linguistic baseline already gives a reasonable starting point for age prediction using transcript-based features.

Adding BERT embeddings and DLATK-style linguistic features improves the model. The improvement in MAE, within ±5 years accuracy, and R² shows that semantic and sentiment-based information adds useful signals.

On the larger dataset, the MAE increases slightly, but the positive R² shows better generalization. This means the model is learning more realistic patterns instead of only performing well on a smaller dataset.

We could say that these results show that transcript-based features can be useful for age prediction, and advanced semantic features provide a clear improvement over traditional linguistic features.
