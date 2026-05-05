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

The traditional setup uses only transcript-based linguistic features. Ridge Regression performs best, which shows that the signal captured here is mostly linear.

The MAE of about 3.2 years means predictions are reasonably close. However, the negative R² shows the model is not learning strong patterns and struggles to explain variation in age.

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

Adding BERT embeddings and DLATK-style features improves performance.

The MAE drops from 3.222 to 3.126. Accuracy within ±5 years also improves slightly.

The best model changes to Random Forest, which shows that the new features introduce non-linear patterns. The R² becomes slightly positive, meaning the model starts learning meaningful structure.

---

## 3. Larger Dataset Experiment

**Input size:** N = 9000  
**Usable samples:** 132

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

With more data, the MAE increases slightly because the dataset becomes more diverse and harder.

This is expected. The important improvement is that R² becomes more positive, which means the model generalizes better instead of overfitting.

---

# 4. Feature Ablation Study

To understand the contribution of each feature group, we perform a controlled ablation study.

---

## 4.1 Traditional Features Only

**Samples:** 133  
**Features:** 19  

| Model | MAE | R² |
|---|---:|---:|
| Ridge | 3.865 | -0.259 |
| Random Forest | 3.627 | -0.131 |
| Gradient Boosting | 4.039 | -0.354 |

**Best Model:** Random Forest

| Metric | Value |
|---|---:|
| Final MAE | 3.634 |
| Final R² | -0.051 |
| Within ±5 years | 73.7% |

### Explanation

Using only traditional linguistic features leads to weak performance.

The MAE is relatively high and R² is negative. This confirms that basic features alone are not enough to capture age-related patterns effectively.

---

## 4.2 Traditional + BERT Features

**Samples:** 133  
**Features:** 29  

| Model | MAE | R² |
|---|---:|---:|
| Ridge | 3.728 | -0.193 |
| Random Forest | 3.444 | 0.004 |
| Gradient Boosting | 3.666 | -0.186 |

**Best Model:** Random Forest

| Metric | Value |
|---|---:|
| Final MAE | 3.450 |
| Final R² | 0.043 |
| Within ±5 years | 75.2% |

### Explanation

Adding BERT embeddings gives a clear improvement.

MAE drops from 3.634 to 3.450. R² becomes positive, which shows that semantic understanding from BERT helps the model learn meaningful patterns.

---

## 4.3 Traditional + BERT + DLATK

**Samples:** 133  
**Features:** 41  

| Model | MAE | R² |
|---|---:|---:|
| Ridge | 3.686 | -0.141 |
| Random Forest | 3.440 | 0.016 |
| Gradient Boosting | 3.636 | -0.147 |

**Best Model:** Random Forest

| Metric | Value |
|---|---:|
| Final MAE | 3.447 |
| Final R² | 0.066 |
| Within ±5 years | 75.9% |

### Explanation

Adding DLATK features on top of BERT gives further improvement.

The MAE improves slightly, and R² increases to 0.066, which is the best among all setups. This shows that sentiment, topic, and linguistic style features provide additional useful signals.

---

## 4.4 Final Ablation Summary

| Setup | Best Model | Samples | Features | Final MAE | Final R² | Within ±5 Years |
|---|---|---:|---:|---:|---:|---:|
| Traditional Only | Random Forest | 133 | 19 | 3.634 | -0.051 | 73.7% |
| Traditional + BERT | Random Forest | 133 | 29 | 3.450 | 0.043 | 75.2% |
| Traditional + BERT + DLATK | Random Forest | 133 | 41 | 3.447 | 0.066 | 75.9% |

---

## Key Insights from Ablation

| Observation | Meaning |
|---|---|
| BERT improves MAE significantly | Semantic features capture age-related patterns |
| R² becomes positive after BERT | Model starts learning real structure |
| DLATK gives additional gains | Linguistic style and sentiment add value |
| Random Forest is consistently best | Non-linear modeling is important |
| More features improve generalization | Performance becomes more stable |

---

## Final Summary

The results show a clear progression:

- Traditional features provide a basic baseline but are limited.
- BERT embeddings introduce strong semantic understanding and improve performance.
- DLATK features further refine predictions by adding linguistic and psychological signals.

the best performance is achieved by combining all feature types. The improvement in MAE, R², and accuracy shows that age prediction from text benefits from both semantic and stylistic information.

We could say that these results show that transcript-based features can be useful for age prediction, and advanced semantic features provide a clear improvement over traditional linguistic features. The consistent success of Random Forest also highlights that the relationship between language and age is non-linear and requires models that can capture complex patterns.

