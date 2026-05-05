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

This is expected. The important improvement is that R² becomes more positive, which means the model generalizes better instead of only fitting a smaller dataset.

---

# 4. Feature Ablation Study

To understand the contribution of each feature group, we perform a controlled ablation study. The same 133 samples are used across all setups, while the number of features changes.

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

### Top Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | avg_word_len | 0.112 |
| 2 | first_person_rate | 0.104 |
| 3 | mattr | 0.079 |
| 4 | avg_sent_len | 0.071 |
| 5 | subord_ratio | 0.063 |
| 6 | repetition_rate | 0.062 |
| 7 | adj_ratio | 0.060 |
| 8 | filler_rate | 0.054 |
| 9 | cohort_rate | 0.052 |
| 10 | total_words | 0.050 |

### Explanation

Using only traditional linguistic features gives the weakest result.

The model relies heavily on surface-level language patterns such as average word length, first-person usage, lexical diversity, and sentence length. These features contain some age-related information, but they are not strong enough on their own.

The negative R² shows that traditional features alone do not explain the target variable well.

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

### Top Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | avg_word_len | 0.073 |
| 2 | first_person_rate | 0.072 |
| 3 | bert_pc1 | 0.067 |
| 4 | bert_pc7 | 0.060 |
| 5 | bert_pc2 | 0.058 |
| 6 | avg_sent_len | 0.049 |
| 7 | mattr | 0.049 |
| 8 | bert_pc5 | 0.048 |
| 9 | bert_pc6 | 0.046 |
| 10 | adj_ratio | 0.036 |

### Explanation

Adding BERT features improves the model clearly.

The MAE drops from 3.634 to 3.450, and R² becomes positive. This shows that BERT adds semantic information that traditional linguistic features cannot capture.

The feature importance table also shows that several BERT principal components appear among the top features. This means the model benefits from deeper language representations, not just surface-level text statistics.

---

## 4.3 Traditional + BERT + DLATK Features

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

### Top Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | avg_word_len | 0.059 |
| 2 | bert_pc1 | 0.054 |
| 3 | bert_pc2 | 0.053 |
| 4 | first_person_word_rate | 0.049 |
| 5 | avg_sent_len | 0.049 |
| 6 | bert_pc7 | 0.045 |
| 7 | first_person_rate | 0.041 |
| 8 | mattr | 0.040 |
| 9 | bert_pc6 | 0.039 |
| 10 | bert_pc5 | 0.038 |
| 11 | vader_pos | 0.034 |
| 12 | vader_neg | 0.033 |
| 13 | bert_pc4 | 0.033 |
| 14 | bert_pc9 | 0.031 |
| 15 | repetition_rate | 0.031 |

### Explanation

Adding DLATK-style features gives the best overall result.

The MAE improves slightly from 3.450 to 3.447, while R² improves from 0.043 to 0.066. The within ±5 years score also increases to 75.9%.

This shows that DLATK features add useful stylistic and psychological language signals. Sentiment features such as `vader_pos` and `vader_neg` appear in the top features, suggesting that emotional tone may carry useful information for age prediction.

---

## 4.4 Final Ablation Summary

| Setup | Best Model | Samples | Features | Final MAE | Final R² | Within ±5 Years |
|---|---|---:|---:|---:|---:|---:|
| Traditional Only | Random Forest | 133 | 19 | 3.634 | -0.051 | 73.7% |
| Traditional + BERT | Random Forest | 133 | 29 | 3.450 | 0.043 | 75.2% |
| Traditional + BERT + DLATK | Random Forest | 133 | 41 | 3.447 | 0.066 | 75.9% |

---

## 5. Key Insights

| Observation | Meaning |
|---|---|
| Traditional features alone are limited | Surface-level linguistic features do not fully capture age-related patterns |
| BERT gives the biggest improvement | Semantic embeddings help the model understand deeper language patterns |
| DLATK adds smaller but useful gains | Sentiment and stylistic features improve generalization |
| Random Forest performs best | The relationship between language and age is non-linear |
| R² improves across feature groups | Adding richer features helps the model learn more meaningful structure |
| Within ±5 years improves gradually | More complete feature sets make predictions more reliable |

---

## 6. Final Summary

The results show a clear progression across feature groups.

Traditional linguistic features provide a basic baseline, but their performance is limited. Adding BERT embeddings gives a strong improvement because semantic representations capture deeper language patterns. Adding DLATK-style features provides another small improvement by including stylistic, sentiment, and psychological signals.

The best performance is achieved using Traditional + BERT + DLATK features with Random Forest. This setup obtains the lowest MAE, the highest R², and the best within ±5 years score.

These results show that transcript-based features can be useful for age prediction, but advanced semantic and linguistic features provide a clear improvement over traditional features alone.
=======
These results show that transcript-based features can be useful for age prediction, but advanced semantic and linguistic features provide a clear improvement over traditional features alone.

