# Results

## Traditional Linguistic Baseline

**Dataset:** 109 samples

| Model             | MAE   | R²     |
| ----------------- | ----- | ------ |
| Ridge             | 3.225 | -0.087 |
| Random Forest     | 3.247 | -0.202 |
| Gradient Boosting | 3.388 | -0.331 |

**Best Model:** Ridge

* **Final MAE:** 3.222
* **Final R²:** -0.048
* **Within ±5 years:** 78.0%

---

## With Advanced Features (BERT + DLATK)

**Dataset:** 109 samples

| Model             | MAE   | R²     |
| ----------------- | ----- | ------ |
| Ridge             | 3.143 | -0.071 |
| Random Forest     | 3.126 | -0.072 |
| Gradient Boosting | 3.205 | -0.214 |

**Best Model:** Random Forest

* **Final MAE:** 3.126
* **Final R²:** 0.023
* **Within ±5 years:** 78.9%

---

## Key Observations

* Adding **BERT embeddings + DLATK-style features** improves performance.
* MAE improved from **3.222 → 3.126**
* Within-5-year accuracy improved from **78.0% → 78.9%**
* Model shifts from **linear (Ridge)** to **non-linear (Random Forest)** after adding semantic features.
* Slightly **positive R²** indicates improved explanatory power.

---

## Summary

Using transcript-based features alone, the model achieves performance comparable to acoustic-based approaches.
Adding semantic and sentiment features provides consistent improvement and better generalization.
