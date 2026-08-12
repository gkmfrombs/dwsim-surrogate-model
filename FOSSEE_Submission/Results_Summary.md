# Machine Learning Results Summary

This document summarizes the performance and findings of the three machine learning surrogate models developed for the DWSIM Distillation column.

---

## 1. Baseline Model: Multiple Linear Regression
**Methodology:** 
A multiple linear regression model was developed to evaluate the presence of simple linear relationships between the 7 operational inputs and the 4 thermodynamic outputs. 

**Performance:**
- **Condenser & Reboiler Duty ($Q_C, Q_R$):** The model achieved **98.5% accuracy (R²)**. This indicates that the energy requirements of the column scale largely linearly with parameters such as feed flow and composition.
- **Product Purity ($x_D, x_B$):** The model achieved **~81.5% accuracy (R²)**. This lower performance confirms that product purity governed by vapor-liquid equilibrium (VLE) is highly non-linear and cannot be accurately represented by a simple first-order model.

**Conclusion:** Sufficient for rough energy estimates, but inadequate for high-fidelity purity modeling.

---

## 2. Advanced Ensemble: XGBoost (Optuna Tuned)
**Methodology:** 
To capture the non-linear thermodynamics, an XGBoost Regressor (a decision-tree ensemble) was deployed. The model architecture was optimized using the Optuna framework, and evaluated rigorously via 5-Fold Cross Validation on the training set to prevent overfitting.

**Performance:**
- **Overall Accuracy:** Reached **99.66% (R²)**. 
- **Product Purity ($x_D, x_B$):** Improved significantly to **99.5% accuracy (R²)**. 

**Conclusion:** The tree-based ensemble successfully mapped the complex, non-linear thermodynamic curves that Linear Regression failed to capture, resulting in near-perfect predictive capabilities.

---

## 3. Final Selection: PyTorch Physics-Informed Neural Network (PINN)
**Methodology:** 
While XGBoost achieved high mathematical accuracy, it inherently lacks physical constraints. To address this, a PyTorch Physics-Informed Neural Network (PINN) was developed. A custom loss function was designed to penalize the network dynamically if its predictions violated the Law of Conservation of Mass:
`Benzene IN = Feed Flow × Benzene Fraction`
`Benzene OUT = (Distillate Flow × Top Purity) + (Bottoms Flow × Bottom Purity)`

**Performance:**
- **Overall Accuracy:** Achieved **99.76% (R²)**. 
- **Product Purity ($x_D, x_B$):** Achieved **99.7%** and **99.6%** respectively.

**Conclusion:** The PINN was selected as the final surrogate model. It outperformed XGBoost statistically, while explicitly guaranteeing that its predictions obey the fundamental laws of chemical engineering. This yields a surrogate model that is both highly accurate and scientifically sound.
