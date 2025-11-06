# 📊 Mathematics for Data Science & Machine Learning

> A comprehensive reference guide covering all essential mathematical formulas used in Data Science and Machine Learning

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat&logo=github)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📑 Table of Contents

- [Linear Algebra](#1-linear-algebra)
- [Calculus](#2-calculus)
- [Probability](#3-probability)
- [Statistics](#4-statistics)
- [Linear & Logistic Regression](#5-linear--logistic-regression)
- [Neural Networks](#6-neural-networks)
- [Optimization Algorithms](#7-optimization-algorithms)
- [Evaluation Metrics](#8-evaluation-metrics)
- [Clustering](#9-clustering)
- [Deep Learning](#10-deep-learning)
- [Dimensionality Reduction](#11-dimensionality-reduction)
- [Information Theory](#12-information-theory)
- [Support Vector Machines](#13-support-vector-machines)
- [Decision Trees & Ensembles](#14-decision-trees--ensemble-methods)
- [Bias-Variance Tradeoff](#15-bias-variance-tradeoff)

---

## 1. Linear Algebra

### 🔢 Vectors and Matrices

| Formula | Expression |
|---------|------------|
| **Dot Product** | `a · b = Σ(aᵢbᵢ) = a₁b₁ + a₂b₂ + ... + aₙbₙ` |
| **Matrix Multiplication** | `C = AB where Cᵢⱼ = Σₖ(AᵢₖBₖⱼ)` |
| **Transpose** | `(AB)ᵀ = BᵀAᵀ` |
| **Identity Matrix** | `AI = IA = A` |
| **Inverse Matrix** | `AA⁻¹ = A⁻¹A = I` |

### 📏 Norms

| Norm Type | Formula |
|-----------|---------|
| **L1 Norm (Manhattan)** | `‖x‖₁ = Σ\|xᵢ\|` |
| **L2 Norm (Euclidean)** | `‖x‖₂ = √(Σxᵢ²)` |
| **Frobenius Norm** | `‖A‖F = √(ΣᵢΣⱼ aᵢⱼ²)` |

### 🎯 Eigenvalues and Eigenvectors

```
Eigenvalue Equation:     Av = λv
Characteristic Equation: det(A - λI) = 0
Trace:                   tr(A) = Σλᵢ = Σaᵢᵢ
Determinant:             det(A) = Πλᵢ
```

---

## 2. Calculus

### 📐 Basic Derivatives

| Rule | Formula |
|------|---------|
| **Power Rule** | `d/dx(xⁿ) = nxⁿ⁻¹` |
| **Chain Rule** | `d/dx[f(g(x))] = f'(g(x)) · g'(x)` |
| **Product Rule** | `d/dx[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)` |
| **Quotient Rule** | `d/dx[f(x)/g(x)] = [f'(x)g(x) - f(x)g'(x)]/g(x)²` |

### ∇ Partial Derivatives

```
Gradient:        ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
Hessian Matrix:  Hᵢⱼ = ∂²f/(∂xᵢ∂xⱼ)
Jacobian Matrix: Jᵢⱼ = ∂fᵢ/∂xⱼ
```

### 🧮 Common Activation Function Derivatives

| Function | Derivative |
|----------|------------|
| **Sigmoid** | `σ'(x) = σ(x)(1 - σ(x))` |
| **Tanh** | `tanh'(x) = 1 - tanh²(x)` |
| **ReLU** | `ReLU'(x) = {1 if x > 0, 0 otherwise}` |
| **Exponential** | `d/dx(eˣ) = eˣ` |
| **Logarithm** | `d/dx(ln x) = 1/x` |

---

## 3. Probability

### 🎲 Basic Probability Rules

```
Probability:              P(A) = (Favorable outcomes)/(Total outcomes)
Complement Rule:          P(Aᶜ) = 1 - P(A)
Addition Rule:            P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
Multiplication Rule:      P(A ∩ B) = P(A|B)P(B) = P(B|A)P(A)
Conditional Probability:  P(A|B) = P(A ∩ B)/P(B)
```

### 🔮 Bayes' Theorem

```
Bayes' Rule:      P(A|B) = [P(B|A)P(A)]/P(B)

Extended Form:    P(A|B) = [P(B|A)P(A)]/[P(B|A)P(A) + P(B|Aᶜ)P(Aᶜ)]
```

> **💡 Key Application:** Fundamental in ML for classification, spam detection, and probabilistic reasoning

### 📊 Expected Value & Variance

| Concept | Formula |
|---------|---------|
| **Expected Value** | `E[X] = Σ xᵢP(xᵢ)` or `∫ xf(x)dx` |
| **Variance** | `Var(X) = E[(X - μ)²] = E[X²] - (E[X])²` |
| **Standard Deviation** | `σ = √Var(X)` |
| **Covariance** | `Cov(X,Y) = E[(X - μₓ)(Y - μᵧ)]` |
| **Correlation** | `ρ(X,Y) = Cov(X,Y)/(σₓσᵧ)` |

---

## 4. Statistics

### 📈 Descriptive Statistics

| Measure | Formula |
|---------|---------|
| **Mean** | `μ = (1/n)Σxᵢ` |
| **Sample Variance** | `s² = [1/(n-1)]Σ(xᵢ - x̄)²` |
| **Population Variance** | `σ² = (1/n)Σ(xᵢ - μ)²` |
| **Standard Error** | `SE = σ/√n` |

### 🎯 Hypothesis Testing

```
Z-score:             z = (x - μ)/σ
T-statistic:         t = (x̄ - μ)/(s/√n)
Confidence Interval: CI = x̄ ± z(α/2) · SE
```

**Key Terms:**
- **Type I Error (α):** Rejecting true null hypothesis
- **Type II Error (β):** Failing to reject false null hypothesis
- **Power:** 1 - β

---

## 5. Linear & Logistic Regression

### 📉 Linear Regression

| Component | Formula |
|-----------|---------|
| **Simple Model** | `y = β₀ + β₁x + ε` |
| **Matrix Form** | `y = Xβ + ε` |
| **Normal Equation** | `β = (XᵀX)⁻¹Xᵀy` |
| **Predicted Values** | `ŷ = Xβ` |

### 📊 Loss Functions

```
Mean Squared Error (MSE):     MSE = (1/n)Σ(yᵢ - ŷᵢ)²
Root Mean Squared Error:      RMSE = √MSE
Mean Absolute Error (MAE):    MAE = (1/n)Σ|yᵢ - ŷᵢ|
R-squared:                    R² = 1 - [Σ(yᵢ - ŷᵢ)²]/[Σ(yᵢ - ȳ)²]
Adjusted R²:                  R²adj = 1 - [(1 - R²)(n - 1)]/(n - p - 1)
```

### 🔄 Logistic Regression

| Component | Formula |
|-----------|---------|
| **Sigmoid Function** | `σ(z) = 1/(1 + e⁻ᶻ)` |
| **Logit** | `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ` |
| **Probability** | `P(y=1\|x) = σ(wᵀx + b)` |
| **Odds** | `Odds = P(y=1)/P(y=0) = eᶻ` |
| **Log-Odds** | `log(Odds) = z` |

```
Log Loss (Binary Cross-Entropy):
L = -(1/n)Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

### 🛡️ Regularization

| Type | Cost Function |
|------|---------------|
| **Ridge (L2)** | `J(β) = Σ(yᵢ - ŷᵢ)² + λΣβⱼ²` |
| **Lasso (L1)** | `J(β) = Σ(yᵢ - ŷᵢ)² + λΣ\|βⱼ\|` |
| **Elastic Net** | `J(β) = Σ(yᵢ - ŷᵢ)² + λ₁Σ\|βⱼ\| + λ₂Σβⱼ²` |

---

## 6. Neural Networks

### ⚡ Activation Functions

| Function | Formula |
|----------|---------|
| **Sigmoid** | `σ(x) = 1/(1 + e⁻ˣ)` |
| **Tanh** | `tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)` |
| **ReLU** | `ReLU(x) = max(0, x)` |
| **Leaky ReLU** | `f(x) = {x if x > 0, αx otherwise}` |
| **Softmax** | `softmax(xᵢ) = eˣⁱ/Σⱼeˣʲ` |

### 🔄 Forward Propagation

```
Linear Combination:  z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
Activation:          a⁽ˡ⁾ = g(z⁽ˡ⁾)
```

### ⬅️ Backpropagation

```
Output Layer Error:   δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ g'(z⁽ᴸ⁾)
Hidden Layer Error:   δ⁽ˡ⁾ = [(W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾] ⊙ g'(z⁽ˡ⁾)
Weight Gradient:      ∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾(a⁽ˡ⁻¹⁾)ᵀ
Bias Gradient:        ∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

---

## 7. Optimization Algorithms

### ⬇️ Gradient Descent

| Variant | Update Rule |
|---------|-------------|
| **Batch GD** | `θ := θ - α∇J(θ)` |
| **Stochastic GD** | `θ := θ - α∇J(θ; xⁱ, yⁱ)` |
| **Mini-batch GD** | Uses small batches of data |

### 🚀 Advanced Optimizers

```
Momentum:
  v := βv + (1-β)∇J(θ)
  θ := θ - αv

RMSprop:
  s := βs + (1-β)(∇J)²
  θ := θ - α∇J/√(s + ε)

Adam:
  m := β₁m + (1-β₁)∇J
  v := β₂v + (1-β₂)(∇J)²
  θ := θ - α·m̂/√(v̂ + ε)
```

---

## 8. Evaluation Metrics

### ✅ Classification Metrics

| Metric | Formula |
|--------|---------|
| **Accuracy** | `(TP + TN)/(TP + TN + FP + FN)` |
| **Precision** | `TP/(TP + FP)` |
| **Recall (Sensitivity)** | `TP/(TP + FN)` |
| **Specificity** | `TN/(TN + FP)` |
| **F1-Score** | `2·(Precision·Recall)/(Precision + Recall)` |
| **F-beta Score** | `(1 + β²)·(Precision·Recall)/(β²·Precision + Recall)` |

### 📉 Confusion Matrix

```
                    Predicted
                 Positive  Negative
Actual Positive     TP        FN
       Negative     FP        TN
```

**Legend:**
- **TP** = True Positive
- **TN** = True Negative
- **FP** = False Positive (Type I Error)
- **FN** = False Negative (Type II Error)

### 📈 ROC Curve

```
TPR (True Positive Rate):  TP/(TP + FN)
FPR (False Positive Rate): FP/(FP + TN)
AUC (Area Under Curve):    ∫₀¹ TPR(FPR⁻¹(x))dx
```

---

## 9. Clustering

### 🎯 K-Means

```
Objective Function:  minimize Σᵏⱼ₌₁ Σₓ∈Cⱼ ||x - μⱼ||²
Centroid Update:     μⱼ = (1/|Cⱼ|)Σₓ∈Cⱼ x
```

### 📏 Distance Metrics

| Metric | Formula |
|--------|---------|
| **Euclidean** | `d(x, y) = √(Σᵢ(xᵢ - yᵢ)²)` |
| **Manhattan** | `d(x, y) = Σᵢ\|xᵢ - yᵢ\|` |
| **Cosine Similarity** | `cos(θ) = (x·y)/(‖x‖·‖y‖)` |

### 📊 Silhouette Score

```
s(i) = [b(i) - a(i)]/max{a(i), b(i)}

Where:
  a(i) = mean distance to points in same cluster
  b(i) = mean distance to points in nearest cluster
```

---

## 10. Deep Learning

### 🔄 Batch Normalization

```
Normalize:        x̂ = (x - μ_B)/√(σ²_B + ε)
Scale and Shift:  y = γx̂ + β
```

### 🧠 Convolutional Neural Networks (CNN)

```
Convolution Operation:  (f * g)(t) = Σₓ f(x)g(t - x)
Output Size:            O = [(W - K + 2P)/S] + 1

Where:
  W = input size
  K = kernel size
  P = padding
  S = stride
```

### 🔁 Recurrent Neural Networks (RNN)

```
Hidden State:  hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + bₕ)
Output:        yₜ = Wₕᵧhₜ + bᵧ
```

### 🧬 Long Short-Term Memory (LSTM)

```
Forget Gate:  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
Input Gate:   iₜ = σ(Wᵢ·[hₜ₋₁, xₜ] + bᵢ)
Output Gate:  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
Cell State:   Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```

### 💧 Dropout

```
Training:  Output = mask ⊙ activation / (1 - p)
Testing:   Use all neurons (no dropout)
```

---

## 11. Dimensionality Reduction

### 📊 Principal Component Analysis (PCA)

```
Covariance Matrix:          Σ = (1/n)XᵀX
Principal Components:       Eigenvectors of Σ
Explained Variance Ratio:   λᵢ/Σⱼλⱼ
Projection:                 Z = XW (W = eigenvectors)
```

### 🔢 Singular Value Decomposition (SVD)

```
Decomposition:   X = UΣVᵀ
Reduced Form:    X ≈ UₖΣₖVₖᵀ
```

---

## 12. Information Theory

### 📡 Entropy and Information

| Measure | Formula |
|---------|---------|
| **Entropy** | `H(X) = -Σ P(xᵢ)log₂P(xᵢ)` |
| **Cross-Entropy** | `H(p,q) = -Σ p(x)log q(x)` |
| **KL Divergence** | `DKL(P‖Q) = Σ P(x)log[P(x)/Q(x)]` |
| **Mutual Information** | `I(X;Y) = H(X) - H(X\|Y) = H(Y) - H(Y\|X)` |
| **Conditional Entropy** | `H(Y\|X) = -ΣₓΣᵧ P(x,y)log P(y\|x)` |

---

## 13. Support Vector Machines

### 🎯 Linear SVM

```
Decision Function:  f(x) = wᵀx + b
Margin:             2/||w||
Optimization:       minimize ½||w||²
                    subject to yᵢ(wᵀxᵢ + b) ≥ 1
```

### 🛡️ Soft Margin SVM

```
Objective:   minimize ½||w||² + CΣξᵢ
Constraint:  yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

### 🔧 Kernel Functions

| Kernel | Formula |
|--------|---------|
| **Linear** | `K(x, x') = xᵀx'` |
| **Polynomial** | `K(x, x') = (xᵀx' + c)ᵈ` |
| **RBF (Gaussian)** | `K(x, x') = exp(-γ‖x - x'‖²)` |
| **Sigmoid** | `K(x, x') = tanh(αxᵀx' + c)` |

---

## 14. Decision Trees & Ensemble Methods

### 🌳 Impurity Measures

| Measure | Formula |
|---------|---------|
| **Gini Impurity** | `Gini = 1 - Σᵢ pᵢ²` |
| **Entropy** | `H = -Σᵢ pᵢlog₂(pᵢ)` |
| **Classification Error** | `E = 1 - max(pᵢ)` |

```
Information Gain:  IG(D, A) = H(D) - Σᵥ[(|Dᵥ|/|D|)·H(Dᵥ)]
```

### 🌲 Ensemble Methods

```
Bagging (Random Forest):
  Prediction: ŷ = (1/B)Σᵇ fᵇ(x)

AdaBoost:
  Sample Weight:  wᵢ⁽ᵗ⁺¹⁾ = wᵢ⁽ᵗ⁾·exp[αₜ·I(yᵢ ≠ hₜ(xᵢ))]
  Model Weight:   αₜ = ½ln[(1 - εₜ)/εₜ]

Gradient Boosting:
  Update:   Fₘ(x) = Fₘ₋₁(x) + γₘhₘ(x)
  Residual: rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]
```

---

## 15. Bias-Variance Tradeoff

### ⚖️ Error Decomposition

```
Total Error = Bias² + Variance + Irreducible Error

Bias:     Bias = E[ŷ] - y
Variance: Variance = E[(ŷ - E[ŷ])²]
```

**Key Insights:**
- **High Bias** → Underfitting (model too simple)
- **High Variance** → Overfitting (model too complex)
- **Goal:** Find the optimal balance between bias and variance

---

## 🎯 Quick Reference: Loss Functions

### Regression Losses

| Loss | Formula | Use Case |
|------|---------|----------|
| **MSE** | `L = (1/n)Σ(yᵢ - ŷᵢ)²` | Standard regression |
| **MAE** | `L = (1/n)Σ\|yᵢ - ŷᵢ\|` | Robust to outliers |
| **Huber** | `L = {½(y - ŷ)² if \|y - ŷ\| ≤ δ, δ\|y - ŷ\| - ½δ² otherwise}` | Combines MSE & MAE |

### Classification Losses

| Loss | Formula | Use Case |
|------|---------|----------|
| **Binary Cross-Entropy** | `L = -[ylog(ŷ) + (1-y)log(1-ŷ)]` | Binary classification |
| **Categorical Cross-Entropy** | `L = -Σᵢ yᵢlog(ŷᵢ)` | Multi-class classification |
| **Hinge Loss** | `L = max(0, 1 - y·ŷ)` | SVM classification |

---

## 🎓 Feature Engineering

### Normalization Techniques

| Method | Formula | Range |
|--------|---------|-------|
| **Min-Max Scaling** | `x' = (x - min)/(max - min)` | [0, 1] |
| **Z-Score Normalization** | `x' = (x - μ)/σ` | ~ [-3, 3] |
| **Max Abs Scaling** | `x' = x/\|max\|` | [-1, 1] |

### Polynomial Features

```
Degree 2: [x₁, x₂] → [1, x₁, x₂, x₁², x₁x₂, x₂²]
Degree 3: [x₁, x₂] → [1, x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]
```

---

## 🌟 Contributing

Contributions are welcome! If you find any errors or want to add more formulas:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/new-formulas`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new formulas'`)
5. Push to the branch (`git push origin feature/new-formulas`)
6. Create a Pull Request

---

## 📚 Resources

- [Mathematics for Machine Learning Book](https://mml-book.github.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💡 Tips for Using This Guide

1. **Bookmark** this page for quick reference during ML projects
2. **Print** specific sections you use frequently
3. **Practice** implementing these formulas in code
4. **Understand** the intuition behind each formula, not just memorization
5. **Share** with fellow ML practitioners and students

---

## 🤝 Acknowledgments

Created with ❤️ for the Machine Learning community. Special thanks to all contributors and the open-source ML community.

---

## ⭐ Star This Repo!

If you find this helpful, please consider giving it a star! It helps others discover this resource.

---

## 👥 Contributors

<div align="center">

### Created and Maintained by

<a href="https://github.com/iNSRawat">
  <img src="https://github.com/iNSRawat.png" width="100" height="100" style="border-radius: 50%;" alt="iNSRawat"/>
</a>

**[@iNSRawat](https://github.com/iNSRawat)**

*Creator & Maintainer*

---

### 🤝 Want to Contribute?

We welcome contributions! If you have:
- Formula corrections or additions
- Documentation improvements
- Feature suggestions
- Bug reports

Please feel free to:
1. Fork the repository
2. Make your changes
3. Submit a pull request

---

### 📬 Connect

- 💼 **LinkedIn**: [Connect with me](https://linkedin.com/in/insrawat)
- 🐦 **Twitter**: [@iNSRawat](https://twitter.com/insrawat)
- 📧 **GitHub**: [Open an issue](https://github.com/iNSRawat/Mathematics-Formulas-for-Data-Science-Machine-Learning/issues)

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

Made with ❤️ by [@iNSRawat](https://github.com/iNSRawat)

</div>

</div>

---

**Happy Learning! 🚀📊🤖**
