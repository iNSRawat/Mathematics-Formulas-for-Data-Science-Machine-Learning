# Mathematics-Formulas-for-Data-Science-Machine-Learning
A comprehensive collection of mathematics formulas essential for data science and machine learning

## Table of Contents
1. [Linear Algebra](#linear-algebra)
2. [Calculus](#calculus)
3. [Probability & Statistics](#probability--statistics)
4. [Optimization](#optimization)
5. [Machine Learning Algorithms](#machine-learning-algorithms)

## Linear Algebra

### Vector Operations
- **Dot Product**: $\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i$
- **Vector Norm (L2)**: $||\vec{x}||_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$
- **Vector Norm (L1)**: $||\vec{x}||_1 = \sum_{i=1}^{n} |x_i|$

### Matrix Operations
- **Matrix Multiplication**: $(AB)_{ij} = \sum_{k=1}^{n} A_{ik}B_{kj}$
- **Matrix Transpose**: $(A^T)_{ij} = A_{ji}$
- **Matrix Inverse**: $AA^{-1} = A^{-1}A = I$
- **Determinant (2x2)**: $det(A) = ad - bc$ for $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$

### Eigenvalues and Eigenvectors
- **Eigenvalue Equation**: $A\vec{v} = \lambda\vec{v}$
- **Characteristic Equation**: $det(A - \lambda I) = 0$

## Calculus

### Derivatives
- **Power Rule**: $\frac{d}{dx}x^n = nx^{n-1}$
- **Chain Rule**: $\frac{d}{dx}f(g(x)) = f'(g(x))g'(x)$
- **Product Rule**: $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$

### Partial Derivatives
- **Gradient**: $\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right]$
- **Hessian Matrix**: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

### Integration
- **Definite Integral**: $\int_{a}^{b} f(x)dx$
- **Integration by Parts**: $\int u dv = uv - \int v du$

## Probability & Statistics

### Probability Basics
- **Probability**: $P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}$
- **Conditional Probability**: $P(A|B) = \frac{P(A \cap B)}{P(B)}$
- **Bayes' Theorem**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

### Distributions
- **Normal Distribution**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- **Bernoulli Distribution**: $P(X=x) = p^x(1-p)^{1-x}$
- **Binomial Distribution**: $P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}$

### Statistical Measures
- **Mean**: $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$
- **Variance**: $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$
- **Standard Deviation**: $\sigma = \sqrt{\sigma^2}$
- **Covariance**: $Cov(X,Y) = E[(X-E[X])(Y-E[Y])]$
- **Correlation**: $\rho_{X,Y} = \frac{Cov(X,Y)}{\sigma_X\sigma_Y}$

## Optimization

### Gradient Descent
- **Update Rule**: $\theta_{new} = \theta_{old} - \alpha\nabla J(\theta)$
- **Learning Rate**: $\alpha$ (controls step size)

### Stochastic Gradient Descent
- **Update Rule**: $\theta = \theta - \alpha\nabla J(\theta; x^{(i)}, y^{(i)})$

### Loss Functions
- **Mean Squared Error (MSE)**: $MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$
- **Cross-Entropy Loss**: $L = -\sum_{i=1}^{n}y_i\log(\hat{y}_i)$

## Machine Learning Algorithms

### Linear Regression
- **Hypothesis**: $h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n = \theta^Tx$
- **Cost Function**: $J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2$
- **Normal Equation**: $\theta = (X^TX)^{-1}X^Ty$

### Logistic Regression
- **Sigmoid Function**: $\sigma(z) = \frac{1}{1+e^{-z}}$
- **Hypothesis**: $h_\theta(x) = \sigma(\theta^Tx)$
- **Cost Function**: $J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$

### Neural Networks
- **Activation Functions**:
  - **ReLU**: $f(x) = max(0, x)$
  - **Tanh**: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
  - **Softmax**: $\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}$

### Support Vector Machines
- **Decision Function**: $f(x) = sign(w^Tx + b)$
- **Objective**: Maximize margin $\frac{2}{||w||}$

### K-Means Clustering
- **Distance Metric**: $d(x, \mu_k) = ||x - \mu_k||^2$
- **Centroid Update**: $\mu_k = \frac{1}{|C_k|}\sum_{x \in C_k}x$

### Principal Component Analysis (PCA)
- **Covariance Matrix**: $C = \frac{1}{n}X^TX$
- **Principal Components**: Eigenvectors of C

### Regularization
- **L1 Regularization (Lasso)**: $J(\theta) = MSE + \lambda\sum_{i=1}^{n}|\theta_i|$
- **L2 Regularization (Ridge)**: $J(\theta) = MSE + \lambda\sum_{i=1}^{n}\theta_i^2$

### Evaluation Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
- **R² Score**: $R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$

---

## Contributing
Feel free to contribute by adding more formulas or improving existing ones!

## License
MIT License
