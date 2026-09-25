# Machine Learning Made Simple — Doc 8: Polynomial Regression & The 5 Assumptions of Linear Regression
**Worksheet:** [Worksheet 08](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_08_Polynomial_Regression_and_Assumptions.pdf)  
**Lab Assignment:** [Lab 08 Polynomial Regression](file:///Users/mayanksharma/Downloads/AML/04_Notebooks/Lab_08_Polynomial_Regression/solved/Poly_using_gradient_final.ipynb)  
**Topics:** Failure of a Straight Line on Curved Data · Feature Transformation ($\phi(x) = (x, x^2)$) · Why Polynomial Regression is Still a "Linear" Model · The 3D Mental Model (A Curve Becomes a Plane) · Higher-Degree Polynomials & Degree-vs-Flexibility · The Overfitting Danger · Why Assumptions Matter (Inference vs. Prediction) · Assumption 1: Linearity · Assumption 2: Normality of Residuals · Assumption 3: Homoscedasticity vs. Heteroscedasticity · Assumption 4: Independence & Autocorrelation · Assumption 5: Multicollinearity · Proof of Singularity in $(\mathbf{X}^T \mathbf{X})$ · Comprehensive Diagnostic & Fix Guide

---

## 🗺️ Where Are We in the Journey?

In our previous documents:
* [Doc 3](file:///Users/mayanksharma/Downloads/AML/docs/03_Simple_Linear_Regression_OLS.md) & [Doc 4](file:///Users/mayanksharma/Downloads/AML/docs/04_Multiple_Linear_Regression_OLS.md): Fitting straight lines and hyperplanes using Ordinary Least Squares (OLS).
* [Doc 5](file:///Users/mayanksharma/Downloads/AML/docs/05_Batch_Gradient_Descent_MLR.md) & [Doc 6](file:///Users/mayanksharma/Downloads/AML/docs/06_Stochastic_and_MiniBatch_Gradient_Descent.md): Iterative optimization (Batch, Stochastic, and Mini-Batch Gradient Descent).
* [Doc 7](file:///Users/mayanksharma/Downloads/AML/docs/07_Regression_and_Classification_Evaluation_Metrics.md): Judging model performance using MAE, RMSE, $R^2$, Adjusted $R^2$, and Confusion Matrices.

Up to this point, we have assumed that a straight line or flat plane can model the data.  
**But what if the real world is curved?**
* In biology, bacterial growth curves accelerate exponentially.
* In business, marketing returns bend upward as word-of-mouth kicks in, then taper off.
* In physics, gravitational trajectories follow parabolas ($d = \frac{1}{2}gt^2$).

In **Doc 8**, we explore two foundational concepts:
1. **Part I (Polynomial Regression):** How to fit beautiful curves without changing our linear algorithms at all!
2. **Part II (The 5 Assumptions of Linear Regression):** The statistical health check ensuring our linear model is trustworthy and valid.

---

# PART I: POLYNOMIAL REGRESSION

## Section 1: The Failure of a Straight Line

### 1. The Real-World Hook
> **The Story:** Suppose we want to predict the internship stipend ($y$, in ₹k) of an NST student based on a single feature: their **Project Portfolio Depth Score ($x$)** rated from $1$ (basic calculator app) to $10$ (production-ready distributed system).

Does each extra point of project depth increase stipend by a constant amount?
* Moving from score $2 \to 4$ might only increase stipend from ₹24k to ₹42k (+$18\text{k}$).
* But moving from score $8 \to 10$ unlocks top-tier tier-1 companies and global referrals, shooting stipend from ₹91k to ₹128k (+$37\text{k}$)!

```
   Stipend (y)                           Stipend (y)
      ^                                     ^
      |         /                           |                  • (10, 128)
      |        /                            |             • (8, 91)
      |       /                             |        • (6, 62)
      |      /                              |    • (4, 42)
      |     /                               | • (2, 24)
      +-----------> Score (x)               +------------------------> Score (x)
     (a) Linear Pattern                    (b) Upward Curving Pattern
         Constant slope                        Slope increases as x grows!
```

---

### 2. Why Ordinary Linear Regression Fails
A simple linear model is:

$$y = \beta_0 + \beta_1 x$$

* It has **only one slope ($\beta_1$)**.
* The slope is completely **constant**. If $x$ increases by 1 unit, $y$ increases by exactly $\beta_1$, regardless of whether $x = 2$ or $x = 9$.
* Trying to force a rigid straight line through an upward-curving scatter plot produces systematic under-prediction at the ends and over-prediction in the middle!

We need a model whose **slope can change as $x$ changes**.

---

## Section 2: The Core Trick — Creating a New Feature ($x^2$)

Do not think of Polynomial Regression as a new machine learning algorithm.  
**It is simply a Feature Transformation!**

We take our original input column $x$ and mathematically generate a brand new column: **$x^2$**!

| Student | Raw Feature $x$ | Transformed Feature $x^2$ | Actual Stipend $y$ (in ₹k) |
| :---: | :---: | :---: | :---: |
| **A** | 2 | $2^2 = \mathbf{4}$ | 24 |
| **B** | 4 | $4^2 = \mathbf{16}$ | 42 |
| **C** | 6 | $6^2 = \mathbf{36}$ | 62 |
| **D** | 8 | $8^2 = \mathbf{64}$ | 91 |
| **E** | 10 | $10^2 = \mathbf{100}$ | 128 |

Now feed both columns into our standard Multiple Linear Regression engine:

$$\mathbf{y = \beta_0 + \beta_1 x + \beta_2 x^2}$$

* When $x$ is small ($x = 2$), $x^2 = 4$ (contributes very little).
* When $x$ is large ($x = 10$), $x^2 = 100$ (grows rapidly, bending the line aggressively upward!).

```
   Original 1D Space:                    2D Transformed Feature Space:
   
        [ x ]                                     [ x ,  x^2 ]
      (CGPA Score)                           (CGPA Score, Squared Score)
          |                                               |
          +-------> Feature Map: ϕ(x) = (x, x^2) -------->+
```

---

## Section 3: The 3D Mental Model — A Curve Becomes a Plane!

> 🤯 **Mind-Bending Insight:**  
> Why is Polynomial Regression called "Linear" Regression if it draws curves?

Let's rename the new feature $x^2$ as a new variable **$z$**:

$$z = x^2$$

Substituting $z$ into our model gives:

$$y = \beta_0 + \beta_1 x + \beta_2 z$$

Look at that equation! That is the standard equation of a **perfectly flat 2D plane living in 3D space $(x, z, y)$**!

```
               y (Target: Stipend)
               ^
               |       /----------------/  (Flat Plane in 3D: y = β0 + β1*x + β2*z)
               |      /                /
               |     /----------------/
               |    /   / (The parabolic slice z = x^2)
               +-----------------------------> x (Original Score)
              /
             v  z = x^2 (Squared Score)
```

1. In 3D feature space $(x, z, y)$, the model is fitting a **completely flat, linear plane**.
2. However, data points are physically restricted to lie only along the parabolic track where $z = x^2$.
3. When you slice this flat 3D plane along the track $z = x^2$ and view it from the 2D front $(x, y)$, **it looks like a smooth curve**!

> 📌 **Takeaway:**  
> Polynomial Regression is **linear in its parameters ($\beta_0, \beta_1, \beta_2$)**, even though it is **non-linear in the original input feature ($x$)**.  
> You can train it using the exact same OLS formula $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$ or Gradient Descent without changing a single line of optimizer code!

---

## Section 4: Generalizing to Higher Degrees & Overfitting

The **Polynomial Degree ($d$)** is the highest power of $x$ included in the feature expansion:
* **Degree 1 (Linear):** $y = \beta_0 + \beta_1 x$ $\longrightarrow$ Straight line (0 bends).
* **Degree 2 (Quadratic):** $y = \beta_0 + \beta_1 x + \beta_2 x^2$ $\longrightarrow$ Parabola (1 bend, U-shape or inverted U).
* **Degree 3 (Cubic):** $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3$ $\longrightarrow$ S-curve (up to 2 bends).
* **Degree $d$:** $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d$ $\longrightarrow$ Up to $d - 1$ bends ($d + 1$ total columns in $\mathbf{X}$).

---

### The Overfitting Danger (Degree 1 vs. Degree 2 vs. Degree 8)

```
      Degree 1: Underfitting            Degree 2: Optimal Fit           Degree 8: Overfitting
      
      y                                 y                                 y
      ^      •                          ^      •                          ^   /\ •
      |     /   •                       |     /   •                       |  /  \ / \ •
      |    / •                          |    ( •                          | •    V   •
      |   /     •                       |   /     •                       |           •
      +------------------> x            +------------------> x            +------------------> x
     High Bias (Too simple)            Sweet Spot (Captures signal)      High Variance (Fits noise)
```

* **Degree 1 (Underfitting):** The model is too rigid. High bias, cannot capture the curve.
* **Degree 2 (Good Fit):** Smoothly follows the underlying physical trend. Low training error, low test error.
* **Degree 8 (Severe Overfitting):** The model has so many parameters that it twists like a snake, passing through every single training point. Its training error drops to zero, but when tested on new students, **its predictions explode catastrophically**!

> ⚠️ **Golden Rule:**  
> A model with lower training error is **not** automatically better! Never choose polynomial degree by looking at training error. **Always evaluate candidate degrees using cross-validation or a validation set.**

---

# PART II: THE 5 ASSUMPTIONS OF LINEAR REGRESSION

## Why Assumptions Matter: Inference vs. Prediction

Linear regression is not just a black-box prediction tool; in science, medicine, and economics, it is used for **Inference** (understanding relationships):
* *"Does increasing advertising budget by ₹10,000 cause a statistically significant increase in sales?"*
* *"What is the 95% confidence interval for the effect of blood pressure on heart attack risk?"*

For statistical inferences, p-values, and hypothesis tests to be valid, **5 fundamental assumptions must hold**.

---

### Overview of the 5 Assumptions:

```
                       +---------------------------------------+
                       |   The 5 Linear Regression Assumptions |
                       +-------------------+-------------------+
                                           |
         +-----------------+---------------+-----------------+-----------------+
         |                 |                                 |                 |
         v                 v                                 v                 v
   [ 1. Linearity ] [ 2. Normality ]               [ 3. Homoscedasticity ] [ 4. Independence ]
    Relationship      Residuals follow              Constant error          No autocorrelation
    is linear.        Bell Curve N(0, σ^2).         spread across y_hat.    between residuals.
                                           |
                                           v
                             [ 5. No Multicollinearity ]
                              Features are not redundant
                              linear duplicates of each other.
```

---

## Section 5: Assumption 1 — Linearity

### The Assumption:
The relationship between the independent features $\mathbf{X}$ and the mean of the dependent target $y$ is **linear in the parameters**.  
*(Changes in features lead to proportional, additive changes in the target).*

* **What it means:** It does *not* mean every point must fall on a line. It means the **central trend** must be straight, not curved or circular.
* **How to Check:**
  1. **Scatter plots:** Plot $y$ against each individual feature $x$.
  2. **Residual Plot ($e$ vs. $\hat{y}$):** Plot residuals on vertical axis against predicted values on horizontal axis.  
     * ✅ **Good:** Residuals are randomly scattered like a swarm of bees around the zero line with no pattern.  
     * ❌ **Violated:** Residuals form a distinct **U-shape or inverted U-curve**.
* **Remedies if Violated:**
  * Add **Polynomial Features** ($x^2, x^3$).
  * Apply **Nonlinear Transformations** (e.g., $\log(x)$, $\sqrt{x}$, or $1/x$).

---

## Section 6: Assumption 2 — Normality of Residuals

### The Assumption:
The error terms (residuals) are assumed to be **normally distributed with mean zero**:

$$e_i \sim \mathcal{N}(0, \sigma^2)$$

```
          Residual Frequency
                 ^
                 |           /\
                 |          /  \      Bell-Shaped Curve
                 |         /    \     Centered at Zero (Mean = 0)
                 |       _/      \_
                 +---------+----------> Residual (e)
                         -3σ 0 +3σ
```

* **What it means:** Most residuals are small and cluster near zero; very large positive or negative errors are rare; errors are roughly symmetric.
* **Why it matters:** Required for calculating reliable p-values, t-statistics, and confidence intervals.
* **How to Check:**
  1. **Histogram of Residuals:** Look for a symmetric, bell-shaped distribution.
  2. **Q-Q Plot (Quantile-Quantile Plot):** Data points should hug the 45-degree diagonal reference line.
* **Remedies if Violated:**
  * Transform the target variable using **Log Transform ($\log(y)$)** or **Box-Cox transformation**.
  * Remove severe, unrepresentative data entry outliers.

---

## Section 7: Assumption 3 — Homoscedasticity (Equal Variance)

### The Assumption:
The variance of the residuals must be **constant** across all levels of predicted values $\hat{y}$:

$$\text{Var}(e_i | x_i) = \sigma^2 \quad \text{for all } i$$

If the error spread changes systematically, the model suffers from **Heteroscedasticity** (Unequal Variance).

```
     Homoscedasticity (Good - Equal Band):        Heteroscedasticity (Bad - Funnel Shape):
     
     Residual (e)                                 Residual (e)
        ^                                            ^
     +3 |   •  •  •  •  •  •  •  •                +3 |            •    •      •
      0 +----------------------------> y_hat       0 +---------•----•----•---------> y_hat
     -3 |   •  •  •  •  •  •  •  •                -3 |            •    •      •
        (Constant vertical thickness)                (Funnel widens as y_hat grows!)
```

* **Real-World Example:** Predicting restaurant bill tips. For a ₹100 meal, tips vary between ₹10 and ₹20 (small spread). For a ₹10,000 banquet, tips vary between ₹500 and ₹3,000 (huge spread!). The error spread fans outward like a cone.
* **Why it matters:** OLS coefficients remain unbiased, but their standard error calculations are wrong, invalidating hypothesis tests.
* **How to Check:** Scatter plot of **Residuals ($e$) vs. Predicted Values ($\hat{y}$)**. Look for a **cone / funnel / fan shape**.
* **Remedies if Violated:**
  * Apply a **Log transformation** on the target: $y \to \log(y)$.
  * Use **Weighted Least Squares (WLS)**.

---

## Section 8: Assumption 4 — No Autocorrelation (Independence of Errors)

### The Assumption:
Residuals must be **mutually independent** of one another:

$$\text{Cov}(e_i, e_j) = 0 \quad \text{for all } i \ne j$$

Knowing the sign or size of residual $e_i$ gives you **zero information** about the next residual $e_{i+1}$.

* **When it occurs:** Almost universally in **Time-Series and Sequential Data** (e.g., daily stock prices, quarterly inflation, hourly temperature). If today's temperature is unexpectedly high, tomorrow's temperature is also likely to be higher than predicted.
* **Visualizing Autocorrelation:** Residuals plotted across time show **long streaks of positive errors followed by long streaks of negative errors** (cyclical waves).
* **How to Check:**
  1. Plot **Residuals vs. Time / Index**.
  2. **Durbin-Watson Test Statistic ($d$):**
     * $d \approx 2.0$: **No autocorrelation** (Independent errors ✓).
     * $d < 1.5$: Positive autocorrelation.
     * $d > 2.5$: Negative autocorrelation.
* **Remedies if Violated:**
  * Include **Lagged Features** (e.g., $y_{t-1}, y_{t-2}$).
  * Switch from OLS to dedicated time-series models (**ARIMA / SARIMAX**).

---

## Section 9: Assumption 5 — No (or Little) Multicollinearity

### The Assumption:
The independent features in $\mathbf{X}$ must **not be highly linearly correlated** with one another.

---

### 1. Inference vs. Prediction (The Critical Nuance)
* **If your goal is Pure Prediction:** Multicollinearity is **less harmful**. The model can still make accurate overall predictions $\hat{y}$.
* **If your goal is Inference / Interpretation:** Multicollinearity is **fatal**!
  * Coefficients swing wildly and unpredictably.
  * Standard errors explode, making statistically significant features appear insignificant ($p > 0.05$).
  * You cannot answer: *"Which feature actually drove the result?"*

---

### 2. Perfect Multicollinearity: Mathematical Proof of Singularity

Suppose Feature 2 is an exact duplicate of Feature 1: $X_2 = 2 X_1$.  
In linear algebra, this means the columns of design matrix $\mathbf{X}$ are **linearly dependent**.

#### 🔍 Step-by-Step Proof that $(\mathbf{X}^T \mathbf{X})^{-1}$ Does Not Exist:
1. Because columns of $\mathbf{X}$ are linearly dependent, there exists a **non-zero vector $\mathbf{c} \ne \mathbf{0}$** such that:
   $$\mathbf{X}\mathbf{c} = \mathbf{0}$$
2. Multiply both sides from the left by $\mathbf{X}^T$:
   $$\mathbf{X}^T \mathbf{X}\mathbf{c} = \mathbf{X}^T \mathbf{0}$$
3. Therefore:
   $$\mathbf{X}^T \mathbf{X}\mathbf{c} = \mathbf{0}$$
4. But $\mathbf{c} \ne \mathbf{0}$! This means the square matrix $\mathbf{X}^T \mathbf{X}$ maps a non-zero vector to zero.
5. In linear algebra, any square matrix with a non-trivial null space has **linearly dependent columns**!
6. Therefore, $\mathbf{X}^T \mathbf{X}$ is **singular**:
   $$\det(\mathbf{X}^T \mathbf{X}) = 0$$
7. **Conclusion:** **The inverse $(\mathbf{X}^T \mathbf{X})^{-1}$ does not exist!** The OLS equation crashes.

---

### 3. Worked $3 \times 3$ Singular Matrix Example

Consider design matrix $\mathbf{X}$ where Column 3 is the exact sum of Column 1 and Column 2 ($\text{Col } 3 = \text{Col } 1 + \text{Col } 2$):

$$\mathbf{X} = \begin{bmatrix} 1 & 1 & 2 \\ 1 & 2 & 3 \\ 1 & 3 & 4 \end{bmatrix}$$

Multiplying $\mathbf{X}^T \mathbf{X}$:

$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \\ 2 & 3 & 4 \end{bmatrix} \begin{bmatrix} 1 & 1 & 2 \\ 1 & 2 & 3 \\ 1 & 3 & 4 \end{bmatrix} = \begin{bmatrix} 3 & 6 & 9 \\ 6 & 14 & 20 \\ 9 & 20 & 29 \end{bmatrix}$$

Let's compute the determinant of $\mathbf{X}^T \mathbf{X}$:
$$\det(\mathbf{X}^T \mathbf{X}) = 3(14 \cdot 29 - 20 \cdot 20) - 6(6 \cdot 29 - 20 \cdot 9) + 9(6 \cdot 20 - 14 \cdot 9)$$
$$\det(\mathbf{X}^T \mathbf{X}) = 3(406 - 400) - 6(174 - 180) + 9(120 - 126)$$
$$\det(\mathbf{X}^T \mathbf{X}) = 3(6) - 6(-6) + 9(-6) = 18 + 36 - 54 = \mathbf{0}$$

**Determinant is ZERO!** Inverting this matrix is mathematically impossible.

---

### 4. How to Detect & Fix Multicollinearity:
* **Detection:**
  1. **Correlation Matrix Heatmap:** Look for pairs with correlation $|r| > 0.85$.
  2. **Variance Inflation Factor (VIF):**
     $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
     * $\text{VIF} = 1$: No collinearity.
     * $\text{VIF} > 5$: Moderate collinearity (investigate).
     * $\text{VIF} > 10$: **Severe multicollinearity! Must be fixed!**
* **Remedies:**
  * **Drop one of the correlated features** (e.g., keep `Temperature_Celsius`, drop `Temperature_Fahrenheit`).
  * Combine correlated features using **Principal Component Analysis (PCA)**.
  * Use **Ridge Regression (L2 regularization)**: Adds $\lambda \mathbf{I}$ to guarantee invertibility: $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1}$.

---

## Section 10: Master Diagnostic & Remediation Cheat Sheet

| Assumption | What It Demands | Primary Diagnostic Tool | The Telltale Warning Sign | How to Fix It |
| :--- | :--- | :--- | :--- | :--- |
| **1. Linearity** | Central trend is linear | Residuals vs. $\hat{y}$ plot | Distinct **U-shape** or curve | Add polynomial terms ($x^2$), log transform $x$ |
| **2. Normality** | Errors follow $\mathcal{N}(0, \sigma^2)$ | Residual histogram / Q-Q plot | Skewed tail, points peeling off 45° line | Log/Box-Cox transform $y$, drop extreme anomalies |
| **3. Homoscedasticity** | Constant error spread | Residuals vs. $\hat{y}$ scatter | **Cone / funnel shape** widening outward | Log transform $y$, Weighted Least Squares |
| **4. Independence** | Errors are uncorrelated | Residuals vs. Time plot, Durbin-Watson | Cyclical wave streaks, $d < 1.5$ | Add lag features ($y_{t-1}$), ARIMA models |
| **5. No Multicollinearity** | Features are independent | Correlation heatmap, VIF | Correlation $> 0.85$, $\text{VIF} > 10$ | Drop redundant feature, PCA, Ridge regression |

---

## 📝 Practice P4: True / False Exam Rapid Fire

| Statement | Answer | Rationale |
| :--- | :---: | :--- |
| **(a) Polynomial Regression requires a completely different training algorithm than Linear Regression.** | **False** | It uses the exact same OLS or Gradient Descent algorithm; only the feature coordinates were transformed. |
| **(b) If $R^2$ increases when we add $x^3$, the cubic term is guaranteed to be useful.** | **False** | Raw $R^2$ always increases even for useless noise features. Check **Adjusted $R^2$** or validation loss. |
| **(c) Heteroscedasticity means unequal error spread across the predicted range.** | **True** | By definition, the variance of residuals changes systematically with $\hat{y}$. |
| **(d) Multicollinearity always makes predictions inaccurate.** | **False** | Predictions can still be accurate; only individual coefficient interpretations and p-values are corrupted. |
| **(e) Linearity requires every data point to lie exactly on a straight line.** | **False** | It only requires the **central expected trend** to be linear. Random scatter around the line is normal. |
| **(f) In perfect multicollinearity, $\det(\mathbf{X}^T \mathbf{X}) = 1$.** | **False** | The determinant is **$0$**, making the matrix singular and non-invertible. |

---

## 🚀 Forward Handoff to Next Lecture (Lecture 9)

In this lecture, we saw that as polynomial degree increases (Degree 1 $\to$ 2 $\to$ 8), the model's flexibility changes dramatically:
* Low degrees **underfit** (high bias).
* High degrees **overfit** (high variance).

In **Lecture 9 (The Bias-Variance Tradeoff)**:
* We mathematically decompose expected prediction error into **$\text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$**.
* We explore model complexity curves and learn how to find the optimal sweet spot!

---

## 📋 Summary of Sheet 8 (Key Takeaways)

1. **Polynomial Regression:** Transforms features ($x \to x, x^2, \dots, x^d$) to bend curves while remaining strictly linear in parameters $\boldsymbol{\beta}$.
2. **The 3D Mental Model:** A 2D curve is simply a flat 2D plane in 3D feature space $(x, x^2, y)$ sliced along a parabola.
3. **The 5 Assumptions:** Linearity, Normality of Residuals, Homoscedasticity, No Autocorrelation, and No Multicollinearity.
4. **Residual Plots:** The ultimate diagnostic tool. A healthy model shows a random, uniform swarm around zero.
5. **Multicollinearity Breakdown:** Perfect multicollinearity forces $\det(\mathbf{X}^T \mathbf{X}) = 0$, rendering OLS matrix inversion impossible.
