# Machine Learning Made Simple — Doc 5: Batch Gradient Descent for Multiple Linear Regression
**Worksheet:** [Worksheet 05](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_05_Batch_Gradient_Descent_MLR.pdf)  
**Lab Assignment:** [Lab 05 Batch Gradient Descent MLR](file:///Users/mayanksharma/Downloads/AML/04_Notebooks/Lab_05_Batch_Gradient_Descent_MLR/solved/Lab_5_Final_BGD_MLR_student_todo.ipynb)  
**Topics:** Why OLS Fails at Scale · Geometric Meaning of the Gradient · Downhill Optimization Analogy · The Gradient Descent Update Rule · Simultaneous vs. Sequential Updates · Vectorizing the MSE Loss · Step-by-Step Gradient Derivation $\nabla L = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$ · Complete By-Hand Epoch Calculation · Feature Scale Imbalance & Need for Standardization · Epochs & The Full Training Loop · Learning Rate Dynamics (Too Small vs. Too Large vs. Just Right) · Validation-Based Hyperparameter Tuning · Advantages & Disadvantages of Batch GD

---

## 🗺️ Where Are We in the Journey?

In our previous documents:
* [Doc 1](file:///Users/mayanksharma/Downloads/AML/docs/01_ML_Project_Lifecycle_EDA_and_Preprocessing.md): Problem Framing, Data Collection, EDA, and Preprocessing.
* [Doc 2](file:///Users/mayanksharma/Downloads/AML/docs/02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.md): Data Splitting, Feature Engineering, Evaluation Metrics, and Model Drift.
* [Doc 3](file:///Users/mayanksharma/Downloads/AML/docs/03_Simple_Linear_Regression_OLS.md): Simple Linear Regression with 1 feature, deriving slope $m$ and intercept $c$.
* [Doc 4](file:///Users/mayanksharma/Downloads/AML/docs/04_Multiple_Linear_Regression_OLS.md): Multiple Linear Regression, matrix calculus, and the closed-form Normal Equation: $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$.

In Doc 4, we saw that the closed-form solution has a fatal flaw: **inverting $\mathbf{X}^T \mathbf{X}$ takes $O(m^3)$ operations**. When you have 10,000 or 100,000 features, inverting the matrix runs out of memory and takes days or weeks! Furthermore, if two features are collinear, the matrix cannot be inverted at all!

Now in **Doc 5**, we unlock the fundamental engine of modern artificial intelligence: **Gradient Descent**!  
Instead of trying to calculate the perfect answer in one giant, expensive matrix inversion, Gradient Descent starts with a rough guess and **takes small, iterative steps downhill** until it reaches the optimal weights.

---

## Section 1: Why OLS Is Not Enough (The Two Limitations)

In Lecture 4, the closed-form OLS formula seemed like pure magic:

$$\boldsymbol{\theta}^* = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$$

*(In this worksheet, we use $\boldsymbol{\theta}$ or $\boldsymbol{\beta}$ interchangeably to denote the parameter vector).*

There is no trial and error, no learning rate, and no loops. So why do we need another method?

---

### Limitation 1: The Computational Bottleneck ($O(m^3)$ Scaling)
If our dataset has $m$ features, the matrix $\mathbf{X}^T \mathbf{X}$ has dimensions $(m+1) \times (m+1)$.  
Inverting an $(m+1) \times (m+1)$ matrix using Gauss-Jordan elimination requires approximately:

$$\mathbf{O(m^3) \text{ operations}}$$

* **Small dataset ($m = 5$ features):** $5^3 = 125$ operations $\to$ Completed in $0.000001$ seconds.
* **Computer vision / GenAI ($m = 50,000$ features):**  
  $$(50,000)^3 = 125,000,000,000,000 = 1.25 \times 10^{14} \text{ operations!}$$  
  Even on a powerful modern server, forming and inverting a $50,001 \times 50,001$ matrix requires **hundreds of gigabytes of RAM** and would take **hours or days**!

---

### Limitation 2: Singular Matrices & Multicollinearity
The closed-form formula strictly requires that $(\mathbf{X}^T \mathbf{X})$ is invertible ($\det(\mathbf{X}^T \mathbf{X}) \ne 0$).  
If two features are exact duplicates (e.g., `Price_USD` and `Price_Cents = 100 * Price_USD`), the matrix becomes **singular**. The inverse does not exist, and OLS crashes with a `LinAlgError: Singular matrix`.

> 💡 **The Solution:**  
> We need an optimization method that **never inverts a matrix**.  
> That method is **Batch Gradient Descent**!

---

### 📝 Practice P1: Scenario Evaluation

A data science team is deciding whether to use analytical OLS or Batch Gradient Descent:

1. **Scenario A:** Dataset has 10,000 rows but only 3 features (`Age`, `Income`, `Credit_Score`).  
   * **Recommendation:** **OLS (Normal Equation)**.  
   * **Why?** With only 3 features, $\mathbf{X}^T \mathbf{X}$ is a tiny $4 \times 4$ matrix. Inverting a $4 \times 4$ matrix takes microseconds. (Note: The number of rows, 10,000, does not affect the size of the matrix being inverted).
2. **Scenario B:** A computer-vision regression model uses 50,000 image features to predict manufacturing defect severity.  
   * **Recommendation:** **Batch Gradient Descent**.  
   * **Why?** Inverting a $50,001 \times 50,001$ matrix costs $O(m^3) \approx 10^{14}$ operations and would exhaust system memory. Gradient Descent avoids matrix inversion entirely.
3. **Scenario C:** A junior engineer accidentally includes both `Price_USD` and `Price_Cents` as inputs ($100 \times \text{Price\_USD}$).  
   * **What error will OLS throw?** `Singular Matrix` (non-invertible matrix due to perfect multicollinearity, $\det = 0$).  
   * **Can Batch Gradient Descent still run?** **YES!** Gradient Descent does not compute an inverse. It can still navigate down the loss surface, though the learned weights may not be unique.

---

## Section 2: Introduction to Gradient Descent (Intuition & Geometry)

### 1. The Foggy Mountain Analogy
> **The Story:** Imagine you are blindfolded on a foggy mountain in a dense mist. You cannot see the village at the bottom of the valley. How do you find your way down to the lowest point?  
> 
> * You feel the slope of the ground beneath your feet with your boots.  
> * If the ground slopes steeply upward in front of you, you **step in the exact opposite direction (downward)**.  
> * You take a step, pause, feel the new slope, and take another step downward.  
> * By repeating this simple process over and over, you will eventually reach the bottom of the valley!

In Machine Learning:
* **The Mountain:** The Loss Function surface $L(\boldsymbol{\theta})$ (our prediction error).
* **The Valley Floor:** The minimum error (the best weights $\boldsymbol{\theta}^*$).
* **Feeling the Slope:** Calculating the **Gradient** ($\nabla L$).
* **Taking a Step Downhill:** Updating the weights in the **negative gradient direction** ($-\nabla L$).

```
        Loss L(θ)
           ^
           |         \               /
           |          \             /   <--- Convex Loss Surface
           |           \           /
           |      Step  \  • (Current θ)
           |             \  |
           |              v |  Slope = +ve (Gradient points uphill)
           |               \|  Negative Gradient points downhill!
           |                •  New θ = θ - α * Gradient
           |                 \
           |                  • Global Minimum (Bottom of Bowl)
           +---------------------------------------------> θ
```

---

### 2. Calculus Recap: Derivatives & Gradients

#### Single-Variable Case ($f(x)$):
The derivative $f'(x) = \frac{df}{dx}$ gives the instantaneous slope of the function:
* If $f'(x) > 0$: The function is rising to the right $\to$ Move **left** (decrease $x$).
* If $f'(x) < 0$: The function is falling to the right $\to$ Move **right** (increase $x$).
* In both cases: **Move in the opposite direction of the derivative: $-f'(x)$!**

#### Multivariable Case ($f(\theta_0, \theta_1, \dots, \theta_m)$):
When a function depends on multiple parameters, the slope in all directions is packaged into a vector of partial derivatives called the **Gradient Vector ($\nabla L$)**:

$$\nabla L(\boldsymbol{\theta}) = \begin{bmatrix} \frac{\partial L}{\partial \theta_0} \\ \frac{\partial L}{\partial \theta_1} \\ \vdots \\ \frac{\partial L}{\partial \theta_m} \end{bmatrix}$$

> 🌟 **The Fundamental Geometric Fact:**  
> * The gradient vector $\nabla L$ points in the direction of **steepest increase** (fastest uphill).  
> * Therefore, the negative gradient $-\nabla L$ points in the direction of **steepest decrease** (fastest downhill)!

---

### 3. The Coordinate-Wise Update Rule

For a single parameter $\theta$, the update equation after one step is:

$$\theta_{\text{new}} = \theta_{\text{old}} - \alpha \frac{dL}{d\theta}\Bigg|_{\theta_{\text{old}}}$$

* **$\alpha$ (Alpha):** The **Learning Rate** (a positive hyperparameter chosen by the engineer that controls how big of a step we take).
* **$\frac{dL}{d\theta}$:** The slope at the current position.
* **The Minus Sign ($-$):** Ensures we step downhill toward lower loss, rather than uphill!

For Multiple Linear Regression with $m+1$ parameters $(\theta_0, \theta_1, \dots, \theta_m)$:

$$\theta_{j,\text{new}} = \theta_{j,\text{old}} - \alpha \frac{\partial L}{\partial \theta_j}\Bigg|_{\boldsymbol{\theta}_{\text{old}}} \quad \text{for } j = 0, 1, \dots, m$$

---

### ⚠️ WARNING: The Simultaneous Update Rule!

When updating parameters in code, **all partial derivatives must be computed simultaneously using $\boldsymbol{\theta}_{\text{old}}$** before any parameter is updated:

```python
# ❌ WRONG (Sequential Update - Corrupts the gradient!)
theta_0 = theta_0 - alpha * grad_0(theta_0, theta_1)
theta_1 = theta_1 - alpha * grad_1(theta_0, theta_1)  # Uses NEW theta_0! Bad!

# ✅ CORRECT (Simultaneous Update)
temp_0 = theta_0 - alpha * grad_0(theta_0, theta_1)
temp_1 = theta_1 - alpha * grad_1(theta_0, theta_1)
theta_0 = temp_0
theta_1 = temp_1
```

---

### 4. The Vectorized Update Rule

Instead of writing $m+1$ separate loops, we write all updates in a single, compact vector equation:

$$\mathbf{\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \alpha \nabla L(\boldsymbol{\theta}_{\text{old}})}$$

```
      θ_new           =         θ_old          -        α        *       ∇L(θ_old)
  (Updated Weights)         (Current Weights)      (Step Size)          (Gradient Vector)
  
   [ (m+1) x 1 ]             [ (m+1) x 1 ]           Scalar              [ (m+1) x 1 ]
   
     +-------+                 +-------+                                   +-----------+
     |  θ0   |                 |  θ0   |                                   |  ∂L/∂θ0   |
     |  θ1   |        =        |  θ1   |       -        α        *         |  ∂L/∂θ1   |
     |  ...  |                 |  ...  |                                   |    ...    |
     |  θm   |                 |  θm   |                                   |  ∂L/∂θm   |
     +-------+                 +-------+                                   +-----------+
```

---

### Parameter Initialization:
* **Convex Loss Functions (Linear & Logistic Regression):** The error surface is a single bowl with only one minimum. We can safely initialize $\boldsymbol{\theta}_{\text{old}} = \mathbf{0}$ (the zero vector). It is guaranteed to reach the global minimum regardless of where it starts.
* **Non-Convex Loss Functions (Deep Neural Networks):** Surfaces have many local traps and saddle points, so parameters are initialized randomly (covered in the Deep Learning course).

---

## Section 3: Deriving the Batch Gradient Descent Update for MSE

Let's derive the exact mathematical formula for the gradient vector $\nabla L$ when using **Mean Squared Error (MSE)**.

### Step 1: Define the MSE Loss Function
For $n$ training samples and design matrix $\mathbf{X} \in \mathbb{R}^{n \times (m+1)}$, the Mean Squared Error is:

$$L(\boldsymbol{\theta}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \mathbf{e}^T \mathbf{e} = \frac{1}{n} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$

*(Note: $(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})$ because squaring removes any sign).*

---

### Step 2: Expand the Matrix Loss Function
Using the transpose property $(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T = \boldsymbol{\theta}^T \mathbf{X}^T - \mathbf{y}^T$:

$$L(\boldsymbol{\theta}) = \frac{1}{n} \left[ \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{y} - \mathbf{y}^T \mathbf{X}\boldsymbol{\theta} + \mathbf{y}^T \mathbf{y} \right]$$

Since $\boldsymbol{\theta}^T \mathbf{X}^T \mathbf{y}$ is a scalar, it equals its transpose $\mathbf{y}^T \mathbf{X}\boldsymbol{\theta}$. Combining the two middle terms:

$$L(\boldsymbol{\theta}) = \frac{1}{n} \left[ \mathbf{y}^T \mathbf{y} - 2\mathbf{y}^T \mathbf{X}\boldsymbol{\theta} + \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\theta} \right]$$

---

### Step 3: Differentiate with Respect to $\boldsymbol{\theta}$ Using Our Matrix Identities
From Doc 4, we recall our three matrix calculus identities:
1. $\frac{\partial (\mathbf{y}^T \mathbf{y})}{\partial \boldsymbol{\theta}} = \mathbf{0}$
2. $\frac{\partial (-2\mathbf{y}^T \mathbf{X}\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = -2\mathbf{X}^T \mathbf{y}$
3. $\frac{\partial (\boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = 2\mathbf{X}^T \mathbf{X}\boldsymbol{\theta}$

Applying these to our loss:

$$\nabla L = \frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{1}{n} \left[ \mathbf{0} - 2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X}\boldsymbol{\theta} \right]$$

$$\nabla L = \frac{1}{n} \left[ -2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X}\boldsymbol{\theta} \right]$$

---

### Step 4: Factor Out $2\mathbf{X}^T$ to Get the Final Gradient Formula!
$$\mathbf{\nabla L(\boldsymbol{\theta}) = \frac{2}{n} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) = \frac{2}{n} \mathbf{X}^T (\mathbf{\hat{y}} - \mathbf{y})}$$

Look at how intuitive this gradient is:
* $\mathbf{\hat{y}} - \mathbf{y}$: The **residual vector** (how wrong our current predictions are).
* $\mathbf{X}^T$: Projects these errors onto each input feature column.
* $\frac{2}{n}$: Averages the gradient across all $n$ samples.

---

### Step 5: The Final Batch Gradient Descent Update Rule

Plugging our gradient $\nabla L$ directly into the vector update rule:

$$\mathbf{\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \alpha \frac{2}{n} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta}_{\text{old}} - \mathbf{y})}$$

> 🌟 **Why is it called "Batch" Gradient Descent?**  
> Because to compute this single update step, the algorithm uses the **entire batch of all $n$ rows in the dataset** ($\mathbf{X}$ and $\mathbf{y}$). Every single student in the database contributes to each step.

*(Note on Conventions: Some textbooks define the loss with an extra $\frac{1}{2}$ as $L = \frac{1}{2n}\|\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\|^2$ solely to cancel out the factor of $2$. In this course, we adhere to the strict standard MSE definition, keeping the factor of $2$).*

---

## Section 4: Worked Example on the NST Student Dataset

Let's compute **one full update step (Epoch 1)** completely by hand using a real dataset of 3 students ($n = 3$):
* $x_1$: Study Hours
* $x_2$: Attendance Percentage
* $y$: Exam Score

| Student | Study Hours ($x_1$) | Attendance % ($x_2$) | Exam Score ($y$) |
| :---: | :---: | :---: | :---: |
| **1** | 2 | 60 | 35 |
| **2** | 4 | 80 | 55 |
| **3** | 6 | 90 | 70 |

* **Hyperparameters:**
  * Initial parameters: $\boldsymbol{\theta}_{\text{old}} = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$
  * Learning rate: $\alpha = 0.0001$

---

### Step 1: Write the Design Matrix $\mathbf{X}$ and Target Vector $\mathbf{y}$
$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 60 \\ 1 & 4 & 80 \\ 1 & 6 & 90 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 35 \\ 55 \\ 70 \end{bmatrix}$$

---

### Step 2: Compute Current Predictions $\mathbf{\hat{y}}$
Since all weights are initially zero:
$$\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\theta}_{\text{old}} = \begin{bmatrix} 1 & 2 & 60 \\ 1 & 4 & 80 \\ 1 & 6 & 90 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$$

---

### Step 3: Compute the Residual Vector $(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$
$$\mathbf{X}\boldsymbol{\theta} - \mathbf{y} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} - \begin{bmatrix} 35 \\ 55 \\ 70 \end{bmatrix} = \begin{bmatrix} -35 \\ -55 \\ -70 \end{bmatrix}$$

---

### Step 4: Compute the Matrix Product $\mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$
Transpose $\mathbf{X}$ and multiply by the residual vector:

$$\mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 4 & 6 \\ 60 & 80 & 90 \end{bmatrix} \begin{bmatrix} -35 \\ -55 \\ -70 \end{bmatrix}$$

* **Row 1 (Bias gradient component):**  
  $1(-35) + 1(-55) + 1(-70) = -35 - 55 - 70 = \mathbf{-160}$
* **Row 2 (Hours gradient component):**  
  $2(-35) + 4(-55) + 6(-70) = -70 - 220 - 420 = \mathbf{-710}$
* **Row 3 (Attendance gradient component):**  
  $60(-35) + 80(-55) + 90(-70) = -2100 - 4400 - 6300 = \mathbf{-12,800}$

$$\mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) = \begin{bmatrix} -160 \\ -710 \\ -12,800 \end{bmatrix}$$

---

### Step 5: Compute the MSE Gradient Vector $\nabla L$
Multiply by $\frac{2}{n} = \frac{2}{3}$:

$$\nabla L = \frac{2}{3} \begin{bmatrix} -160 \\ -710 \\ -12,800 \end{bmatrix} = \begin{bmatrix} \frac{-320}{3} \\ \frac{-1420}{3} \\ \frac{-25600}{3} \end{bmatrix} \approx \begin{bmatrix} \mathbf{-106.67} \\ \mathbf{-473.33} \\ \mathbf{-8533.33} \end{bmatrix}$$

---

### Step 6: Update Parameters Simultaneously
Using $\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \alpha \nabla L$ with $\alpha = 0.0001$:

$$\boldsymbol{\theta}_{\text{new}} = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix} - 0.0001 \begin{bmatrix} -106.67 \\ -473.33 \\ -8533.33 \end{bmatrix}$$

$$\boldsymbol{\theta}_{\text{new}} = \begin{bmatrix} 0 - (-0.01067) \\ 0 - (-0.04733) \\ 0 - (-0.85333) \end{bmatrix} \approx \mathbf{\begin{bmatrix} 0.011 \\ 0.047 \\ 0.853 \end{bmatrix}}$$

**Epoch 1 is complete!**  
All three parameters have moved from $0$ into positive territory, reflecting that both hours and attendance positively contribute to score!

---

### 📝 Practice P4: Complete Epoch 2 By Hand

Using our updated parameters $\boldsymbol{\theta} = [0.011, 0.047, 0.853]^T$:

1. **Compute new predictions $\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\theta}$:**
   * $\hat{y}_1 = 1(0.011) + 2(0.047) + 60(0.853) = 0.011 + 0.094 + 51.18 = \mathbf{51.285}$
   * $\hat{y}_2 = 1(0.011) + 4(0.047) + 80(0.853) = 0.011 + 0.188 + 68.24 = \mathbf{68.439}$
   * $\hat{y}_3 = 1(0.011) + 6(0.047) + 90(0.853) = 0.011 + 0.282 + 76.77 = \mathbf{77.063}$

2. **Compute the new residual vector $(\mathbf{\hat{y}} - \mathbf{y})$:**
   $$\mathbf{\hat{y}} - \mathbf{y} = \begin{bmatrix} 51.285 - 35 \\ 68.439 - 55 \\ 77.063 - 70 \end{bmatrix} = \begin{bmatrix} \mathbf{+16.285} \\ \mathbf{+13.439} \\ \mathbf{+7.063} \end{bmatrix}$$
   *(Notice that predictions overshot the actual scores, so residuals became positive, which will cause the next gradient update to push weights downward!)*

---

### 💡 The Feature Scale Imbalance Trap (Crucial Insight!)

Look closely at our parameter updates in Epoch 1:
* $\theta_1$ (Study Hours) increased by only **$+0.047$**
* $\theta_2$ (Attendance) increased by a massive **$+0.853$** ($18\times$ larger!)

**Why was the update for Attendance so much larger?**  
Not because attendance is $18\times$ more important, but simply because **the raw numbers for attendance are $60 - 90$**, while **study hours are only $2 - 6$**!  
When you compute $\mathbf{X}^T \mathbf{e}$, the feature values multiply directly into the gradient. Features with large magnitudes produce gigantic gradients, dominating the parameter updates.

```
       Unscaled Features:                     Scaled Features (Normalized):
       (Elongated, narrow ravine)              (Symmetric, circular bowl)
       
       θ2 ^                                    θ2 ^
          |      /==========/                     |         /-------\
          |     /  Valley  /                      |        /         \
          |    /==========/                       |        \         /
          +--------------------> θ1               +--------------------> θ1
          (Oscillates wildly back                 (Direct, straight path
           and forth, slow descent)                to the center minimum!)
```

> 📌 **Golden ML Rule:**  
> **Always perform Feature Scaling (Standardization / Z-score Normalization) before running Gradient Descent!**  
> Rescaling all features to have mean 0 and standard deviation 1 transforms stretched, distorted loss ravines into smooth, spherical bowls, allowing gradient descent to converge up to $100\times$ faster!

---

## Section 5: Epochs and the Full Training Loop

Training a machine learning model means repeating this update process over and over until the parameters stop changing.

### Key Definitions:
* **Iteration:** A single update of the parameter vector $\boldsymbol{\theta}$.
* **Epoch:** One complete pass through the entire training dataset.  
  *(In Batch Gradient Descent, **1 Iteration = 1 Epoch**, because every single step processes the whole dataset).*
* **Convergence:** The state where the loss function stops decreasing and the parameters stabilize ($\|\boldsymbol{\theta}_{\text{new}} - \boldsymbol{\theta}_{\text{old}}\| < \epsilon$, where $\epsilon \approx 10^{-5}$).

---

### The Complete Batch GD Training Loop

```
                     +---------------------------------------+
                     | 1. Initialize θ = [0, 0, ..., 0]^T    |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | 2. Set Hyperparameters:               |
                     |    Learning Rate α, Max Epochs        |
                     +-------------------+-------------------+
                                         |
    +------------------------------------+<-----------------------------------+
    |                                    |                                    |
    |                                    v                                    |
    |                +---------------------------------------+                |
    |                | 3. Compute Predictions:               |                |
    |                |    y_hat = X * θ                      |                |
    |                +-------------------+-------------------+                |
    |                                    |                                    |
    |                                    v                                    |
    |                +---------------------------------------+                |
    |                | 4. Compute Residuals:                 |                |
    |                |    e = X * θ - y                      |                |
    |                +-------------------+-------------------+                |
    |                                    |                                    |
    |                                    v                                    |
    |                +---------------------------------------+                |
    |                | 5. Compute Full Batch Gradient:       |                |
    |                |    grad = (2/n) * X^T * e             |                |
    |                +-------------------+-------------------+                |
    |                                    |                                    |
    |                                    v                                    |
    |                +---------------------------------------+                |
    |                | 6. Update Parameters Simultaneously:  |                |
    |                |    θ = θ - α * grad                   |                |
    |                +-------------------+-------------------+                |
    |                                    |                                    |
    |                                    v                                    |
    |                +---------------------------------------+                |
    |                | 7. Check Convergence Criteria:        |                |
    |                |    Is change < ε OR Epoch == Max?     |                |
    |                +-------------------+-------------------+                |
    |                                    |                                    |
    |                     NO             |             YES                    |
    +------------------------------------+------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     | STOP: Return optimal parameters θ*    |
                     +---------------------------------------+
```

---

### 📝 Practice P5: Epoch Trivia
* **Question:** If the NST dataset has 3 students and we train for 100 epochs using Batch Gradient Descent, how many times does the model compute a gradient using the full dataset?  
* **Answer:** Exactly **100 times** (once per epoch).

---

## Section 6: The Learning Rate ($\alpha$) & Hyperparameter Tuning

The learning rate $\alpha$ is the single most critical dial in gradient descent. Because it is chosen by the human engineer *before* training starts, it is a **Hyperparameter**.

```
    Case 1: α is Too Small               Case 2: α is Too Large             Case 3: α is Just Right
    
    Loss                                 Loss                                Loss
     ^                                    ^                                   ^
     | \                                  |    \   /\     / (Diverges!)       | \
     |  \                                 |     \ /  \   /                    |  \
     |   \                                |      V    \_/                     |   \
     |    \________ (Painfully slow)      |                                   |    \__________ (Optimal!)
     +---------------------> Epoch        +---------------------> Epoch       +---------------------> Epoch
```

* **Case 1 ($\alpha$ too small, e.g., $10^{-7}$):**  
  The model steps in the right direction, but steps are microscopic. It may take 1,000,000 epochs to reach the bottom, wasting CPU time and electricity.
* **Case 2 ($\alpha$ too large, e.g., $5.0$):**  
  The step is so huge that it completely leaps over the valley floor and lands higher up on the opposite mountain slope! The error explodes and diverges toward infinity (`NaN` or `OverflowError`).
* **Case 3 ($\alpha$ well-chosen, e.g., $0.01$):**  
  The loss drops steeply during early epochs, then gracefully flattens as it approaches the bottom of the bowl.

---

### How to Tune $\alpha$ Using a Validation Set (Never Test Set!)

Never use the Test Set to choose hyperparameters! The test set must remain locked in a vault until the very end.  
Instead, we tune $\alpha$ on a **held-out Validation Set**:

$$L_{\text{val}} = \frac{1}{n_{\text{val}}} \sum_{i=1}^{n_{\text{val}}} (y_i - \hat{y}_i)^2$$

### The Step-by-Step Tuning Protocol:
1. Try candidate learning rates on a **logarithmic scale**: e.g., $\alpha \in \{0.1, 0.01, 0.001\}$.
2. Train a separate model for each candidate $\alpha$ starting from the exact same initial weights ($\boldsymbol{\theta} = \mathbf{0}$).
3. Track the validation loss curve across epochs.
4. Immediately eliminate candidates that diverge or oscillate wildly.
5. Pick the $\alpha$ that achieves the **lowest validation loss**.

---

### 📝 Guided Dry Run: Selecting the Best $\alpha$

Suppose our validation set has 3 residuals: $e = [2, -1, 3]^T$.  
The validation loss is:
$$L_{\text{val}} = \frac{2^2 + (-1)^2 + 3^2}{3} = \frac{4 + 1 + 9}{3} = \frac{14}{3} \approx \mathbf{4.67}$$

We train three models using different learning rates and observe their final validation losses:

| Candidate Learning Rate ($\alpha$) | Final Validation Loss ($L_{\text{val}}$) | Outcome |
| :---: | :---: | :--- |
| $\alpha = 0.1$ | $18.4$ | Suboptimal (too aggressive, oscillations near bottom) |
| $\alpha = 0.01$ | **$6.2$** | **WINNER! Lowest validation error!** |
| $\alpha = 0.001$ | $9.7$ | Too slow (underfitting within the epoch budget) |

* **Selection:** We choose **$\alpha = 0.01$** because it achieved the lowest error on unseen validation data.

---

## Section 7: Advantages & Disadvantages of Batch Gradient Descent

| Advantages | Disadvantages |
| :--- | :--- |
| **1. Never inverts a matrix:** Scales easily to datasets with 100,000+ features where OLS crashes. | **1. Computationally heavy per step:** Every single parameter update requires scanning all $n$ rows. |
| **2. Handles collinearity:** Can run even when $\mathbf{X}^T \mathbf{X}$ is singular and OLS fails. | **2. Impractical for Big Data:** On a dataset with 50,000,000 rows, a single step takes minutes! |
| **3. Guaranteed global minimum:** Because MSE is strictly convex, it is mathematically guaranteed to reach the global optimum (with an appropriate $\alpha$). | **3. Requires full dataset in RAM:** The entire design matrix $\mathbf{X}$ must be loaded into memory. |
| **4. Smooth, stable descent:** Using all samples produces a clean, noise-free gradient vector pointing straight downhill. | **4. Sensitive to learning rate:** Bad $\alpha$ choices lead to glacial crawl or explosive divergence. |

---

## 📝 Practice P6: True / False Exam Rapid Fire

| Statement | Answer | Rationale |
| :--- | :---: | :--- |
| **(a) If the gradient points downhill, the loss is guaranteed to decrease after the parameter update, regardless of the learning rate.** | **False** | If $\alpha$ is too large, the algorithm will overshoot the valley and land higher up the opposite wall, causing the loss to explode. |
| **(b) We use the test set to choose the best learning rate $\alpha$.** | **False** | The test set must be kept locked for final unbiased evaluation. Hyperparameters must be tuned on the **Validation Set**. |
| **(c) Batch Gradient Descent explicitly computes the inverse of $\mathbf{X}^T \mathbf{X}$.** | **False** | BGD never computes any matrix inverse—it only performs matrix-vector multiplications. |
| **(d) Because Batch GD uses all training examples in every update, it is well-suited for a dataset with 50 million rows.** | **False** | Scanning 50 million rows for every single step is painfully slow. For massive row counts, **Mini-Batch GD** (Doc 6) is preferred! |

---

## 🚀 Forward Handoff to Next Lecture (Lecture 6)

In this lecture, we solved the feature-scaling problem of OLS ($O(m^3)$) by taking iterative steps downhill.  
However, notice the new bottleneck:
* When $n$ (the number of data points) is in the **millions or billions**, calculating $\frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$ on every single step is painfully slow!

In **Lecture 6 (Stochastic & Mini-Batch Gradient Descent)**:
* We discover how taking quick, noisy steps on small subsets (**mini-batches of 32, 64, or 128 samples**) allows models to train up to $1,000\times$ faster!
* We will explore **Stochastic Gradient Descent (SGD)**, batch size tradeoffs, and the mathematics of noisy optimization.

---

## 📋 Summary of Sheet 5 (Key Takeaways)

1. **Why Gradient Descent?** OLS requires $O(m^3)$ matrix inversion and crashes on singular matrices. Gradient descent avoids matrix inversion completely.
2. **The Gradient ($\nabla L$):** Points in the direction of steepest increase. Stepping in the **negative gradient ($-\nabla L$)** guarantees downhill progress on convex surfaces.
3. **Simultaneous Updates:** All partial derivatives must be computed at the current position before updating parameters.
4. **The MSE Gradient Formula:**
   $$\nabla L = \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$$
5. **The Vector Update Rule:**
   $$\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \alpha \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta}_{\text{old}} - \mathbf{y})$$
6. **Feature Scaling is Mandatory:** Unstandardized features warp the loss surface into narrow canyons, causing wild oscillations.
7. **Tuning $\alpha$:** Use logarithmic candidate searches on a validation set to find the sweet spot between divergence and slow convergence.
