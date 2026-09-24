# Machine Learning Made Simple — Doc 4: Multiple Linear Regression & The Normal Equation (OLS)
**Worksheet:** [Worksheet 04](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_04_Multiple_Linear_Regression_OLS.pdf)  
**Lab Assignment:** [Lab 04 Multiple Linear Regression OLS](file:///Users/mayanksharma/Downloads/AML/04_Notebooks/Lab_04_Multiple_Linear_Regression_OLS/solved/Lab_4_student_notebook.ipynb)  
**Topics:** Multiple Features & Hyperplanes · The Design Matrix $\mathbf{X}$ · The Bias Trick (Column of 1s) · Vectorizing Predictions $\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\beta}$ · Residual Vectors & Loss as $\mathbf{e}^T \mathbf{e}$ · Expanding the Loss Function · Matrix Calculus Identities (Constant, Linear, Quadratic Forms) · Full Derivation of the Normal Equation · The Closed-Form Solution $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$ · Multicollinearity & Invertibility · Computational Complexity $O(m^3)$ · Complete $3 \times 3$ Hand Calculation Walkthrough · SLR vs. MLR Deep Comparison

---

## 🗺️ Where Are We in the Journey?

In our previous documents:
* [Doc 1](file:///Users/mayanksharma/Downloads/AML/docs/01_ML_Project_Lifecycle_EDA_and_Preprocessing.md): Problem Definition, Data Collection, EDA, and Preprocessing.
* [Doc 2](file:///Users/mayanksharma/Downloads/AML/docs/02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.md): Data Splitting, Feature Engineering, Evaluation Metrics, and Model Drift.
* [Doc 3](file:///Users/mayanksharma/Downloads/AML/docs/03_Simple_Linear_Regression_OLS.md): Simple Linear Regression with **one feature ($x$)**, deriving slope $m$ and intercept $c$ using standard calculus.

Now in **Doc 4**, we graduate to the real world: **Multiple Linear Regression (MLR)**!  
In real problems, a single feature is rarely enough. A student's stipend depends on CGPA, IQ, coding projects, and internship experience. An apartment's rent depends on square footage, bedrooms, floor number, and distance to the metro.

When you have 5, 50, or 500 features, solving for each slope individually with scalar calculus becomes an unmanageable nightmare.  
**The solution? Matrix Algebra!**  
In this document, we will discover how compact matrix notation compresses hundreds of equations into a single line, derive the legendary **Normal Equation** using matrix calculus, compute a full $3 \times 3$ model by hand, and understand why large datasets eventually force us toward **Gradient Descent**.

---

## Section 1: The MLR Dataset — From a Line to a Hyperplane

### 1. The Real-World Dilemma
In Lecture 3, we predicted student stipends using only CGPA:
$$\hat{y} = m(\text{CGPA}) + c$$

What if two students both have an 8.5 CGPA, but one built 10 full-stack applications while the other never wrote a line of code outside class? They will clearly receive different stipends!  
To build accurate models, we must feed our algorithm multiple input features:
* $x_1$ = CGPA
* $x_2$ = IQ score
* $x_3$ = Hackathon projects completed
* $x_4$ = Attendance percentage

---

### 2. General Data Representation
In Multiple Linear Regression, we organize our dataset into a table of $n$ rows (data points / samples) and $m$ columns (input features):

| Feature $X_1$ | Feature $X_2$ | Feature $X_3$ | $\dots$ | Feature $X_m$ | Actual Target ($Y$) | Model Prediction ($\hat{Y}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $x_{11}$ | $x_{12}$ | $x_{13}$ | $\dots$ | $x_{1m}$ | $y_1$ | $\hat{y}_1$ |
| $x_{21}$ | $x_{22}$ | $x_{23}$ | $\dots$ | $x_{2m}$ | $y_2$ | $\hat{y}_2$ |
| $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ | $\vdots$ | $\vdots$ |
| $x_{n1}$ | $x_{n2}$ | $x_{n3}$ | $\dots$ | $x_{nm}$ | $y_n$ | $\hat{y}_n$ |

* **$n$:** The number of data points / observations (rows).
* **$m$:** The number of features / predictors (columns).
* **$x_{ij}$:** The value of feature $j$ for student $i$.
* **$y_i$:** The actual ground-truth label for sample $i$.
* **$\hat{y}_i$:** The model's predicted output for sample $i$.

---

### 3. The MLR Prediction Equation
For any input point with $m$ features $(x_1, x_2, \dots, x_m)$, the prediction equation is:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \dots + \beta_m x_m$$

* **$\beta_0$ (Beta-zero):** The **intercept** (the predicted value when all features $x_1, \dots, x_m = 0$).
* **$\beta_j$ ($j = 1, \dots, m$):** The **partial regression coefficient** for feature $j$.  
  *(It represents the expected change in $\hat{y}$ when feature $x_j$ increases by 1 unit, **holding all other features constant**).*

```
1 Input Feature (SLR):           2 Input Features (MLR):            m > 2 Features (MLR):
y = β0 + β1*x                    y = β0 + β1*x1 + β2*x2             y = β0 + β1*x1 + ... + βm*xm

   y                                y                                  y
   ^                                ^                                  ^
   |     /                          |      /---------/                 |      [ Hyperplane ]
   |    /   (Line in 2D)            |     /         / (Plane in 3D)    |     (Flat surface in
   |   /                            |    /---------/                   |      m+1 dimensions)
   +-----------> x                  +-----------------> x2             +-----------------> x
                                     \
                                      \-> x1
```

> 🌟 **Key Geometric Insight:**  
> Just like an equation with 1 input feature defines a **1D line** in 2D space, and an equation with 2 input features defines a **2D flat plane** in 3D space, an equation with $m$ features defines a **flat $m$-dimensional hyperplane** living in $(m+1)$-dimensional space!

---

### 📝 Practice P1: MLR in Context

A real-estate agency wants to predict monthly apartment rent using 4 features:
* $x_{\text{area}}$: Floor area in sqft
* $x_{\text{beds}}$: Number of bedrooms
* $x_{\text{metro}}$: Distance to the nearest metro station in km
* $x_{\text{floor}}$: Floor level in the building

1. **Write the MLR prediction equation and count total parameters:**  
   $$\hat{y} = \beta_0 + \beta_1 x_{\text{area}} + \beta_2 x_{\text{beds}} + \beta_3 x_{\text{metro}} + \beta_4 x_{\text{floor}}$$  
   * **Total Parameters:** **5 parameters** ($\beta_0, \beta_1, \beta_2, \beta_3, \beta_4$).  
   *(A model with $m$ features always has $m + 1$ learnable parameters due to the intercept $\beta_0$).*

2. **If you collect a 5th feature (Age of the building in years) and refit the model, will $\beta_1$ (for area) stay the same?**  
   * **Answer:** **No!**  
   * **Why?** In MLR, all coefficients are estimated **jointly**. Because real-world features are often correlated with each other (e.g., older buildings might have different average room sizes), adding a new feature shifts the optimal values of existing coefficients.

---

## Section 2: Writing Predictions for Each Data Point

If we have $n$ students, we can write out the prediction for every single row individually:

$$\begin{aligned}
\text{Row 1:} \quad \hat{y}_1 &= \beta_0 + \beta_1 x_{11} + \beta_2 x_{12} + \dots + \beta_m x_{1m} \\
\text{Row 2:} \quad \hat{y}_2 &= \beta_0 + \beta_1 x_{21} + \beta_2 x_{22} + \dots + \beta_m x_{2m} \\
\vdots \quad\quad &\quad\quad\quad\quad \vdots \\
\text{Row } i\text{:} \quad \hat{y}_i &= \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_m x_{im} \\
\vdots \quad\quad &\quad\quad\quad\quad \vdots \\
\text{Row } n\text{:} \quad \hat{y}_n &= \beta_0 + \beta_1 x_{n1} + \beta_2 x_{n2} + \dots + \beta_m x_{nm}
\end{aligned}$$

Stacking all $n$ predictions into a single column vector $\mathbf{\hat{y}}$:

$$\mathbf{\hat{y}} = \begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n \end{bmatrix} = \begin{bmatrix} \beta_0 + \beta_1 x_{11} + \beta_2 x_{12} + \dots + \beta_m x_{1m} \\ \beta_0 + \beta_1 x_{21} + \beta_2 x_{22} + \dots + \beta_m x_{2m} \\ \vdots \\ \beta_0 + \beta_1 x_{n1} + \beta_2 x_{n2} + \dots + \beta_m x_{nm} \end{bmatrix}$$

Notice the beautiful structure:
* The coefficients $(\beta_0, \beta_1, \dots, \beta_m)$ are **identical across all rows** (these are the fixed model parameters we want to learn).
* Only the input feature values change from row to row!

---

### 📝 Practice P2: Numerical Predictions

A teacher uses CGPA ($x_1$) and Weekly Study Hours ($x_2$) to predict final exam scores using the trained model:

$$\hat{y} = 10 + 5x_1 + 2x_2$$

| Student | CGPA ($x_1$) | Study Hours ($x_2$) | Calculation ($10 + 5x_1 + 2x_2$) | Predicted Score ($\hat{y}$) |
| :---: | :---: | :---: | :---: | :---: |
| **Arjun** | 8 | 6 | $10 + 5(8) + 2(6) = 10 + 40 + 12$ | **62** |
| **Priya** | 7 | 10 | $10 + 5(7) + 2(10) = 10 + 35 + 20$ | **65** |
| **Rahul** | 9 | 4 | $10 + 5(9) + 2(4) = 10 + 45 + 8$ | **63** |

* **Who is predicted to score highest?**  
  **Priya** ($\hat{y} = 65$). Even though her CGPA is lower than Rahul's, her high study hours (10 hrs) make up for it.
* **If a new student Maya (CGPA = 8.5, Study Hours = 8) joins, do we need to retrain the model?**  
  **No!** A trained model makes predictions instantly by plugging in values:  
  $\hat{y} = 10 + 5(8.5) + 2(8) = 10 + 42.5 + 16 = \mathbf{68.5}$.  
  Retraining is only needed when underlying patterns change (Model Drift).

---

## Section 3: Converting to Matrix Form — The "Bias Trick"

Writing $n$ separate algebraic equations is messy and inefficient. Can we pack all $n$ equations into **a single matrix multiplication**?

Look at each equation:
$$\hat{y}_i = \beta_0(1) + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_m x_{im}$$

Notice how $\beta_0$ has no feature next to it?  
We can imagine an invisible feature $x_{i0}$ that is **always equal to $1$** for every single data point: $\beta_0 \times 1$.

This brilliant mathematical technique is called the **"Bias Trick"** (or intercept augmentation):
We prepend an entire **column of 1s** to the left of our dataset!

$$\begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n \end{bmatrix} = \begin{bmatrix} 1 & x_{11} & x_{12} & \dots & x_{1m} \\ 1 & x_{21} & x_{22} & \dots & x_{2m} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \dots & x_{nm} \end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_m \end{bmatrix}$$

In compact matrix notation, this entire system reduces to just **three letters**:

$$\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\beta}$$

---

### Anatomy & Dimensions of the MLR Matrix Formulation

```
       y_hat           =           X                    *          β
   (Prediction)              (Design Matrix)                 (Parameters)
   
     [ n x 1 ]             [ n x (m + 1) ]                 [ (m + 1) x 1 ]
   
     +-------+             +---+---+---+---+                 +-------+
     | y_hat1|             | 1 |x11|x12|x1m|                 |  β0   |
     | y_hat2|     =       | 1 |x21|x22|x2m|        *        |  β1   |
     |  ...  |             |...|...|...|...|                 |  ...  |
     | y_hatn|             | 1 |xn1|xn2|xnm|                 |  βm   |
     +-------+             +---+---+---+---+                 +-------+
```

* **$\mathbf{X}$ (The Design Matrix):** Size $n \times (m + 1)$.  
  Has $n$ rows (samples) and $m + 1$ columns ($1$ column of all ones for the bias + $m$ feature columns).
* **$\boldsymbol{\beta}$ (The Parameter Vector):** Size $(m + 1) \times 1$.  
  Column vector containing the intercept $\beta_0$ and all $m$ slopes.
* **$\mathbf{\hat{y}}$ (The Prediction Vector):** Size $n \times 1$.  
  Column vector of predictions for all $n$ samples.

---

### 📝 Practice P3: Building the Design Matrix

Using the 3-student dataset from P2 ($\beta_0 = 10, \beta_1 = 5, \beta_2 = 2$):

1. **Construct the Design Matrix $\mathbf{X}$ and state its dimensions:**  
   $$\mathbf{X} = \begin{bmatrix} 1 & 8 & 6 \\ 1 & 7 & 10 \\ 1 & 9 & 4 \end{bmatrix} \quad \text{Dimensions: } \mathbf{3 \times 3} \quad (n = 3, m = 2 \implies m+1 = 3)$$

2. **Construct the Parameter Vector $\boldsymbol{\beta}$ and state its dimensions:**  
   $$\boldsymbol{\beta} = \begin{bmatrix} 10 \\ 5 \\ 2 \end{bmatrix} \quad \text{Dimensions: } \mathbf{3 \times 1}$$

3. **Verify by multiplying Row 1 of $\mathbf{X}$ by $\boldsymbol{\beta}$:**  
   $$\text{Row 1} \times \boldsymbol{\beta} = [1, 8, 6] \begin{bmatrix} 10 \\ 5 \\ 2 \end{bmatrix} = 1(10) + 8(5) + 6(2) = 10 + 40 + 12 = \mathbf{62}$$  
   *It matches Arjun's predicted score of 62 perfectly!*

4. **If a 4th student joins the dataset, what changes?**  
   * **Rows of $\mathbf{X}$:** Increases from 3 to 4 ($4 \times 3$).
   * **Columns of $\mathbf{X}$:** Unchanged (still 3 columns).
   * **Length of $\boldsymbol{\beta}$:** Unchanged (still $3 \times 1$).

---

## Section 4: Residuals & Comparing Two Candidate Models

Just as in simple linear regression, the **residual** for the $i$-th data point is the difference between actual output and model prediction:

$$e_i = y_i - \hat{y}_i$$

The **Total Squared Error (Loss)** across all $n$ points is:

$$L = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

---

### 📝 Practice P4: Comparing Two Models

Two candidate models predict monthly sales (in ₹ lakhs) across 3 retail stores:

| Store | Actual Sales ($y$) | Model A ($\hat{y}_A$) | Residual $e_A = y - \hat{y}_A$ | Model B ($\hat{y}_B$) | Residual $e_B = y - \hat{y}_B$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Store 1** | 50 | 48 | $50 - 48 = \mathbf{+2}$ | 52 | $50 - 52 = \mathbf{-2}$ |
| **Store 2** | 30 | 36 | $30 - 36 = \mathbf{-6}$ | 28 | $30 - 28 = \mathbf{+2}$ |
| **Store 3** | 45 | 44 | $45 - 44 = \mathbf{+1}$ | 46 | $45 - 46 = \mathbf{-1}$ |

1. **Model A's residual for Store 2 is $-6$ (negative). What does this mean?**  
   * The model predicted $36$ while actual was $30$. A negative residual means the model **over-predicted** sales.
2. **Compute Total Squared Error (TSE) for both models:**  
   $$\text{TSE}_A = (+2)^2 + (-6)^2 + (+1)^2 = 4 + 36 + 1 = \mathbf{41}$$  
   $$\text{TSE}_B = (-2)^2 + (+2)^2 + (-1)^2 = 4 + 4 + 1 = \mathbf{9}$$
3. **Which model fits better?**  
   **Model B** is significantly superior because its total squared error is much smaller ($9 < 41$).

---

## Section 5: The Vectorized Loss Function — Why $\mathbf{e}^T \mathbf{e}$?

Instead of calculating residuals row by row, let's stack all errors into a **Residual Vector $\mathbf{e}$**:

$$\mathbf{e} = \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_n \end{bmatrix} = \begin{bmatrix} y_1 - \hat{y}_1 \\ y_2 - \hat{y}_2 \\ \vdots \\ y_n - \hat{y}_n \end{bmatrix} = \mathbf{y} - \mathbf{\hat{y}}$$

Substituting our matrix model $\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\beta}$:

$$\mathbf{e} = \mathbf{y} - \mathbf{X}\boldsymbol{\beta}$$

### The Dot Product Magic: $\mathbf{e}^T \mathbf{e}$
Now, multiply the row vector $\mathbf{e}^T$ by the column vector $\mathbf{e}$:

$$\mathbf{e}^T \mathbf{e} = [e_1, e_2, \dots, e_n] \begin{bmatrix} e_1 \\ e_2 \\ \vdots \\ e_n \end{bmatrix} = e_1^2 + e_2^2 + \dots + e_n^2 = \sum_{i=1}^n e_i^2$$

> 🌟 **Key Insight:**  
> The single matrix operation **$\mathbf{e}^T \mathbf{e}$** automatically squares every error and sums them up!  
> We can write the entire loss function in one elegant matrix expression:
> 
> $$L(\boldsymbol{\beta}) = \mathbf{e}^T \mathbf{e} = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Notice that $L(\boldsymbol{\beta})$ takes a parameter vector $\boldsymbol{\beta} \in \mathbb{R}^{m+1}$ and outputs **a single scalar number** (the total squared error).

---

## Section 6: Expanding the Loss Function

To find the minimum of $L(\boldsymbol{\beta})$ using calculus, we must first expand the matrix product $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ term by term.

### Step 1: Apply the Transpose of a Difference
Recall the matrix transpose rule $(A - B)^T = A^T - B^T$, and the product rule $(AB)^T = B^T A^T$:

$$(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T = \mathbf{y}^T - (\mathbf{X}\boldsymbol{\beta})^T = \mathbf{y}^T - \boldsymbol{\beta}^T \mathbf{X}^T$$

### Step 2: Multiply the Two Brackets
$$L(\boldsymbol{\beta}) = (\mathbf{y}^T - \boldsymbol{\beta}^T \mathbf{X}^T)(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Using FOIL (First, Outer, Inner, Last):

$$L(\boldsymbol{\beta}) = \mathbf{y}^T \mathbf{y} - \mathbf{y}^T \mathbf{X}\boldsymbol{\beta} - \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta}$$

---

### Step 3: Checking Dimensions of Each Term
Let's verify that every single term is a $1 \times 1$ scalar:
* $\mathbf{y}^T \mathbf{y}$: $(1 \times n)(n \times 1) = \mathbf{1 \times 1}$ (scalar)
* $\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}$: $(1 \times n)(n \times [m+1])([m+1] \times 1) = \mathbf{1 \times 1}$ (scalar)
* $\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y}$: $(1 \times [m+1])([m+1] \times n)(n \times 1) = \mathbf{1 \times 1}$ (scalar)
* $\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta}$: $(1 \times [m+1])([m+1] \times [m+1])([m+1] \times 1) = \mathbf{1 \times 1}$ (scalar)

---

### Step 4: The Scalar Transpose Trick
Look at the two middle terms: $\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}$ and $\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y}$.  
Since both terms are scalars ($1 \times 1$ numbers), **the transpose of a scalar is equal to itself** (e.g., $7^T = 7$)!

Let's take the transpose of the second middle term:
$$(\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y})^T = \mathbf{y}^T (\mathbf{X}^T)^T (\boldsymbol{\beta}^T)^T = \mathbf{y}^T \mathbf{X}\boldsymbol{\beta}$$

They are identical numbers!  
Therefore, we can combine the two middle terms:

$$-\mathbf{y}^T \mathbf{X}\boldsymbol{\beta} - \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y} = -2\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}$$

---

### The Final Expanded Loss Function (Equation 1)
$$\mathbf{L(\boldsymbol{\beta}) = \underbrace{\mathbf{y}^T \mathbf{y}}_{\text{Constant}} - \underbrace{2\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}}_{\text{Linear in } \boldsymbol{\beta}} + \underbrace{\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta}}_{\text{Quadratic in } \boldsymbol{\beta}}}$$

Look at the structure of this equation! It is the exact matrix equivalent of a simple parabola $c - 2bx + ax^2$:
1. $\mathbf{y}^T \mathbf{y}$: Contains no $\boldsymbol{\beta}$ $\to$ **Constant term**.
2. $-2\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}$: Contains $\boldsymbol{\beta}$ to the first power $\to$ **Linear term**.
3. $\boldsymbol{\beta}^T (\mathbf{X}^T \mathbf{X})\boldsymbol{\beta}$: Contains $\boldsymbol{\beta}$ twice $\to$ **Quadratic term**.

---

### 📝 Practice P5: Verify the Expansion with Numbers

Let $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 1 & 3 \end{bmatrix}$, $\mathbf{y} = \begin{bmatrix} 5 \\ 7 \end{bmatrix}$, $\boldsymbol{\beta} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

1. **Compute $\mathbf{X}\boldsymbol{\beta}$ and $\mathbf{e} = \mathbf{y} - \mathbf{X}\boldsymbol{\beta}$:**  
   $$\mathbf{X}\boldsymbol{\beta} = \begin{bmatrix} 1(1) + 2(1) \\ 1(1) + 3(1) \end{bmatrix} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$$  
   $$\mathbf{e} = \mathbf{y} - \mathbf{X}\boldsymbol{\beta} = \begin{bmatrix} 5 - 3 \\ 7 - 4 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

2. **Compute Total Squared Error $\mathbf{e}^T \mathbf{e}$ directly:**  
   $$\mathbf{e}^T \mathbf{e} = [2, 3] \begin{bmatrix} 2 \\ 3 \end{bmatrix} = 2^2 + 3^2 = 4 + 9 = \mathbf{13}$$

3. **Compute each term of the expanded form separately:**
   * $\mathbf{y}^T \mathbf{y} = [5, 7] \begin{bmatrix} 5 \\ 7 \end{bmatrix} = 25 + 49 = \mathbf{74}$
   * $\mathbf{y}^T \mathbf{X}\boldsymbol{\beta} = [5, 7] \begin{bmatrix} 3 \\ 4 \end{bmatrix} = 15 + 28 = \mathbf{43}$
   * $\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y} = [3, 4] \begin{bmatrix} 5 \\ 7 \end{bmatrix} = 15 + 28 = \mathbf{43}$ *(Confirms middle terms are identical!)*
   * $\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = (\mathbf{X}\boldsymbol{\beta})^T (\mathbf{X}\boldsymbol{\beta}) = [3, 4] \begin{bmatrix} 3 \\ 4 \end{bmatrix} = 9 + 16 = \mathbf{25}$

4. **Combine the expanded terms:**  
   $$L = \mathbf{y}^T \mathbf{y} - 2(\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}) + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = 74 - 2(43) + 25 = 74 - 86 + 25 = \mathbf{13}$$  
   *It matches $\mathbf{e}^T \mathbf{e} = 13$ exactly!*

---

## Section 7 to 10: The Three Golden Matrix Calculus Identities

To minimize $L(\boldsymbol{\beta})$, we must take the derivative with respect to the vector $\boldsymbol{\beta}$ and set it to the zero vector:

$$\nabla_{\boldsymbol{\beta}} L = \frac{\partial L}{\partial \boldsymbol{\beta}} = \begin{bmatrix} \frac{\partial L}{\partial \beta_0} \\ \frac{\partial L}{\partial \beta_1} \\ \vdots \\ \frac{\partial L}{\partial \beta_m} \end{bmatrix} = \mathbf{0}$$

Because our expanded loss has 3 terms (constant, linear, and quadratic), we need **three fundamental matrix calculus identities**.

---

### Identity 1: Derivative of a Constant Term
> **Question:** If a scalar function $f(\boldsymbol{\beta}) = c$ never changes when $\boldsymbol{\beta}$ changes, what is its derivative?

$$\frac{\partial c}{\partial \boldsymbol{\beta}} = \mathbf{0} \quad \text{(the zero vector)}$$

Applied to our loss function: $\mathbf{y}^T \mathbf{y}$ contains only actual labels and zero $\boldsymbol{\beta}$ parameters:

$$\mathbf{\frac{\partial (\mathbf{y}^T \mathbf{y})}{\partial \boldsymbol{\beta}} = \mathbf{0}} \quad \text{--- (Identity 1)}$$

---

### Identity 2: Derivative of a Linear Form
> **Scalar Analogy:** $\frac{d}{dx}(ax) = a$.  
> What is the matrix version for a linear dot product $f(\boldsymbol{\beta}) = \mathbf{a}^T \boldsymbol{\beta}$?

Let $\mathbf{a} = [a_0, a_1, \dots, a_m]^T$.  
Expanding the dot product into scalar form:
$$f(\boldsymbol{\beta}) = \mathbf{a}^T \boldsymbol{\beta} = a_0 \beta_0 + a_1 \beta_1 + \dots + a_m \beta_m$$

Taking the partial derivative with respect to each component $\beta_j$:
$$\frac{\partial f}{\partial \beta_0} = a_0, \quad \frac{\partial f}{\partial \beta_1} = a_1, \quad \dots, \quad \frac{\partial f}{\partial \beta_m} = a_m$$

Stacking all partial derivatives back into a vector:

$$\mathbf{\frac{\partial (\mathbf{a}^T \boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = \mathbf{a}}$$

Applied to our linear loss term $-2\mathbf{y}^T \mathbf{X}\boldsymbol{\beta}$:  
Here, $\mathbf{a}^T = -2\mathbf{y}^T \mathbf{X}$, which means $\mathbf{a} = (-2\mathbf{y}^T \mathbf{X})^T = -2\mathbf{X}^T \mathbf{y}$.

$$\mathbf{\frac{\partial (-2\mathbf{y}^T \mathbf{X}\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T \mathbf{y}} \quad \text{--- (Identity 2)}$$

---

### Identity 3: Derivative of a Quadratic Form
> **Scalar Analogy:** $\frac{d}{dx}(ax^2) = 2ax$.  
> What is the derivative of the quadratic matrix form $f(\boldsymbol{\beta}) = \boldsymbol{\beta}^T \mathbf{A}\boldsymbol{\beta}$ where $\mathbf{A}$ is a symmetric matrix ($\mathbf{A}^T = \mathbf{A}$)?

#### 🔍 Complete $2 \times 2$ Proof:
Let $\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{12} & a_{22} \end{bmatrix}$ (symmetric, so $a_{21} = a_{12}$) and $\boldsymbol{\beta} = \begin{bmatrix} \beta_1 \\ \beta_2 \end{bmatrix}$.

* **Step 1 — Compute $\mathbf{A}\boldsymbol{\beta}$:**  
  $$\mathbf{A}\boldsymbol{\beta} = \begin{bmatrix} a_{11}\beta_1 + a_{12}\beta_2 \\ a_{12}\beta_1 + a_{22}\beta_2 \end{bmatrix}$$
* **Step 2 — Compute $\boldsymbol{\beta}^T (\mathbf{A}\boldsymbol{\beta})$:**  
  $$f(\boldsymbol{\beta}) = \beta_1(a_{11}\beta_1 + a_{12}\beta_2) + \beta_2(a_{12}\beta_1 + a_{22}\beta_2) = a_{11}\beta_1^2 + 2a_{12}\beta_1 \beta_2 + a_{22}\beta_2^2$$
* **Step 3 — Differentiate component-wise:**  
  $$\frac{\partial f}{\partial \beta_1} = 2a_{11}\beta_1 + 2a_{12}\beta_2 = 2(a_{11}\beta_1 + a_{12}\beta_2)$$  
  $$\frac{\partial f}{\partial \beta_2} = 2a_{12}\beta_1 + 2a_{22}\beta_2 = 2(a_{12}\beta_1 + a_{22}\beta_2)$$
* **Step 4 — Stack and factor out the 2:**  
  $$\frac{\partial f}{\partial \boldsymbol{\beta}} = \begin{bmatrix} \frac{\partial f}{\partial \beta_1} \\ \frac{\partial f}{\partial \beta_2} \end{bmatrix} = 2 \begin{bmatrix} a_{11}\beta_1 + a_{12}\beta_2 \\ a_{12}\beta_1 + a_{22}\beta_2 \end{bmatrix} = 2\mathbf{A}\boldsymbol{\beta}$$

General Result: For any symmetric matrix $\mathbf{A}$:

$$\mathbf{\frac{\partial (\boldsymbol{\beta}^T \mathbf{A}\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 2\mathbf{A}\boldsymbol{\beta}}$$

#### Applying Identity 3 to Our Loss Function:
Our quadratic term is $\boldsymbol{\beta}^T (\mathbf{X}^T \mathbf{X})\boldsymbol{\beta}$. Here, $\mathbf{A} = \mathbf{X}^T \mathbf{X}$.  
**Is $\mathbf{X}^T \mathbf{X}$ symmetric?**  
$$(\mathbf{X}^T \mathbf{X})^T = \mathbf{X}^T (\mathbf{X}^T)^T = \mathbf{X}^T \mathbf{X} \quad \text{✓ YES!}$$

Therefore:

$$\mathbf{\frac{\partial (\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 2\mathbf{X}^T \mathbf{X}\boldsymbol{\beta}} \quad \text{--- (Identity 3)}$$

---

## Section 11: The Grand Derivation of the Normal Equation

Now we assemble all three identities to differentiate our expanded loss function:

$$L(\boldsymbol{\beta}) = \mathbf{y}^T \mathbf{y} - 2\mathbf{y}^T \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta}$$

### Step 1: Differentiate Term by Term
$$\frac{\partial L}{\partial \boldsymbol{\beta}} = \mathbf{0} - 2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X}\boldsymbol{\beta}$$

$$\frac{\partial L}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X}\boldsymbol{\beta}$$

---

### Step 2: Set the Gradient to Zero for the Minimum
$$-2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$

Divide both sides by $2$:

$$-\mathbf{X}^T \mathbf{y} + \mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = \mathbf{0}$$

Move $\mathbf{X}^T \mathbf{y}$ to the right-hand side:

$$\mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$$

> 🏆 **THE NORMAL EQUATION:**  
> This historic formula is known worldwide as the **Normal Equation**.  
> It states the exact balance condition where the gradient of the error surface is zero.

---

### Step 3: Solve for $\boldsymbol{\beta}^*$ (The Closed-Form OLS Solution)
To isolate $\boldsymbol{\beta}^*$, multiply both sides from the left by the inverse matrix $(\mathbf{X}^T \mathbf{X})^{-1}$:

$$(\mathbf{X}^T \mathbf{X})^{-1} (\mathbf{X}^T \mathbf{X})\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Since $(\mathbf{X}^T \mathbf{X})^{-1} (\mathbf{X}^T \mathbf{X}) = \mathbf{I}$ (the identity matrix):

$$\mathbf{\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}}$$

### How to Read This Formula:
1. Compute $\mathbf{X}^T \mathbf{X}$ (a square $(m+1) \times (m+1)$ matrix).
2. Invert it: $(\mathbf{X}^T \mathbf{X})^{-1}$.
3. Compute $\mathbf{X}^T \mathbf{y}$ (an $(m+1) \times 1$ vector).
4. Multiply them together to get the optimal parameter vector $\boldsymbol{\beta}^*$ directly!

No guessing, no learning rate, no loops, no convergence checks!

---

## Section 12 & 13: When Does the Closed-Form Fail?

The OLS formula $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ looks deceptively simple. But in the real world, two major roadblocks can break it:

---

### Roadblock 1: Multicollinearity (Non-Invertible / Singular Matrix)
To compute $(\mathbf{X}^T \mathbf{X})^{-1}$, the matrix **must be invertible** ($\det(\mathbf{X}^T \mathbf{X}) \ne 0$).  
A matrix is non-invertible (singular) if its columns are **linearly dependent**.

> ⚠️ **The Multicollinearity Trap:**  
> Suppose an engineer collects three features:
> * $x_1$: Temperature in Celsius ($^\circ\text{C}$)
> * $x_2$: Temperature in Fahrenheit ($^\circ\text{F}$)
> * Since $^\circ\text{F} = 1.8(^\circ\text{C}) + 32$, Column 2 is an exact linear duplicate of Column 1!  
> 
> When two or more columns provide completely redundant information, $\det(\mathbf{X}^T \mathbf{X}) = 0$.  
> **The computer will crash with a `Singular Matrix: Non-invertible` error!**

* **Fixes:** Drop redundant features during EDA (Doc 1), use Variance Inflation Factor (VIF), or apply **Ridge Regression (L2 regularization)** which adds $\lambda \mathbf{I}$ to guarantee invertibility: $(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1}$.

---

### Roadblock 2: Computational Complexity ($O(m^3)$ Scaling)

Even if $(\mathbf{X}^T \mathbf{X})$ is invertible, is it always practical to invert?

$\mathbf{X}^T \mathbf{X}$ has size $(m+1) \times (m+1)$, where $m$ is the number of features.  
Using standard Gauss-Jordan elimination, inverting a $k \times k$ matrix requires **three nested loops**:
1. **Outer loop:** Iterate through all $k$ pivot columns $\to k$ iterations.
2. **Middle loop:** Eliminate all $k$ rows per pivot $\to k$ iterations.
3. **Inner loop:** Scale and update $\sim 2k$ entries in the augmented matrix $\to 2k$ operations.

$$\text{Total Operations} = k \times k \times 2k = 2k^3 \implies \mathbf{O(k^3)} = \mathbf{O(m^3)}$$

### Real-World Scaling Reality Check:

| Number of Features ($m$) | Matrix Inversion Cost ($m^3$) | Estimated Time on a 1 GFLOP/s Laptop |
| :---: | :---: | :---: |
| **$m = 10$** | $10^3 = 1,000$ operations | $0.000001$ seconds (Instant) |
| **$m = 100$** | $100^3 = 1,000,000$ operations | $0.001$ seconds (Blink of an eye) |
| **$m = 1,000$** | $1,000^3 = 10^9$ operations | $1$ second |
| **$m = 10,000$** | $(10^4)^3 = 10^{12}$ operations | $\sim 17$ minutes |
| **$m = 100,000$** | $(10^5)^3 = 10^{15}$ operations | **$\approx 11.6$ DAYS!** |

> 📌 **The Big Takeaway:**  
> * When $m$ is small ($m < 2,000$): **Always use the OLS closed-form solution** (it is fast and exact).  
> * When $m$ is massive ($m > 10,000$, e.g., image pixels or NLP text tokens): **OLS is completely impractical**. We must switch to **Gradient Descent** (Doc 5)!

---

## Section 14: Complete $3 \times 3$ Hand Calculation Walkthrough

Let's work through a full, step-by-step numerical example by hand!  
Suppose we have 3 students ($n = 3$) with 2 input features ($m = 2$):

| Student | Feature $x_1$ | Feature $x_2$ | Actual Output $y$ |
| :---: | :---: | :---: | :---: |
| **A** | 1 | 0 | 2 |
| **B** | 0 | 1 | 3 |
| **C** | 1 | 1 | 6 |

Let's compute $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ step by step!

---

### Step 1: Construct the Design Matrix $\mathbf{X}$ and Target Vector $\mathbf{y}$
Remember to include the column of 1s on the left for the intercept!

$$\mathbf{X} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 2 \\ 3 \\ 6 \end{bmatrix}$$

---

### Step 2: Compute the Transpose $\mathbf{X}^T$
Flip the rows and columns of $\mathbf{X}$:

$$\mathbf{X}^T = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}$$

---

### Step 3: Compute $\mathbf{X}^T \mathbf{X}$
Multiply the $(3 \times 3)$ matrix $\mathbf{X}^T$ by the $(3 \times 3)$ matrix $\mathbf{X}$:

$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 1 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

* Row 1 $\times$ Col 1: $1(1) + 1(1) + 1(1) = 3$
* Row 1 $\times$ Col 2: $1(1) + 1(0) + 1(1) = 2$
* Row 1 $\times$ Col 3: $1(0) + 1(1) + 1(1) = 2$
* Row 2 $\times$ Col 1: $1(1) + 0(1) + 1(1) = 2$
* Row 2 $\times$ Col 2: $1(1) + 0(0) + 1(1) = 2$
* Row 2 $\times$ Col 3: $1(0) + 0(1) + 1(1) = 1$
* Row 3 $\times$ Col 1: $0(1) + 1(1) + 1(1) = 2$
* Row 3 $\times$ Col 2: $0(1) + 1(0) + 1(1) = 1$
* Row 3 $\times$ Col 3: $0(0) + 1(1) + 1(1) = 2$

$$\mathbf{X}^T \mathbf{X} = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 2 & 1 \\ 2 & 1 & 2 \end{bmatrix} \quad \text{(Notice it is perfectly symmetric!)}$$

---

### Step 4: Compute $\mathbf{X}^T \mathbf{y}$
Multiply the $(3 \times 3)$ matrix $\mathbf{X}^T$ by the $(3 \times 1)$ vector $\mathbf{y}$:

$$\mathbf{X}^T \mathbf{y} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 6 \end{bmatrix} = \begin{bmatrix} 1(2) + 1(3) + 1(6) \\ 1(2) + 0(3) + 1(6) \\ 0(2) + 1(3) + 1(6) \end{bmatrix} = \begin{bmatrix} 2 + 3 + 6 \\ 2 + 0 + 6 \\ 0 + 3 + 6 \end{bmatrix} = \begin{bmatrix} \mathbf{11} \\ \mathbf{8} \\ \mathbf{9} \end{bmatrix}$$

---

### Step 5: Check Determinant $\det(\mathbf{X}^T \mathbf{X})$
$$\det(\mathbf{X}^T \mathbf{X}) = 3 \cdot \det\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} - 2 \cdot \det\begin{bmatrix} 2 & 1 \\ 2 & 2 \end{bmatrix} + 2 \cdot \det\begin{bmatrix} 2 & 2 \\ 2 & 1 \end{bmatrix}$$

$$\det(\mathbf{X}^T \mathbf{X}) = 3(4 - 1) - 2(4 - 2) + 2(2 - 4) = 3(3) - 2(2) + 2(-2) = 9 - 4 - 4 = \mathbf{1}$$

Since $\det = 1 \ne 0$, the matrix is **fully invertible**!

---

### Step 6: Invert $(\mathbf{X}^T \mathbf{X})^{-1}$
Using the adjugate matrix formula $A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$:

$$(\mathbf{X}^T \mathbf{X})^{-1} = \begin{bmatrix} 3 & -2 & -2 \\ -2 & 2 & 1 \\ -2 & 1 & 2 \end{bmatrix}$$

*(Verification: $\begin{bmatrix} 3 & 2 & 2 \\ 2 & 2 & 1 \\ 2 & 1 & 2 \end{bmatrix} \begin{bmatrix} 3 & -2 & -2 \\ -2 & 2 & 1 \\ -2 & 1 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = \mathbf{I}$).*

---

### Step 7: Multiply to Find Optimal Parameters $\boldsymbol{\beta}^*$
$$\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} = \begin{bmatrix} 3 & -2 & -2 \\ -2 & 2 & 1 \\ -2 & 1 & 2 \end{bmatrix} \begin{bmatrix} 11 \\ 8 \\ 9 \end{bmatrix}$$

* $\beta_0 = 3(11) - 2(8) - 2(9) = 33 - 16 - 18 = \mathbf{-1}$
* $\beta_1 = -2(11) + 2(8) + 1(9) = -22 + 16 + 9 = \mathbf{+3}$
* $\beta_2 = -2(11) + 1(8) + 2(9) = -22 + 8 + 18 = \mathbf{+4}$

$$\boldsymbol{\beta}^* = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix} = \begin{bmatrix} \mathbf{-1} \\ \mathbf{3} \\ \mathbf{4} \end{bmatrix}$$

---

### Step 8: Write the Final Model
$$\mathbf{\hat{y} = -1 + 3x_1 + 4x_2}$$

---

### Step 9: Verify Against the Actual Dataset
Let's plug each student's feature values back into our fitted model:
* **Student A ($x_1 = 1, x_2 = 0$):**  
  $\hat{y} = -1 + 3(1) + 4(0) = -1 + 3 + 0 = \mathbf{2}$ *(Actual $y = 2$ ✓ Perfect match!)*
* **Student B ($x_1 = 0, x_2 = 1$):**  
  $\hat{y} = -1 + 3(0) + 4(1) = -1 + 0 + 4 = \mathbf{3}$ *(Actual $y = 3$ ✓ Perfect match!)*
* **Student C ($x_1 = 1, x_2 = 1$):**  
  $\hat{y} = -1 + 3(1) + 4(1) = -1 + 3 + 4 = \mathbf{6}$ *(Actual $y = 6$ ✓ Perfect match!)*

---

### 🔍 Reflection Questions:

#### Reflect 1: Why did our model achieve a 100% perfect fit?
In this example, we had **$n = 3$ data points** and **$3$ learnable parameters** ($\beta_0, \beta_1, \beta_2$).  
In linear algebra, a system of 3 independent linear equations with 3 unknowns has **a unique exact solution**!  
* When $n = m + 1$, the hyperplane interpolates all training points exactly ($e_i = 0$ for all $i$).
* But in real-world ML with $n = 100,000$ points and $3$ parameters, real data will have noise and scatter—a straight plane cannot touch all 100,000 points simultaneously.
* **Warning:** If $n \le m+1$ in practice, fitting all points perfectly is often a sign of extreme **Overfitting**, not good learning!

#### Reflect 2: Why would anyone use Gradient Descent if OLS gives the exact formula in one shot?
Because when $m = 20,000$, computing $(\mathbf{X}^T \mathbf{X})^{-1}$ takes trillion of operations and runs out of RAM. Gradient Descent takes simple, inexpensive linear steps ($O(m)$ per step) that scale beautifully to massive datasets.

---

## Section 15: Side-by-Side Comparison — SLR vs. MLR

| Concept | Simple Linear Regression (Doc 3) | Multiple Linear Regression (Doc 4) |
| :--- | :---: | :---: |
| **Number of Features** | $m = 1$ | $m > 1$ (arbitrary $m$) |
| **Parameters** | 2 parameters: slope $m$, intercept $c$ | $m + 1$ parameters: vector $\boldsymbol{\beta} = [\beta_0, \dots, \beta_m]^T$ |
| **Model Form** | $\hat{y} = mx + c$ | $\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\beta}$ |
| **Geometric Representation** | 1D line in 2D space | $m$-dimensional hyperplane in $(m+1)$D space |
| **Loss Function** | $\sum_{i=1}^n (y_i - mx_i - c)^2$ | $\mathbf{e}^T \mathbf{e} = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ |
| **Optimality Condition** | $\frac{\partial L}{\partial m} = 0, \quad \frac{\partial L}{\partial c} = 0$ | $\nabla_{\boldsymbol{\beta}} L = \frac{\partial L}{\partial \boldsymbol{\beta}} = \mathbf{0}$ |
| **Closed-Form Formula** | $m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad c = \bar{y} - m\bar{x}$ | $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ |
| **Invertibility Condition** | $\sum (x_i - \bar{x})^2 \ne 0$ (feature variance $> 0$) | $\det(\mathbf{X}^T \mathbf{X}) \ne 0$ (linearly independent columns) |
| **Failure Mode** | All $x_i$ identical (zero variance) | Multicollinearity (redundant features) |
| **Computational Cost** | $O(n)$ operations | $O(n m^2 + m^3)$ operations |

---

## 📝 Practice P11: True / False Exam Rapid Fire

| Statement | Answer | Rationale |
| :--- | :---: | :--- |
| **(a) SLR is a special case of MLR with $m = 1$.** | **True** | When $m = 1$, $\mathbf{X}$ has 2 columns (1s and $x$), matching the line equation. |
| **(b) If you double every value in column $x_2$, the OLS formula still produces a valid $\boldsymbol{\beta}^*$.** | **True** | Doubling $x_2$ simply halves its coefficient $\beta_2$; the fitted predictions $\mathbf{\hat{y}}$ remain identical. |
| **(c) OLS minimizes the sum of absolute residuals.** | **False** | OLS minimizes the sum of **squared** residuals ($\sum e_i^2$). |
| **(d) If $\mathbf{X}$ has 200 rows and 5 features, then $\mathbf{X}^T \mathbf{X}$ is a $6 \times 6$ matrix.** | **True** | $\mathbf{X}$ has $m + 1 = 6$ columns (1 bias + 5 features). $\mathbf{X}^T \mathbf{X}$ size is $(m+1) \times (m+1) = 6 \times 6$. |
| **(e) Multicollinearity means two features are slightly correlated.** | **False** | Multicollinearity refers to **strong or exact linear dependence** between features causing numerical singularity. |
| **(f) If Gradient Descent and OLS are both applicable, they will find different optimal $\boldsymbol{\beta}^*$ values.** | **False** | Because the loss surface is strictly convex (a single global bowl), converged GD and OLS reach the **exact same unique minimum**. |
| **(g) Adding more features to a model always strictly reduces the Total Squared Error on training data.** | **False** | Training TSE cannot increase, but if the new feature has a coefficient of 0, the error may stay exactly the same. |
| **(h) The Normal Equation is $\mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$.** | **True** | This is the exact stationary condition where the gradient equals $\mathbf{0}$. |
| **(i) If $n < m + 1$, the model can fit training data perfectly, and this is always a good thing.** | **False** | While it can interpolate the training points, this is extreme **overfitting** and will generalize poorly to new test data. |

---

## 🚀 Forward Handoff to Next Lecture (Lecture 5)

Now that we understand the power and limitations of the analytical closed-form OLS solution:
* In **Lecture 5 (Batch Gradient Descent for MLR)**, we dive into the iterative optimization engine that powers modern deep learning!
* We will derive cost gradients with respect to weights and bias step by step, choose learning rates $\alpha$, and watch the algorithm iteratively walk down the 3D loss bowl to find the minimum without ever inverting a matrix!

---

## 📋 Summary of Sheet 4 (Key Takeaways)

1. **The Model:** $\mathbf{\hat{y}} = \mathbf{X}\boldsymbol{\beta}$ extends linear regression to hyperplanes in $(m+1)$-dimensional space.
2. **The Bias Trick:** Prepending a column of 1s to $\mathbf{X}$ absorbs the intercept $\beta_0$ directly into matrix multiplication.
3. **The Loss as a Dot Product:** $L(\boldsymbol{\beta}) = \mathbf{e}^T \mathbf{e} = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$.
4. **Three Matrix Calculus Identities:**
   * $\frac{\partial c}{\partial \boldsymbol{\beta}} = \mathbf{0}$ (Constant)
   * $\frac{\partial (\mathbf{a}^T \boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = \mathbf{a}$ (Linear form)
   * $\frac{\partial (\boldsymbol{\beta}^T \mathbf{A}\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 2\mathbf{A}\boldsymbol{\beta}$ for symmetric $\mathbf{A}$ (Quadratic form)
5. **The Normal Equation:** $\mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$.
6. **The Closed-Form Solution:** $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$.
7. **The Inversion Bottleneck:** Inverting $(m+1) \times (m+1)$ costs $O(m^3)$, making analytical OLS impractical for massive feature sets.
