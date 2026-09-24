# Machine Learning Made Simple — Doc 6: Stochastic & Mini-Batch Gradient Descent
**Worksheet:** [Worksheet 06](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_06_Stochastic_and_MiniBatch_Gradient_Descent.pdf)  
**Lab Assignment:** [Lab 06 MiniBatch SGD MLR](file:///Users/mayanksharma/Downloads/AML/04_Notebooks/Lab_06_MiniBatch_SGD_MLR/solved/Lab_6_Final_BGD_MLR_student_todo.ipynb)  
**Interactive Visualizer:** [Module 06 Gradient Descent Visualizer](file:///Users/mayanksharma/Downloads/AML/03_HTML_Visualizations/Module_06_Gradient_Descent_Interactive_Lecture.html)  
**Topics:** The Waiting Problem of Batch GD · 5 Real-World Limitations of BGD · Stochastic Gradient Descent (SGD) · Shuffling and Order Randomization · Mathematical Derivations of Sample Gradients · Visualizing the Parameter Plane ($\theta_0 - \theta_1$) · Complete Step-by-Step Dry Run (BGD vs. SGD vs. Mini-Batch) · Pseudocode Comparison · Why Gradient Noise Helps Non-Convex Losses · The Goldilocks Compromise: Mini-Batch Gradient Descent · GPU Parallelism & Powers of 2 · Uneven Mini-Batch Splits · Comprehensive 3-Way Comparison Table

---

## 🗺️ Where Are We in the Journey?

In our previous documents:
* [Doc 3](file:///Users/mayanksharma/Downloads/AML/docs/03_Simple_Linear_Regression_OLS.md): Simple Linear Regression with 1 feature.
* [Doc 4](file:///Users/mayanksharma/Downloads/AML/docs/04_Multiple_Linear_Regression_OLS.md): Multiple Linear Regression and the closed-form Normal Equation: $\boldsymbol{\beta}^* = (\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T \mathbf{y}$.
* [Doc 5](file:///Users/mayanksharma/Downloads/AML/docs/05_Batch_Gradient_Descent_MLR.md): Batch Gradient Descent (BGD), taking iterative downhill steps using the entire dataset: $\boldsymbol{\theta}_{\text{new}} = \boldsymbol{\theta}_{\text{old}} - \alpha \frac{2}{n}\mathbf{X}^T(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$.

In Doc 5, we solved the feature-scaling bottleneck ($O(m^3)$) of OLS. But Batch Gradient Descent introduces a massive new problem:
**What if our dataset has 10,000,000 rows?**

In Batch GD, you must scan all 10 million rows, calculate 10 million errors, average them together, and **take only ONE single step**! If a single pass over your dataset takes 30 minutes, taking 1,000 gradient steps would take 3 weeks!

Now in **Doc 6**, we solve the Big Data bottleneck:
1. **Stochastic Gradient Descent (SGD):** Update parameters after **every single row**!
2. **Mini-Batch Gradient Descent (MBGD):** The industry standard—update parameters after **small groups (batches of 32, 64, or 128 samples)**!

---

## Section 1: The Waiting Problem — Why Batch GD Is Not Enough

### The Rhythm of Batch Gradient Descent:
$$\text{Scan all } n \text{ samples} \longrightarrow \text{Average their gradients} \longrightarrow \text{Take ONE parameter step}$$

If your dataset has $n = 10,000,000$ rows:
* **How many parameter updates happen in one epoch?**  
  **Exactly ONE update!**
* **Is it worth reading 10 million rows just to take one tiny step?**  
  **Absolutely not!**

---

### The 5 Real-World Roadblocks of Batch GD:

| Limitation | Why It Happens | When It Becomes Serious | Real-World Industry Example |
| :--- | :--- | :--- | :--- |
| **1. The Waiting Problem** | Averages over all $n$ samples before moving. | A single dataset pass takes hours. | **Ad-click prediction:** 100 million click logs per day. |
| **2. High Cost per Step** | Needs gradients for every single row. | Huge sample size ($n$) and features ($d$). | **Image Processing:** High-resolution satellite images. |
| **3. Memory & I/O Choke** | Data must be loaded into RAM repeatedly. | Dataset is larger than computer RAM. | **E-Commerce logs:** 500 GB recommender system files. |
| **4. Slow Feedback Loop** | The model learns nothing until the end of the pass. | The real world changes quickly in real time. | **Fraud Detection:** Credit card theft happening right now. |
| **5. Redundant Work** | Similar data points yield nearly identical gradients. | Repetitive patterns in big data. | **Search engines:** Millions of identical queries for "weather". |

All 5 roadblocks have the same root cause: **Batch GD demands a 100% exact full-dataset gradient before it takes even one step.**  
Exactness is great for small datasets, but for Big Data, waiting for perfection is a waste of time.

---

### 📝 Practice P1: Scenario Matching

Match each real-world scenario to the primary limitation it creates for Batch Gradient Descent:
1. *A recommendation model trained on 500 GB of user-item click logs.* $\longrightarrow$ **B. Memory and I/O Pressure** (data exceeds RAM).
2. *A fraud detection system that needs to adapt immediately to incoming transactions.* $\longrightarrow$ **A. Slow Feedback** (cannot wait hours for a full pass).
3. *A dataset containing 10 million identical ad impressions.* $\longrightarrow$ **C. Redundant Work** (calculating the same gradient 10 million times).

---

## Section 2: Stochastic Gradient Descent (SGD) — The Instant Learner

### 1. The Classroom Polling Analogy
> **Batch GD:** Like polling **all 1,000 students** in an auditorium before taking a single step forward. It gives a very accurate consensus, but it takes 2 hours just to move one foot!  
> **Stochastic GD (SGD):** Like asking **one random student** for their opinion, taking a step immediately, asking a second random student, taking another step immediately, and repeating!

A single student's opinion might be slightly noisy or biased, but because they are picked randomly, their steps average out toward the true downhill direction—and **you move immediately without waiting**!

---

### 2. Mathematical Definition of SGD
Let $J_i(\boldsymbol{\theta})$ be the loss calculated on **a single training sample $i$**.  
At iteration $t$:
1. Randomly select one sample index $i_t$ uniformly from $\{1, 2, \dots, n\}$:
   $$i_t \sim \text{Uniform}\{1, 2, \dots, n\}$$
2. Compute the gradient using **only that single sample**:
   $$\mathbf{g}_t = \nabla J_{i_t}(\boldsymbol{\theta}_t)$$
3. Immediately update the parameter vector:
   $$\mathbf{\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \mathbf{g}_t}$$

---

### 3. Updates per Epoch: The Dramatic Difference!
* **Batch GD:** 1 epoch with $n = 1,000$ samples $\longrightarrow$ **1 parameter update**.
* **Stochastic GD:** 1 epoch with $n = 1,000$ samples $\longrightarrow$ **1,000 parameter updates**!

Over $E$ epochs, SGD performs **$E \times n$ updates**, compared to Batch GD's mere $E$ updates. Learning starts on Millisecond 1!

---

### ⚠️ WARNING: Why Shuffling Is Mandatory in SGD!

Real-world datasets are almost always ordered (e.g., sorted by date, customer ID, or class label: all fraud transactions grouped at the bottom).  
If you do not shuffle:
* Consecutive updates will chase the local ordering pattern instead of the global objective!
* The weights will drift off in a biased direction.

> 📌 **Golden SGD Rule:**  
> **Always shuffle your dataset at the beginning of every single epoch!**

---

### Common Misconceptions Corrected:

| Misconception | The Reality |
| :--- | :--- |
| *"SGD trains on only 1 sample forever."* | **False!** One *update* uses 1 sample, but one *epoch* uses every single sample in the dataset once! |
| *"The loss must decrease after every single update."* | **False!** Because an update follows a single noisy sample, the total loss can temporarily tick upward before resuming its downward trend. |
| *"Stochastic means random parameters."* | **False!** The update rule is strict calculus. The only randomness is the *order* in which samples are selected. |

---

## Section 3: Mathematical Comparison — Batch GD vs. SGD

```
                 Batch GD (Smooth & Steady)             Stochastic GD (Noisy & Fast)
                 
                 Optimum                                Optimum
                    ^                                      ^
                    |      /                              |    \   /\   /
                    |     /                               |     \_/  \_/
                    |    /                                |      \   /
                    |   /                                 |       \_/
                    +----------> θ0                       +----------> θ0
```

| Aspect | Batch Gradient Descent (BGD) | Stochastic Gradient Descent (SGD) |
| :--- | :--- | :--- |
| **Gradient Evaluated** | Exact average over all $n$ samples. | Exact gradient of **one randomly picked sample**. |
| **Update Timing** | After the entire dataset is processed. | **Immediately** after every single sample. |
| **Updates per Epoch** | **$1$ update** | **$n$ updates** |
| **Computation per Update** | $O(n \cdot d)$ operations | $O(d)$ operations |
| **Memory Requirement** | High (must access all $n$ rows per step). | **Lowest** (streams 1 sample at a time into RAM). |
| **Optimization Path** | Smooth, direct, monotonic descent. | **Noisy zig-zag**, jumping around the mean gradient. |
| **Outlier Sensitivity** | Low (outliers diluted across $n$ samples). | **High** (an outlier causes an erratic jump). |

---

### 💡 Why Noise Can Actually Be a Superpower!
In simple Linear Regression, the loss surface is a convex bowl (one global minimum), so noise is a minor nuisance.  
However, in **Deep Neural Networks**, the loss surface is non-convex with thousands of **shallow local minimum traps and flat saddle points**:
* Batch GD can get stuck forever on a flat plateau where the gradient is zero ($\nabla L = \mathbf{0}$).
* SGD's wild sampling noise kicks the parameters out of shallow traps, helping the optimizer discover much deeper, better minima!

---

## Section 4: Visualizing the Parameter Plane ($\theta_0 - \theta_1$)

If you plot the trajectory of parameters $(\theta_0, \theta_1)$ toward the optimal solution:
* **Batch GD:** Moves in a series of smooth, confident arrows straight toward the bottom of the bowl.
* **Stochastic GD:** Resembles a drunk person walking home—zig-zagging left and right, overshooting and correcting, but **steadily trending toward the center**!

```
         θ1 ^
            |          (Batch GD: Direct, smooth path)
            |          • ----------> • ----------> • (Optimum)
            |                                     ^
            |          /\    /\    /\            /
            |         /  \  /  \  /  \__________/
            |        /    \/    \/
            |       • (SGD: Noisy zig-zag path, but fast!)
            +---------------------------------------------> θ0
```

---

## Section 5: Complete Dry Run — One Epoch of BGD vs. SGD

Let's trace **every single number** by hand on a toy linear regression dataset of 4 points ($n = 4$):
* Input $x = [1, 2, 3, 4]$
* Target $y = [2, 4, 6, 8]$  
*(Notice that $y = 2x$ exactly, so the true parameters are $\theta_0 = 0$ and $\theta_1 = 2$).*

* **Initial Weights:** $\boldsymbol{\theta}_{\text{old}} = \begin{bmatrix} \theta_0 \\ \theta_1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$
* **Learning Rate:** $\alpha = 0.1$
* **Sample Loss Function:** $J_i(\boldsymbol{\theta}) = \frac{1}{2}(\hat{y}_i - y_i)^2 = \frac{1}{2}e_i^2$
* **Sample Gradient:**
  $$\nabla J_i = \begin{bmatrix} \frac{\partial J_i}{\partial \theta_0} \\ \frac{\partial J_i}{\partial \theta_1} \end{bmatrix} = \begin{bmatrix} e_i \\ e_i x_i \end{bmatrix} = \begin{bmatrix} \hat{y}_i - y_i \\ (\hat{y}_i - y_i)x_i \end{bmatrix}$$

---

### 1. Batch GD (One Full Epoch)
In Batch GD, all 4 samples are evaluated using the **initial weights $\boldsymbol{\theta} = [0, 0]^T$**:

| $i$ | $x_i$ | $y_i$ | Prediction $\hat{y}_i = 0 + 0(x_i)$ | Error $e_i = \hat{y}_i - y_i$ | $\text{Grad } \theta_0 = e_i$ | $\text{Grad } \theta_1 = e_i x_i$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 2 | 0 | $-2$ | $-2$ | $(-2)(1) = -2$ |
| **2** | 2 | 4 | 0 | $-4$ | $-4$ | $(-4)(2) = -8$ |
| **3** | 3 | 6 | 0 | $-6$ | $-6$ | $(-6)(3) = -18$ |
| **4** | 4 | 8 | 0 | $-8$ | $-8$ | $(-8)(4) = -32$ |
| **AVG**| — | — | — | — | **$-5$** | **$-15$** |

$$\text{Average Gradient } \nabla L = \begin{bmatrix} -5 \\ -15 \end{bmatrix}$$

**The Single Batch GD Update:**
$$\boldsymbol{\theta}_{\text{new}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} -5 \\ -15 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.5 \\ 1.5 \end{bmatrix}} \quad \text{(Total Updates: 1)}$$

---

### 2. Stochastic GD (One Full Epoch)
Now, shuffle the samples to visit them in random order: **Sample 2 $\to$ Sample 4 $\to$ Sample 1 $\to$ Sample 3**.  
Notice how $\boldsymbol{\theta}$ updates **after every single sample**!

#### Step 1: Visit Sample 2 ($x = 2, y = 4$) with $\boldsymbol{\theta} = [0, 0]^T$
* $\hat{y} = 0 + 0(2) = 0$
* $e = 0 - 4 = -4$
* Gradient: $\nabla J = \begin{bmatrix} -4 \\ (-4)(2) \end{bmatrix} = \begin{bmatrix} -4 \\ -8 \end{bmatrix}$
* Update: $\boldsymbol{\theta} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} -4 \\ -8 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.400 \\ 0.800 \end{bmatrix}}$

#### Step 2: Visit Sample 4 ($x = 4, y = 8$) with new $\boldsymbol{\theta} = [0.4, 0.8]^T$
* $\hat{y} = 0.4 + 0.8(4) = 0.4 + 3.2 = 3.6$
* $e = 3.6 - 8 = -4.4$
* Gradient: $\nabla J = \begin{bmatrix} -4.4 \\ (-4.4)(4) \end{bmatrix} = \begin{bmatrix} -4.4 \\ -17.6 \end{bmatrix}$
* Update: $\boldsymbol{\theta} = \begin{bmatrix} 0.4 \\ 0.8 \end{bmatrix} - 0.1 \begin{bmatrix} -4.4 \\ -17.6 \end{bmatrix} = \begin{bmatrix} 0.4 + 0.44 \\ 0.8 + 1.76 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.840 \\ 2.560 \end{bmatrix}}$

#### Step 3: Visit Sample 1 ($x = 1, y = 2$) with new $\boldsymbol{\theta} = [0.84, 2.56]^T$
* $\hat{y} = 0.84 + 2.56(1) = 3.4$
* $e = 3.4 - 2 = \mathbf{+1.4}$ *(Error flipped to positive! The model overshot!)*
* Gradient: $\nabla J = \begin{bmatrix} +1.4 \\ (+1.4)(1) \end{bmatrix} = \begin{bmatrix} +1.4 \\ +1.4 \end{bmatrix}$
* Update: $\boldsymbol{\theta} = \begin{bmatrix} 0.84 \\ 2.56 \end{bmatrix} - 0.1 \begin{bmatrix} 1.4 \\ 1.4 \end{bmatrix} = \begin{bmatrix} 0.84 - 0.14 \\ 2.56 - 0.14 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.700 \\ 2.420 \end{bmatrix}}$

#### Step 4: Visit Sample 3 ($x = 3, y = 6$) with new $\boldsymbol{\theta} = [0.70, 2.42]^T$ (Practice P3)
* $\hat{y} = 0.70 + 2.42(3) = 0.70 + 7.26 = 7.96$
* $e = 7.96 - 6 = \mathbf{+1.96}$
* Gradient: $\nabla J = \begin{bmatrix} +1.96 \\ (+1.96)(3) \end{bmatrix} = \begin{bmatrix} +1.96 \\ +5.88 \end{bmatrix}$
* Update: $\boldsymbol{\theta} = \begin{bmatrix} 0.70 \\ 2.42 \end{bmatrix} - 0.1 \begin{bmatrix} 1.96 \\ 5.88 \end{bmatrix} = \begin{bmatrix} 0.70 - 0.196 \\ 2.42 - 0.588 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.504 \\ 1.832 \end{bmatrix}}$

---

### The Epoch 1 Showdown:
* **Batch GD after 1 Epoch (1 update):** $\boldsymbol{\theta} = [0.500, 1.500]^T$
* **Stochastic GD after 1 Epoch (4 updates):** $\boldsymbol{\theta} = [0.504, 1.832]^T$

Notice that the true slope is $\theta_1 = 2.0$.  
After just 1 single epoch, **SGD got significantly closer to the true slope ($1.832$ vs $1.500$)** because it adapted its direction 4 times instead of waiting!

> 💡 **Why did the error sign flip in Step 3?**  
> In Step 2, the aggressive update pushed $\theta_1$ up to $2.560$ (above the true value of $2.0$).  
> When the model evaluated Sample 1, it predicted $3.4$ instead of $2$, producing a **positive residual (+1.4)**.  
> This positive residual caused the next gradient to be positive, **automatically pulling $\theta_1$ back downward** ($2.56 \to 2.42 \to 1.83$). This self-correcting dynamic is the magic of SGD!

---

## Section 6: Pseudocode Comparison — Spot the Difference

```python
# ==============================================================
# ALGORITHM 1: BATCH GRADIENT DESCENT
# ==============================================================
for epoch in range(E):
    grad = 0
    for i in range(m):                 # Loop through ALL samples
        grad += compute_gradient(x[i], y[i], theta)
    grad = grad / m                    # Average over the FULL dataset
    theta = theta - alpha * grad       # ONE UPDATE PER EPOCH

# ==============================================================
# ALGORITHM 2: STOCHASTIC GRADIENT DESCENT (SGD)
# ==============================================================
for epoch in range(E):
    shuffle(dataset)                   # 1. Shuffle at start of epoch
    for i in range(m):                 # 2. Loop through samples
        grad = compute_gradient(x[i], y[i], theta)
        theta = theta - alpha * grad   # UPDATE IMMEDIATELY PER SAMPLE!
```

> ⚠️ **The Critical Coding Mistake:**  
> In SGD, **never accumulate gradients into a running sum inside the sample loop**. If you accumulate gradients before updating, you have accidentally rewritten Batch GD!

---

## Section 7 & 8: The Goldilocks Compromise — Mini-Batch Gradient Descent

We now see the fundamental tradeoff:
* **Batch GD:** Gradient is 100% accurate, but taking a step is **too slow**.
* **Stochastic GD:** Taking a step is instant, but the gradient is **too noisy and bounces wildly**.

Can we get the best of both worlds?  
**YES: Mini-Batch Gradient Descent!**

```
+---------------------+-------------------------------+---------------------+
|      Batch GD       |    Mini-Batch Gradient Descent|         SGD         |
|  (Batch Size = n)   |     (Batch Size = 32 to 256)  |   (Batch Size = 1)  |
|   Too Slow / Rigid  |      THE PERFECT SWEET SPOT   |    Too Noisy / Wild |
+---------------------+-------------------------------+---------------------+
```

Instead of asking all 1,000 students or just 1 student, we ask a **small focus group of $b$ students** (e.g., $b = 32$ or $64$):
* A group of 32 cancels out individual sample noise.
* Computing 32 samples takes milliseconds and fits easily into computer cache memory.

---

### Why Are Batch Sizes Always Powers of 2 ($32, 64, 128, 256$)?
There is no mathematical law requiring powers of 2.  
However, **GPU hardware architectures (NVIDIA Tensor Cores and CUDA Warps)** process data in chunks of 32 threads. Using batch sizes like $32, 64, 128$, or $256$ aligns memory perfectly with physical GPU silicon, running up to **$5\times$ faster** than arbitrary numbers like $50$ or $100$!

---

## Section 9: Complete Mini-Batch GD Dry Run ($b = 2$)

Using our same 4-point dataset ($x = [1, 2, 3, 4], y = [2, 4, 6, 8]$), let's execute Mini-Batch GD with batch size $b = 2$.  
Shuffled order: $\{ \text{Sample 2, Sample 4} \}$ (Batch 1) and $\{ \text{Sample 1, Sample 3} \}$ (Batch 2).

### Iteration 1: Process Mini-Batch 1 = {Sample 2, Sample 4}
Start with $\boldsymbol{\theta}^{(0)} = [0, 0]^T$.
* Gradient for Sample 2: $\mathbf{g}_2 = \begin{bmatrix} -4 \\ -8 \end{bmatrix}$
* Gradient for Sample 4: $\mathbf{g}_4 = \begin{bmatrix} -8 \\ -32 \end{bmatrix}$
* **Average Gradient for Batch 1:**
  $$\bar{\mathbf{g}}_1 = \frac{\mathbf{g}_2 + \mathbf{g}_4}{2} = \frac{1}{2} \begin{bmatrix} -4 + (-8) \\ -8 + (-32) \end{bmatrix} = \begin{bmatrix} -6 \\ -20 \end{bmatrix}$$
* **Update 1:**
  $$\boldsymbol{\theta}^{(1)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - 0.1 \begin{bmatrix} -6 \\ -20 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.600 \\ 2.000 \end{bmatrix}}$$

---

### Iteration 2: Process Mini-Batch 2 = {Sample 1, Sample 3}
Now evaluate Batch 2 using our updated weights $\boldsymbol{\theta}^{(1)} = [0.6, 2.0]^T$:
* **Sample 1 ($x = 1, y = 2$):**  
  $\hat{y}_1 = 0.6 + 2.0(1) = 2.6 \implies e_1 = 2.6 - 2 = +0.6$  
  $\mathbf{g}_1 = \begin{bmatrix} +0.6 \\ (+0.6)(1) \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.6 \end{bmatrix}$
* **Sample 3 ($x = 3, y = 6$):**  
  $\hat{y}_3 = 0.6 + 2.0(3) = 6.6 \implies e_3 = 6.6 - 6 = +0.6$  
  $\mathbf{g}_3 = \begin{bmatrix} +0.6 \\ (+0.6)(3) \end{bmatrix} = \begin{bmatrix} 0.6 \\ 1.8 \end{bmatrix}$
* **Average Gradient for Batch 2:**
  $$\bar{\mathbf{g}}_2 = \frac{\mathbf{g}_1 + \mathbf{g}_3}{2} = \frac{1}{2} \begin{bmatrix} 0.6 + 0.6 \\ 0.6 + 1.8 \end{bmatrix} = \begin{bmatrix} \mathbf{0.6} \\ \mathbf{1.2} \end{bmatrix}$$
* **Update 2 (Final parameters after Epoch 1):**
  $$\boldsymbol{\theta}^{(2)} = \begin{bmatrix} 0.6 \\ 2.0 \end{bmatrix} - 0.1 \begin{bmatrix} 0.6 \\ 1.2 \end{bmatrix} = \begin{bmatrix} 0.6 - 0.06 \\ 2.0 - 0.12 \end{bmatrix} = \mathbf{\begin{bmatrix} 0.540 \\ 1.880 \end{bmatrix}}$$

Look at the result: $\boldsymbol{\theta} = [0.54, 1.88]^T$!  
It converged significantly faster than Batch GD ($[0.5, 1.5]$), but without the wild oscillations of pure SGD.

---

### 📝 Uneven Mini-Batch Splits ($b = 3$)
What if dataset size $n$ is not perfectly divisible by batch size $b$? (e.g., $n = 4$ and $b = 3$).  
* Total mini-batches per epoch = $\lceil n / b \rceil = \lceil 4 / 3 \rceil = \mathbf{2 \text{ batches}}$.
* **Batch 1:** Contains 3 samples $\{S2, S4, S1\}$ $\longrightarrow$ Average across 3 gradients.
* **Batch 2:** Contains the 1 remaining sample $\{S3\}$ $\longrightarrow$ Average across 1 gradient.

---

## Section 10: The Master Comparison Table (BGD vs. SGD vs. MBGD)

| Feature | Batch Gradient Descent | Stochastic Gradient Descent (SGD) | Mini-Batch Gradient Descent |
| :--- | :---: | :---: | :---: |
| **Batch Size ($b$)** | $b = n$ (Full dataset) | $b = 1$ (Single sample) | $b = 32, 64, 128, 256$ |
| **Updates per Epoch** | **$1$ update** | **$n$ updates** | **$\lceil n / b \rceil$ updates** |
| **Gradient Accuracy** | Exact true gradient | High sampling noise | **Low noise, stable average** |
| **Memory Consumption** | High (loads entire dataset) | **Lowest** (streams 1 sample) | **Low to Moderate** |
| **Hardware Efficiency** | Good, but memory-bound | Poor (underutilizes GPU cores) | **Maximum GPU Tensor Parallelism** |
| **Convergence Path** | Direct and smooth | Erratic zig-zag | Smooth with slight healthy noise |
| **Can Escape Saddle Points?**| ❌ No (gets stuck) | ✅ Yes (noise knocks it out) | ✅ **Yes** |
| **Industry Adoption** | Educational baselines | Online real-time streaming | **The standard in Deep Learning & LLMs** |

---

## 📝 Practice P7: True / False Exam Rapid Fire

| Statement | Answer | Rationale |
| :--- | :---: | :--- |
| **(a) The loss is guaranteed to decrease after every single SGD update.** | **False** | Because an update follows a single noisy sample, the loss on the overall dataset can temporarily rise. |
| **(b) Mini-Batch GD with a batch size of $b = 1$ is mathematically identical to SGD.** | **True** | Averaging over a batch of size 1 is identical to evaluating a single sample. |
| **(c) If $n = 1,000$ and $b = 100$, one epoch of Mini-Batch GD will perform 1,000 updates.** | **False** | Number of updates = $n / b = 1,000 / 100 = \mathbf{10 \text{ updates}}$ per epoch. |

---

## 🚀 Forward Handoff to Next Lecture (Lecture 7)

In Lectures 5 and 6, we mastered how to optimize regression models using first-order gradient descent.  
However, notice a fundamental limitation:
* Gradient descent takes steps using only the **current slope**.
* In steep ravines, it oscillates wastefully from wall to wall rather than accelerating down the center of the valley!

In **Lecture 7 (Advanced Optimization & Evaluation Metrics)**:
* We introduce **Momentum** and **Nesterov Accelerated Gradient (NAG)**, which give our optimizer physical inertia to rocket down flat valleys!
* We explore the comprehensive suite of regression metrics: **MAE, MAPE, MSE, RMSE, $R^2$, and Adjusted $R^2$**.

---

## 📋 Summary of Sheet 6 (Key Takeaways)

1. **The Waiting Problem:** Batch GD computes only 1 update per epoch, making it impossibly slow for massive datasets.
2. **SGD ($b = 1$):** Updates weights immediately after every single sample ($n$ updates per epoch). Learning begins instantly.
3. **Mandatory Shuffling:** Randomizing sample order before every epoch protects SGD from biased dataset trends.
4. **Mini-Batch GD ($b \in [32, 256]$):** The optimal balance. Averages small groups of samples to slash gradient noise while updating frequently.
5. **GPU Hardware Synergy:** Batch sizes as powers of 2 align with physical GPU warps, maximizing vector parallel throughput.
