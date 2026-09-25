# Machine Learning Made Simple — Doc 7: Regression & Classification Evaluation Metrics
**Worksheet:** [Worksheet 07](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_07_Regression_and_Classification_Evaluation_Metrics.pdf)  
**Lab Assignment:** [Lab 07 Student Performance and Placement](file:///Users/mayanksharma/Downloads/AML/04_Notebooks/Lab_07_Student_Performance_and_Placement/solved/lab_7_St_c.ipynb)  
**Topics:** Training Loss vs. Evaluation Metrics · Signed Residuals · Error-Magnitude Metrics (MAE, MAPE, MSE, RMSE) · $R^2$ Score from the Mean-Only Baseline (TSS, RSS) · Why $R^2$ Is Not "Accuracy" · Overfitting & The Adjusted $R^2$ Penalty · The Bridge to Classification · Why Accuracy Fails on Imbalanced Data · The Confusion Matrix (Scikit-Learn Standard) · Precision, Recall / Sensitivity, and the F1 Score · The Precision-Recall Trade-off · Multiclass Aggregation (Macro, Weighted, Micro Averages) · Decision-Driven Metric Selection Matrix

---

## 🗺️ Where Are We in the Journey?

In our previous documents:
* [Doc 3](file:///Users/mayanksharma/Downloads/AML/docs/03_Simple_Linear_Regression_OLS.md) & [Doc 4](file:///Users/mayanksharma/Downloads/AML/docs/04_Multiple_Linear_Regression_OLS.md): Closed-form Ordinary Least Squares regression for single and multiple features.
* [Doc 5](file:///Users/mayanksharma/Downloads/AML/docs/05_Batch_Gradient_Descent_MLR.md) & [Doc 6](file:///Users/mayanksharma/Downloads/AML/docs/06_Stochastic_and_MiniBatch_Gradient_Descent.md): First-order optimization engines (Batch GD, Stochastic GD, and Mini-Batch GD).

We have built models, computed gradients, and converged to optimal weights.  
**Now comes the critical question: How do we know if our trained model is actually good?**

A model is not "good" just because a computer finished training it. It is good only if its predictions are acceptable for the real-world decision we need to make!
* If a self-driving car misses a stop sign by 1 millisecond, people could die.
* If a spam filter misplaces 1 marketing email out of 1,000, nobody cares.

In this document, we master the complete diagnostic toolkit for judging machine learning models:
1. **Part 1 (Regression Metrics):** Residuals, MAE, MAPE, MSE, RMSE, $R^2$, and Adjusted $R^2$.
2. **Part 2 (Classification Metrics):** The Confusion Matrix, Precision, Recall, F1 Score, and Macro/Weighted/Micro averaging.

---

# PART 1: REGRESSION EVALUATION METRICS

## Section 1: Training Loss vs. Evaluation Metric (The Key Difference)

Before writing any math, you must understand the difference between how a model **learns** and how a model is **judged**:

```
+------------------------------------+------------------------------------+
|        TRAINING LOSS (The Coach)   |     EVALUATION METRIC (The Judge)  |
+------------------------------------+------------------------------------+
| • Evaluated on TRAINING data.      | • Evaluated on TEST / VALIDATION.  |
| • Used during training loops.      | • Used AFTER training has ended.   |
| • Must be smooth & differentiable  | • Does NOT need to be calculus-    |
|   so gradients can be computed.    |   friendly (can use |e|, %, etc.). |
| • Guides weight updates.           | • Answers: "Is this model safe to  |
|                                    |   deploy to customers?"            |
+------------------------------------+------------------------------------+
```

> 🌟 **Key Insight:**  
> The exact same mathematical expression can play both roles!  
> For example, **Mean Squared Error (MSE)** is a *Training Loss* when gradient descent uses its derivative to update weights. But it becomes an *Evaluation Metric* when reported on a held-out test set to compare two competing models. **Context determines the role.**

---

## Section 2: Master Test Dataset & Signed Residuals

Let's evaluate two models (**Model A** and **Model B**) predicting the monthly internship stipend (in thousands of INR, ₹k) for 8 university students ($n = 8$):

The **Signed Residual** for student $i$ under Model A is:

$$e_{A,i} = y_i - \hat{y}_{A,i}$$

* If $e > 0$: The actual stipend was higher than predicted (the model **under-predicted**).
* If $e < 0$: The actual stipend was lower than predicted (the model **over-predicted**).

### Master Test Dataset R:

| Student ID | Actual Stipend ($y_i$) | Model A ($\hat{y}_{A,i}$) | Model B ($\hat{y}_{B,i}$) | Residual ($e_{A,i}$) | Absolute ($|e_{A,i}|$) | Squared ($e_{A,i}^2$) | APE (%) | Squared B ($e_{B,i}^2$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 30 | 32 | 32 | $-2$ | 2 | 4 | $6.67\%$ | 4 |
| **S2** | 35 | 34 | 34 | $+1$ | 1 | 1 | $2.86\%$ | 1 |
| **S3** | 40 | 43 | 43 | $-3$ | 3 | 9 | $7.50\%$ | 9 |
| **S4** | 45 | 42 | 43 | $+3$ | 3 | 9 | $6.67\%$ | 4 |
| **S5** | 50 | 49 | 49 | $+1$ | 1 | 1 | $2.00\%$ | 1 |
| **S6** | 55 | 58 | 58 | $-3$ | 3 | 9 | $5.45\%$ | 9 |
| **S7** | 60 | 56 | 56 | $+4$ | 4 | 16 | $6.67\%$ | 16 |
| **S8** | 65 | 70 | 69 | $-5$ | 5 | 25 | $7.69\%$ | 16 |
| **TOTAL** | — | — | — | **$-4$** | **$22$** | **$74$** | **$45.50\%$** | **$60$** |

*Notice that the sum of signed residuals is $-4$. Positive and negative residuals cancel each other out, hiding the true error magnitude!*

---

## Section 3: The 4 Regression Error Metrics

How do we summarize all 8 errors into a meaningful score?

```
                     +---------------------------------------+
                     |    How Should We Measure Error Size?  |
                     +-------------------+-------------------+
                                         |
            +----------------------------+----------------------------+
            |                                                         |
            v                                                         v
   [ In Original Target Units ]                              [ Relative / Penalized ]
   • MAE:  Equal linear penalty (₹2.75k)                     • MAPE: Percentage error (5.69%)
   • RMSE: Penalizes large mistakes (₹3.04k)                 • MSE:  Squared penalty (9.25 k^2)
```

---

### 1. Mean Absolute Error (MAE)
> **Question Answered:** *"On average, what is the typical absolute miss in original units?"*

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |e_i| = \frac{22}{8} = \mathbf{2.75 \text{ thousand INR}}$$

* **Advantages:** Beautifully intuitive to explain to business stakeholders. Every single unit of error receives an equal, linear penalty.
* **Disadvantages:** Treats a dangerous large error of 10 exactly the same as ten small errors of 1.

---

### 2. Mean Absolute Percentage Error (MAPE)
> **Question Answered:** *"On average, what percentage was the model off relative to the true value?"*

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left| \frac{e_i}{y_i} \right| = \frac{45.50\%}{8} \approx \mathbf{5.69\%}$$

* **Advantages:** Completely **scale-free**! A 5% error is equally meaningful whether predicting ₹50,000 stipends or ₹5,000,000 houses.
* **⚠️ Major Pitfall:** **Undefined when $y_i = 0$** (division by zero) and explodes toward infinity when actual values are tiny!

---

### 3. Mean Squared Error (MSE)
> **Question Answered:** *"What is the average squared error?"*

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n e_i^2 = \frac{74}{8} = \mathbf{9.25 \text{ (thousand INR)}^2}$$

* **Advantages:** Quadratically penalizes large mistakes. Smooth parabola ideal for mathematical optimization (gradient descent).
* **Disadvantages:** The units are **squared** ($\text{INR}^2$). Nobody knows what a "squared rupee" is!

---

### 4. Root Mean Squared Error (RMSE)
> **Question Answered:** *"What is the standard deviation of the unexplained residuals?"*

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{9.25} \approx \mathbf{3.04 \text{ thousand INR}}$$

* **Advantages:** Solves MSE's unit problem by taking the square root—returning the penalty directly to **original target units (₹k)**!
* **Key Comparison:** Notice that $\text{RMSE} = 3.04 > \text{MAE} = 2.75$.  
  RMSE is always $\ge \text{MAE}$. When a dataset contains severe outliers, RMSE pulls significantly higher than MAE.

---

### 📝 Practice P2: Concept Check

1. If an individual error jumps from $1$ to $5$:
   * Its contribution to $\sum |e_i|$ rises by $5 - 1 = \mathbf{4}$.
   * Its contribution to $\sum e_i^2$ rises by $5^2 - 1^2 = 25 - 1 = \mathbf{24}$!  
   * **Which metric family reacts more severely?** **MSE / RMSE**.
2. **True or False:** *"An MSE of 9.25 means the model typically misses predictions by ₹9.25k."*  
   * **Answer:** **FALSE!** MSE is in squared units ($\text{INR}^2$). The comparable original-unit summary is $\text{RMSE} \approx \mathbf{3.04\text{k INR}}$.

---

## Section 4: The $R^2$ Score (The Mean-Only Baseline Comparison)

MAE and RMSE tell us the raw size of errors. But they don't answer:  
**"Is our fancy machine learning model actually better than a lazy dummy guess?"**

---

### Step 1: The Mean-Only Baseline ($\bar{y}$)
Suppose a lazy intern refuses to write ML code. When asked to predict any student's stipend, they simply guess the **sample average** of all historical stipends:

$$\bar{y} = \frac{30 + 35 + 40 + 45 + 50 + 55 + 60 + 65}{8} = \mathbf{47.5 \text{ thousand INR}}$$

The intern predicts ₹47.5k for every single student, ignoring CGPA and all other features!

---

### Step 2: Total Sum of Squares (TSS)
The **Total Sum of Squares** measures the total raw variance that exists in the actual data around the mean baseline:

$$\text{TSS} = \sum_{i=1}^n (y_i - \bar{y})^2$$

* For S1 ($30$): $(30 - 47.5)^2 = (-17.5)^2 = 306.25$
* For S2 ($35$): $(35 - 47.5)^2 = (-12.5)^2 = 156.25$
* For S3 ($40$): $(40 - 47.5)^2 = (-7.5)^2 = 56.25$
* For S4 ($45$): $(45 - 47.5)^2 = (-2.5)^2 = 6.25$
* For S5 ($50$): $(50 - 47.5)^2 = (+2.5)^2 = 6.25$
* For S6 ($55$): $(55 - 47.5)^2 = (+7.5)^2 = 56.25$
* For S7 ($60$): $(60 - 47.5)^2 = (+12.5)^2 = 156.25$
* For S8 ($65$): $(65 - 47.5)^2 = (+17.5)^2 = 306.25$

$$\mathbf{\text{TSS} = 306.25 \times 2 + 156.25 \times 2 + 56.25 \times 2 + 6.25 \times 2 = 1050}$$

If we use the lazy mean-only baseline, **all 1,050 units of variance remain completely unexplained**.

---

### Step 3: Residual Sum of Squares (RSS)
The **Residual Sum of Squares** measures how much squared error is left over after applying **Model A**:

$$\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_{A,i})^2 = \mathbf{74}$$

---

### Step 4: Compute the $R^2$ Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{74}{1050} = 1 - 0.0705 = \mathbf{0.9295 \quad (92.95\%)}$$

$$\text{Explained Variance} = \frac{\text{TSS} - \text{RSS}}{\text{TSS}} = \frac{1050 - 74}{1050} = \frac{976}{1050} = \mathbf{0.9295}$$

> 🌟 **The True Meaning of $R^2$:**  
> **Model A successfully explains 92.95% of the variance** that existed around the mean-only baseline. Only 7.05% remains unexplained noise!

---

### Key $R^2$ Landmarks:

| $R^2$ Value | Meaning |
| :---: | :--- |
| **$R^2 = 1.0$** | **Perfect Fit:** $\text{RSS} = 0$. The model predicts every single point with zero error. |
| **$0 < R^2 < 1$** | **Good Model:** Beats the mean baseline, explaining a fraction of the data's variance. |
| **$R^2 = 0.0$** | **Useless Model:** Exactly as accurate as simply guessing the sample mean for everyone ($\text{RSS} = \text{TSS}$). |
| **$R^2 < 0.0$** | **Horrendous Model:** Predicting the mean would be better! (Happens when testing an overfitted model on unseen test data). |

> ⚠️ **CRITICAL MISCONCEPTION: $R^2$ IS NOT "ACCURACY"!**  
> Beginners often say: *"My model has an $R^2$ of 0.93, so it is 93% accurate and has 7% error."*  
> **This is completely wrong!** $R^2$ is not an accuracy percentage. It is a ratio of variances. Always report MAE or RMSE to describe actual error size!

---

## Section 5: The $R^2$ Trap & Adjusted $R^2$

### The Trap: Why Raw $R^2$ Always Increases
Suppose you add a completely useless 3rd feature to Model A: **the last digit of each student's phone number**.  
In ordinary least squares, adding any extra column to matrix $\mathbf{X}$ gives the model more degrees of freedom to memorize random noise in the training set:
* $\text{RSS}$ will slightly decrease (or stay the same).
* **Raw $R^2$ will ALWAYS increase or stay the same!**

Raw $R^2$ rewards complexity and cannot penalize overfitting!

---

### The Solution: Adjusted $R^2$

Adjusted $R^2$ imposes a mathematical penalty for every additional predictor added to the model:

$$\mathbf{R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}}$$

* **$n$:** Number of observations (sample size).
* **$p$:** Number of features / predictors (**excluding the intercept**).
* **$\frac{n - 1}{n - p - 1}$:** The **Penalty Factor**. As you add more features ($p$ increases), the denominator shrinks, making the penalty factor grow larger!

For a new feature to raise Adjusted $R^2$, it must reduce unexplained variance enough to overcome the increased complexity penalty!

---

### Worked Showdown: Did the Extra Feature Earn Its Place?

* **Model A:** Uses 2 features (`CGPA` + `Interview_Score`). $n = 8, p = 2, \text{RSS} = 74$.
* **Model B:** Uses Model A's features + `Roll_Number_Last_Digit` (noise!). $n = 8, p = 3, \text{RSS} = 60$.

Let's compute both:

$$R^2_{\text{adj}, A} = 1 - \left(\frac{74}{1050}\right) \frac{8 - 1}{8 - 2 - 1} = 1 - (0.07048) \frac{7}{5} = 1 - 0.09867 = \mathbf{0.9013}$$

$$R^2_{\text{adj}, B} = 1 - \left(\frac{60}{1050}\right) \frac{8 - 1}{8 - 3 - 1} = 1 - (0.05714) \frac{7}{4} = 1 - 0.10000 = \mathbf{0.9000}$$

| Model | Predictors ($p$) | $\text{RSS}$ | Raw $R^2$ | Adjusted $R^2$ | Did Feature Earn Place? |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | 2 | 74 | $0.9295$ | **$0.9013$** | **WINNER!** |
| **Model B** | 3 | 60 | **$0.9429$** | $0.9000$ | ❌ Rejected (Noise feature!) |

> 📌 **The Decision:**  
> Even though Model B had a higher raw $R^2$ ($0.9429 > 0.9295$), **Model A has the higher Adjusted $R^2$ ($0.9013 > 0.9000$)**!  
> The roll number feature did not reduce error enough to justify its extra parameter. **Reject Model B!**

---

# PART 2: CLASSIFICATION EVALUATION METRICS

## Section 6: From Error Size to Error Identity

In regression, if actual stipend is ₹55k and the model predicts ₹50k, the error is numerical: $-5\text{k}$.  
In classification, models predict categorical labels:
* "What is the numerical distance between **'Ready'** and **'Needs Practice'**?"

There is no numerical distance! **Classification errors have identity:**
* *Which exact class was confused with which other class?*

---

## Section 7: Multiclass Accuracy & The Confusion Matrix

A university placement team uses a classifier to predict whether 18 students are ready for industry internships:
* **Ready (R):** 8 students
* **Needs Practice (P):** 6 students
* **Needs Support (S):** 4 students (Critical minority class requiring academic intervention)

### Master Test Dataset C (18 Students):
* Exact matches: **12 students**
* Wrong predictions: **6 students**

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{12}{18} = \mathbf{66.7\%}$$

---

### ⚠️ Why Accuracy Is a Dangerous Trap (The Imbalance Problem)

Accuracy tells us that the model was correct 66.7% of the time.  
**What does accuracy hide?**
* Did the model confuse a capable student as needing help?
* Or did it tell a struggling student they are ready, letting them fail their internship interview?
* Accuracy counts all mistakes equally as $-1$. It cannot show which classes were confused.

---

### The Confusion Matrix (Scikit-Learn Standard)
To see the full picture, we build a **Confusion Matrix**.  
* **Rows = Actual True Classes**
* **Columns = Model Predictions**

```
                          PREDICTED CLASSES
                     Pred Ready   Pred Practice   Pred Support   Row Total (Actuals)
                   +------------+---------------+--------------+
  Actual Ready     |     6      |       1       |      1       |        8
                   +------------+---------------+--------------+
  Actual Practice  |     1      |       4       |      1       |        6
                   +------------+---------------+--------------+
  Actual Support   |     0      |       2       |      2       |        4
                   +------------+---------------+--------------+
  Column Total     |     7      |       7       |      4       |       18
```

### Anatomy of the Matrix:
* **The Main Diagonal ($6 + 4 + 2 = 12$):** Exact correct matches!
* **Off-Diagonal Cells ($1 + 1 + 1 + 1 + 0 + 2 = 6$):** Specific classification errors!
  * Cell (Actual Ready, Pred Support) $= 1$: One ready student was wrongly flagged as needing support.
  * Cell (Actual Support, Pred Practice) $= 2$: **Two struggling students were missed completely!**

---

## Section 8: Precision, Recall, and the F1 Score

Let's zoom in on the most operationally critical class: **Needs Support (S)**.  
We frame this as a binary one-vs-rest problem:
* **Positive Class ($+$):** "Needs Support"
* **Negative Class ($- $):** "Not Support" (Ready or Needs Practice)

```
                              PREDICTED
                      Pred Support      Pred Not Support
                   +-----------------+---------------------+
  Actual Support   | True Positive   | False Negative      |
                   | (TP = 2)        | (FN = 2) [MISSED!]  |
                   +-----------------+---------------------+
  Actual Not Supp  | False Positive  | True Negative       |
                   | (FP = 2) [ALARM]| (TN = 12)           |
                   +-----------------+---------------------+
```

* **TP = 2:** Struggling students correctly identified for help.
* **FN = 2:** Struggling students who needed help but were **missed**!
* **FP = 2:** Capable students who were given a **false alarm**.
* **TN = 12:** Capable students correctly recognized as not needing support.

---

### 1. Precision (Quality of Alerts)
> **Question:** *"Out of all students we flagged as needing support, how many actually needed it?"*

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{2}{2 + 2} = \frac{2}{4} = \mathbf{0.500 \quad (50.0\%)}$$

* **When to prioritize Precision:** When a **False Positive (False Alarm)** is extremely costly.  
  *(Example: Spam filtering—marking an urgent job offer email as spam is unacceptable).*

---

### 2. Recall / Sensitivity / True Positive Rate (Catch Rate)
> **Question:** *"Out of all students who truly needed support, how many did our model find?"*

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{2}{2 + 2} = \frac{2}{4} = \mathbf{0.500 \quad (50.0\%)}$$

* **When to prioritize Recall:** When a **False Negative (Missed Detection)** is catastrophic!  
  *(Example: Cancer screening or airport bomb scanners—missing a positive case is fatal; a false alarm can easily be cleared with a second test).*

---

### 3. F1 Score (Harmonic Balance)
> **Question:** *"What is the balanced score between Precision and Recall?"*

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}} = \frac{2(2)}{2(2) + 2 + 2} = \frac{4}{8} = \mathbf{0.500 \quad (50.0\%)}$$

* **Why Harmonic Mean?** If Precision is $1.0$ and Recall is $0.0$, an arithmetic average would misleadingly give $0.50$. The Harmonic Mean punishes extreme imbalances, crashing to **$0.0$**!

---

### 💡 The Big Reveal: Accuracy Was Lying!
Look at the numbers side by side:
* **Overall Model Accuracy:** **$66.7\%$**
* **Needs Support Recall:** **$50.0\%$**

The model missed **half of all students needing academic intervention**!  
A high overall accuracy score can easily mask complete failure on small, critical minority classes.

---

## Section 9: Multiclass Averaging (Macro vs. Weighted vs. Micro)

When evaluating multiclass problems, how do we combine individual class scores into a single overall summary?

### The Per-Class Breakdown:

| Class | True Support Count ($n$) | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Ready** | 8 | $0.857$ | $0.750$ | $0.800$ |
| **Needs Practice** | 6 | $0.571$ | $0.667$ | $0.615$ |
| **Needs Support** | 4 | $0.500$ | $0.500$ | $0.500$ |

---

### 1. Macro Average (Equal Vote for Every Class)
Treats all classes as equally important, regardless of how many samples they have:

$$\text{Macro F1} = \frac{0.800 + 0.615 + 0.500}{3} = \mathbf{0.638}$$

* **When to use:** When minority classes are just as important as majority classes (e.g., fraud detection or rare disease diagnosis).

---

### 2. Weighted Average (Weighted by Class Size)
Weights each class by its actual representation in the dataset ($8/18, 6/18, 4/18$):

$$\text{Weighted F1} = \frac{8(0.800) + 6(0.615) + 4(0.500)}{18} = \frac{6.40 + 3.69 + 2.00}{18} = \mathbf{0.672}$$

* **Why is Weighted F1 higher than Macro F1 here?**  
  Because our weakest class (Needs Support, F1 = 0.500) only has 4 students. Weighting gives it less influence, pulling the average upward toward the strong "Ready" class!

---

### 3. Micro Average (Global Pooling)
Pools all class decisions globally: $\text{Total TP} = 12$, $\text{Total FP} = 6$, $\text{Total FN} = 6$.

$$\text{Micro F1} = \frac{2(12)}{2(12) + 6 + 6} = \frac{24}{36} = \mathbf{0.667}$$

> 📌 **Golden Identity:**  
> In single-label multiclass classification, **Micro Precision = Micro Recall = Micro F1 = Overall Accuracy**!

---

## Section 10: Decision-Driven Metric Selection Matrix

Never choose a metric because it gives the highest number! **Always choose the metric that reflects the cost of real-world mistakes.**

| Real-World Decision Need | The Winning Metric to Choose |
| :--- | :---: |
| Typical regression error in original units with equal linear penalty | **MAE** |
| Large regression misses are dangerous and must be heavily penalized | **RMSE / MSE** |
| Comparing regression models across different currencies or scales | **MAPE** |
| Checking if an ML model explains variance better than guessing the mean | **$R^2$** |
| Checking if adding a new feature earned its place or just overfit | **Adjusted $R^2$** |
| All classification mistakes have equal cost and classes are balanced | **Accuracy** |
| Inspecting exact confusion patterns between classes | **Confusion Matrix** |
| False alarms are costly (Spam filter, criminal conviction) | **Precision** |
| Missed positive cases are catastrophic (Cancer screening, defect detection) | **Recall / Sensitivity** |
| Balancing false alarms and missed cases on imbalanced data | **F1 Score** |
| Evaluating multiclass models where rare minority classes are critical | **Macro Average F1** |
| Evaluating multiclass models where overall volume throughput matters | **Weighted Average F1** |

---

## 🚀 Forward Handoff to Next Lecture (Lecture 8)

Now that we know how to properly evaluate models:
* What happens when the true relationship between $x$ and $y$ is not a straight line, but a curve (e.g., biological growth curves or economic cycles)?
* In **Lecture 8 (Polynomial Regression & OLS Assumptions)**, we learn how to fit complex curves using linear regression by expanding features into higher-order polynomials ($x^2, x^3$), and investigate the **5 Classic Assumptions of Linear Regression**!

---

## 📋 Summary of Sheet 7 (Key Takeaways)

1. **Loss vs. Metric:** Loss trains the model; metrics evaluate the finished model on unseen test data.
2. **Regression Metrics:**
   * **MAE:** Easy interpretation in original units.
   * **RMSE:** Returns to original units while penalizing large blunders quadratically.
   * **$R^2$:** Percentage of variance explained compared to guessing the mean ($\bar{y}$).
   * **Adjusted $R^2$:** Penalizes useless features, preventing complexity bloat.
3. **Classification Metrics:**
   * **Accuracy:** Misleading on imbalanced datasets.
   * **Confusion Matrix:** Shows which specific class was confused with another.
   * **Precision:** Accuracy of positive alerts ($\frac{\text{TP}}{\text{TP} + \text{FP}}$).
   * **Recall:** Coverage of actual positive cases ($\frac{\text{TP}}{\text{TP} + \text{FN}}$).
   * **F1 Score:** Harmonic mean balancing Precision and Recall.
4. **Multiclass Aggregation:** Macro treats all classes equally; Weighted reflects population size; Micro equals Accuracy.
