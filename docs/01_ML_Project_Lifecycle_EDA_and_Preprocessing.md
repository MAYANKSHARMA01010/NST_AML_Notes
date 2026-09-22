# Advanced Machine Learning — Worksheet 01: The ML Project Lifecycle (Part 1)
**Topics:** Problem Definition · Data Collection · Exploratory Data Analysis (EDA) · Data Preprocessing  
**Reference File:** [Worksheet_01_ML_Project_Lifecycle_EDA_and_Preprocessing.pdf](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_01_ML_Project_Lifecycle_EDA_and_Preprocessing.pdf)

---

## 📌 Executive Summary & Lifecycle Architecture

Machine learning is not merely selecting an algorithm and calling `.fit()`. In production systems, success depends on systematic execution of the lifecycle. Worksheet 01 focuses on the first 4 foundational phases:

1. **Problem Definition (The Foundation)**: Translating messy business pain points into measurable mathematical objectives.
2. **Data Collection (Sourcing the Fuel)**: Understanding sources, schema rigidity, and feature data types.
3. **Exploratory Data Analysis (The Diagnostic Stage)**: Diagnosing missing values, outliers, duplicates, and distributions before touching models.
4. **Data Preprocessing (The Engineering Work)**: Transforming messy real-world data into clean numerical matrices via scaling and encoding.

```
+---------------------+     +--------------------+     +-------------------+     +-----------------------+
| 1. Define Problem   | --> | 2. Collect Data    | --> | 3. Exploratory    | --> | 4. Preprocessing      |
| Vague -> Quantified |     | Internal/APIs/Logs |     | Data Analysis     |     | Clean, Scale, Encode  |
+---------------------+     +--------------------+     +-------------------+     +-----------------------+
```

---

## Part 1: Defining the Problem (The Foundation)

### The Hook & Core Trap
> **Scenario:** A music streaming company says: *"Users are leaving! Build an AI to fix this!"*  
> If an engineer immediately opens a laptop and starts training a linear regression model, they are guaranteed to fail. Why? Because an algorithm cannot optimize vague human statements like "fix churn." It requires a quantified loss function and a defined target variable.

### The 5-Step Blueprint
Before writing any code, translate human problems into a mathematical framework:

| Step | Action | Practical Example |
| :--- | :--- | :--- |
| **1. Understand the Problem** | Isolate the exact business friction point. | Users canceling premium subscriptions = churn. |
| **2. Set Quantifiable Goals** | Define a concrete, measurable numerical target. | "Reduce churn by 10% over two quarters." |
| **3. Assess ML Feasibility** | Ask: "Do we need ML? Can a simpler rule solve it?" | If a basic SQL query or rule engine solves it, **skip ML**. |
| **4. Identify Constraints** | Pinpoint hardware, latency, budget, and legal limits. | Real-time predictions in <10 ms on a smartphone; GDPR privacy rules. |
| **5. Stakeholder Alignment** | Ensure PMs, developers, and executives share one goal. | Mutual sign-off on what "success" mathematically means. |

### Blueprint in Action
* ❌ **Vague:** "We want to improve our online retail store."
* ✅ **Rigorous:** "Using historical clickstreams and purchase data, predict which users will stop shopping within 3 months."

---

### Practice & Critical Thinking (Part 1)

#### Practice P1: Rewriting Vague Requests into Rigorous ML Problem Statements
* **Prompt (a):** *"Our hospital needs better patient care."*  
  * **Rigorous ML Statement:** "Using patient vitals, lab results, and admission records from the past 5 years, predict which ICU patients are at risk of readmission within 30 days, targeting a 15% reduction in preventable readmissions."
* **Prompt (b):** *"Make our email system smarter."*  
  * **Rigorous ML Statement:** "Using email metadata (sender, subject, timestamps) and user-labelled spam/non-spam data, classify incoming emails as spam or not-spam with ≥98% precision and ≥95% recall."

#### Reflect 1: Data Access vs. Constraints
* **Question:** A company has pristine customer data, but data privacy regulations strictly forbid using it for model training. Which phase of the blueprint encounters this problem?
* **Answer:** **Phase 4: Identify Constraints**. Even when data physically exists, legal, compliance, and governance restrictions (e.g., GDPR, HIPAA) represent hard operational constraints that must be audited before building models.

> **Key Insight:** A poorly defined problem wastes more time than a bad algorithm. You cannot optimize "make it better"; the algorithm requires an explicit numerical target to minimize.  
> **Takeaway:** Never touch code until you have a quantified target, confirmed data access, and stakeholder agreement.

---

## Part 2: Data Collection (Sourcing the Fuel)

An algorithm possesses no innate real-world understanding; it only knows the data you supply. **Flawed collection = broken model (Garbage In, Garbage Out).**

### 5 Production Data Sources
1. **Internal Warehouses (SQL Databases):** Customer purchase logs, transactional history, relational schemas.
2. **Public Repositories:** Academic/benchmark datasets (e.g., Kaggle, UCI ML Repository).
3. **APIs:** Dynamic live feeds (real-time weather, stock tickers, exchange rates).
4. **Web Scraping:** Automated extraction of external public signals (e.g., competitor e-commerce prices).
5. **Manual Labeling / Human Annotation:** High-skill human labeling (e.g., radiologists marking tumor boundaries on CT scans).

---

### Data Structures & Feature Types Taxonomy

#### 1. Data Structures
* **Structured:** Rigid schema organized in rows and columns (e.g., SQL tables: `OrderID`, `Timestamp`, `Amount`, `UserID`).
* **Unstructured:** No row/column format (e.g., images, raw audio waveforms, video files, free-text reviews).
* **Semi-structured:** Contains tags or keys without a rigid tabular schema (e.g., JSON payloads, XML feeds, IoT device telemetry).
* **Time-series:** Temporal records indexed sequentially across uniform time intervals (e.g., hourly weather, minute-by-minute stock ticks).

#### 2. Feature Types
* **Categorical (Nominal):** Qualitative labels with **no natural ordering** (e.g., Color: Red/Green/Blue; Blood Type: A/B/AB/O; City).
* **Categorical (Ordinal):** Qualitative labels with **a distinct, natural hierarchy/rank** (e.g., Education: High School < Bachelor < Master < PhD; Rating: Low < Med < High).
* **Numerical (Discrete):** Countable integers with distinct values (e.g., number of rooms, items in a digital cart).
* **Numerical (Continuous):** Infinitely divisible measurements (e.g., income, temperature, vehicle speed).

---

### Practice & Critical Thinking (Part 2)

#### Practice P2: Feature Classification
| Feature Description | Structure Type | Feature Type |
| :--- | :--- | :--- |
| **(a)** Food delivery ratings: 1, 2, 3, 4, or 5 stars | **Structured** | **Categorical (Ordinal)** |
| **(b)** Patient body temperature logged hourly for 7 days | **Time-series** | **Numerical (Continuous)** |
| **(c)** Product descriptions written by e-commerce sellers | **Unstructured** | **N/A (Free Text)** |
| **(d)** Government address database postal code (PIN code) | **Structured** | **Categorical (Nominal)** |
| **(e)** Weather station wind speed recorded every 10 minutes | **Time-series** | **Numerical (Continuous)** |

#### Reflect 2: Quality vs. Quantity
* **Question:** A web scraper extracts 100,000 product reviews, but 40% are bot-generated spam. Is more data always better?
* **Answer:** **No.** More data harms performance if it contains noise, systematic bias, or adversarial spam. In machine learning, **data quality strictly dominates raw volume**. Noisy spam must be filtered out prior to training.

> **Takeaway:** Know your sources, verify your data structures, and establish feature types before writing model code.

---

## Part 3: Exploratory Data Analysis (EDA)

EDA is the **diagnostic phase**. A physician never prescribes invasive surgery before taking vitals; an engineer must never run a model before diagnosing raw data characteristics.

### The 7-Point EDA Checklist
1. **Dataset Structure:** Matrix shape (rows × columns), schema data types (`int64`, `float64`, `object`), and naming anomalies.
2. **Missing Values:** Detection of nulls/NaNs, percentage missing per column, and pattern assessment (Missing Completely at Random vs. Systematic).
3. **Outliers:** Identification of extreme values (sensor glitches, input errors, legitimate rare events like high-value fraud).
4. **Duplicates:** Identification of repeated rows arising from joining glitches or duplicate tracking calls.
5. **Distributions:** Assessing normality, skewness (positive/right skew vs. negative/left skew), and multi-modal distributions.
6. **Relationships & Correlation:** Collinearity checks, bivariate associations, and redundant feature identification.
7. **Class Imbalance:** Ratio between target classes (e.g., 99% Non-Fraud vs. 1% Fraud).

---

### Analysis Levels & Visualization Selection

| Analysis Level | Scope & Purpose | Recommended Charts |
| :--- | :--- | :--- |
| **Univariate** | Examines a single column (spread, central tendency, outliers). | **Histogram** (distribution), **Box Plot** (IQR, outliers) |
| **Bivariate** | Analyzes the relationship between two variables. | **Scatter Plot** (continuous vs continuous), **Heatmap** |
| **Multivariate**| Analyzes complex interactions across 3+ variables. | **Pair Plot**, **Grouped Heatmaps**, **Facet Grids** |

#### Common Visualizations Cheat Sheet
* **Histogram:** Skewness, unimodal vs bimodal distributions.
* **Box Plot:** Outlier detection, spread, quartiles (Q1, Q2/Median, Q3, IQR).
* **Bar Chart:** Frequency counts across distinct categorical groups.
* **Scatter Plot:** Co-variation, linear/non-linear trends between two continuous variables.
* **Correlation Heatmap:** Linear correlation matrix scored between -1.0 and +1.0.

---

### Practice & Critical Thinking (Part 3)

#### Practice P3: EDA Protocol on 100,000-row Customer Dataset
* **Dataset Columns:** `Customer_ID`, `Age`, `Gender`, `Annual_Income`, `Spending_Score`, `City`.
* **First 3 Steps to Run:**
  1. Inspect dataset shape, data types, and initial numerical summaries via `df.info()` and `df.describe()`.
  2. Audit missing entries and duplicate rows via `df.isnull().sum()` and `df.duplicated().sum()`.
  3. Visualize single-feature distributions and detect outliers using histograms for `Age`/`Annual_Income` and box plots for `Spending_Score`.
* **Chart Selection:**
  * Chart for `Age` distribution → **Histogram**
  * Chart for `Annual_Income` vs. `Spending_Score` → **Scatter Plot**
  * Chart to verify if `Gender` affects `Spending_Score` → **Bar Chart / Box Plot**

#### Reflect 3: The Imbalance Accuracy Trap
* **Question:** A fraud dataset contains 95% Class A (non-fraud) and 5% Class B (fraud). A dummy classifier predicts Class A 100% of the time, achieving 95% accuracy. Is this model useful?
* **Answer:** **No, it is completely useless.** The model has **0% Recall** on Class B—it catches zero fraud cases. When classes are heavily skewed, raw accuracy is deceptive. Always evaluate with **Precision, Recall, F1-Score, or ROC-AUC**.

> **Takeaway:** EDA first, algorithms later. Understand structure → uncover defects → visualize relationships → proceed to preprocessing.

---

## Part 4: Data Preprocessing (The Real Engineering Work)

Machine learning models represent mathematical operations on numerical matrices. If an equation encounters a null value, execution halts. If one feature spans 1–5 while another spans 10,000–500,000, gradient descent and distance metrics collapse.

---

### Step 1: Handling Missing Values

| Strategy | Operational Action | When to Use | Danger / Risk |
| :--- | :--- | :--- | :--- |
| **Deletion** | Drop incomplete rows or drop the entire column. | Drop rows if missingness <1–2%. Drop column if >80% empty. | Discards valuable information; risks introducing sampling bias. |
| **Imputation** | Fill missing values using mean/median (numerical) or mode (categorical). | Moderate missingness (e.g., 5%–20%). | Can distort variances and create artificial correlations. |

---

### Step 2: Handling Inconsistent Data
* **Mixed Units:** Weight logged in both kilograms and pounds → **Standardize to a single unit**.
* **Typos / Inconsistent Strings:** "Californa", "California", "CA" → **Regex cleaning and Levenshtein string distance matching**.
* **Duplicate Rows:** Repeated identical database events → **Deduplicate via primary key or hashing**.
* **Inconsistent Labels:** "Yes", "Y", "True", "1" → **Map all occurrences to a canonical binary flag (1 / 0)**.

---

### Step 3: Feature Scaling

Distance-based algorithms (KNN, SVM) and optimization solvers (Gradient Descent) are heavily dominated by features with large absolute numerical scales.

#### Comparison of the 4 Primary Scalers

| Scaler | Mathematical Formula | Optimal Use Case | Core Risk / Limitation |
| :--- | :--- | :--- | :--- |
| **A. Min-Max Normalization** | `X_scaled = (X - X_min) / (X_max - X_min)`<br>*(Maps strictly to [0, 1])* | Features with bounded ranges (e.g., image pixels 0–255), neural network inputs. | Extreme outliers squash all non-outlier data into a microscopic range near 0. |
| **B. Z-Score Standardization** | `Z = (X - μ) / σ`<br>*(Mean = 0, Std = 1)* | Data conforming roughly to a Gaussian / Normal distribution. | Does not bound data to a fixed interval; outliers still exist in standard deviations. |
| **C. Max Absolute Scaler** | `X_scaled = X / max(\|X\|)`<br>*(Maps strictly to [-1, 1])* | Sparse matrices (e.g., text TF-IDF vectors, Bag-of-Words). | Preserves true zero entries and signs without destroying sparsity. |
| **D. Robust Scaler** | `X_scaled = (X - Median) / IQR`<br>*(Where IQR = Q3 - Q1)* | Datasets containing severe, non-droppable outliers. | Uses median and IQR, making it immune to extreme maximum or minimum values. |

---

### Hand Computations & Worked Examples

#### 1. Min-Max Scaling Walkthrough
* Dataset: `X = [10, 20, 30, 40, 50]`
* `X_min = 10`, `X_max = 50`, `X_max - X_min = 40`
* For `X = 10`: `(10 - 10) / 40 = 0.0`
* For `X = 30`: `(30 - 10) / 40 = 20 / 40 = 0.5`
* For `X = 50`: `(50 - 10) / 40 = 40 / 40 = 1.0`

#### 2. Z-Score Standardization Walkthrough
* Dataset: `X = [2, 4, 6, 8, 10]`
* Mean `μ = 6`
* Standard Deviation `σ = √( [(-4)² + (-2)² + 0² + 2² + 4²] / 5 ) = √(40 / 5) = √8 ≈ 2.83`
* For `X = 2`: `(2 - 6) / 2.83 = -4 / 2.83 ≈ -1.41`
* For `X = 6`: `(6 - 6) / 2.83 = 0.0`
* For `X = 10`: `(10 - 6) / 2.83 = 4 / 2.83 ≈ +1.41`

#### 3. Robust Scaling Walkthrough (Outlier Resilience)
* Dataset: `X = [100, 150, 200, 250, 50000]` *(contains massive outlier 50,000)*
* **Step 1:** Sorted array: `[100, 150, 200, 250, 50000]`
* **Step 2:** Median `Q2 = 200`
* **Step 3:** `Q1` (lower half median `[100, 150]`): `(100 + 150) / 2 = 125`
* **Step 4:** `Q3` (upper half median `[250, 50000]`): `(250 + 50000) / 2 = 25125`
* **Step 5:** `IQR = Q3 - Q1 = 25125 - 125 = 25000`
* **Step 6 (Scale X = 100):** `(100 - 200) / 25000 = -100 / 25000 = -0.004`
* **Step 7 (Scale X = 50000):** `(50000 - 200) / 25000 = 49800 / 25000 = 1.992`
* **Outcome:** The normal points maintain meaningful separation without being flattened to zero.

#### Practice P5 & Reflect 4: Scaler Selection Rules
* **Salary data ($20k–$5M with CEO outliers at $50M):** → Use **Robust Scaler** (IQR resists extreme salary values).
* **Image pixel intensities (bounded [0, 255], zero outliers):** → Use **Min-Max Normalization** (maps directly to [0, 1]).
* **Why Min-Max fails with an outlier of 50,000:** The denominator `(50000 - 100) = 49900` becomes enormous. All standard observations under 300 are compressed into a tiny sliver between `0.000` and `0.004`, stripping the model of feature variance.

---

### Step 4: Categorical Encoding

Mapping categories arbitrarily to integers (e.g., Red=0, Green=1, Blue=2) introduces artificial mathematical ordering: `Blue (2) > Green (1) > Red (0)` and `(Red + Blue) / 2 = Green`. This severely degrades linear and distance-based algorithms.

#### Encoding Strategies Compared

| Encoding Technique | How It Operates | Best / Safe For | Severe Risk / Danger |
| :--- | :--- | :--- | :--- |
| **Label Encoding** | Assigns an arbitrary integer per category (0, 1, 2...). | Target column `y` (e.g., Spam=1, Ham=0); Tree-based splits. | Imposes artificial mathematical distance in linear/distance models (KNN, SVM). |
| **Ordinal Encoding** | Maps categories to ordered integers reflecting real rank. | Truly ordered features (e.g., Low=0, Med=1, High=2). | Destructive if applied to nominal unordered categories. |
| **One-Hot Encoding** | Creates `N` new binary indicator columns (1 or 0). | Nominal features with low cardinality (<15 categories). | **Curse of Dimensionality** when applied to high-cardinality features. |
| **Binary Encoding** | Converts integer IDs into binary digits across log₂(N) columns. | High-cardinality nominal features (e.g., hundreds of cities/ZIPs). | Produces fewer columns than One-Hot, but harder to interpret directly. |

---

### Practice & Critical Thinking (Part 4)

#### Practice P6: Selecting the Right Encoding
1. **500 city names, model is Logistic Regression:**  
   * **Selection:** **Binary Encoding**.  
   * **Justification:** One-Hot produces 500 sparse columns; Binary Encoding compresses 500 categories into `⌈log₂(500)⌉ = 9 columns`.
2. **Education level (High School < Bachelor < Master < PhD), model is Linear Regression:**  
   * **Selection:** **Ordinal Encoding**.  
   * **Justification:** A true hierarchy exists; integer ranks preserve monotonic ordering for linear coefficients.
3. **Primary Color (Red, Blue, Green), model is KNN:**  
   * **Selection:** **One-Hot Encoding**.  
   * **Justification:** Low-cardinality nominal data; KNN computes Euclidean distances and requires symmetric binary indicators.

#### Practice P7: Binary Encoding by Hand
* **Categories:** `[Cat, Dog, Fish, Bird]`
* Integers assigned: `Cat = 1`, `Dog = 2`, `Fish = 3`, `Bird = 4`
* Convert to Binary:
  * `Cat (1)` → `001` (or `01`)
  * `Dog (2)` → `010` (or `10`)
  * `Fish (3)` → `011` (or `11`)
  * `Bird (4)` → `100`
* **Columns required:** One-Hot needs **4 columns**; Binary needs **2 to 3 columns**.

#### Reflect 5: Why Label Encoding Misleads KNN but NOT Decision Trees
* **KNN (Distance-based):** Computes Euclidean distances: `d(a, b) = √[ Σ(a_i - b_i)² ]`. Assigning arbitrary integers creates false spatial distances (`Blue` is calculated as further from `Red` than `Green`).
* **Decision Trees (Rule-based):** Trees split on orthogonal thresholds (`if feature ≤ 1.5`). They evaluate partition boundaries without computing distances or vector magnitudes.

---

## 🎯 Review & Self-Assessment Checklist
Before moving to Worksheet 02, verify mastery of these core competencies:

- [ ] Translate ambiguous business requests into rigorous ML problem statements with explicit metrics.
- [ ] Differentiate between structured, semi-structured, unstructured, and time-series data.
- [ ] Correctly classify features into Categorical (Nominal vs. Ordinal) and Numerical (Discrete vs. Continuous).
- [ ] Execute an EDA checklist: identify missing values, outliers, duplicates, and class imbalance.
- [ ] Understand why high accuracy on imbalanced data can mask a completely failed model.
- [ ] Compute Min-Max, Z-Score, and Robust Scalers by hand.
- [ ] Choose the optimal categorical encoder (Label, Ordinal, One-Hot, Binary) and defend your choice.
