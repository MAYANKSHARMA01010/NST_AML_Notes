# Machine Learning Made Simple — Doc 2: The ML Project Lifecycle (Part 2)
**Worksheet:** [Worksheet 02](file:///Users/mayanksharma/Downloads/AML/02_Worksheets/Worksheet_02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.pdf)  
**Topics:** Data Splitting · Feature Engineering · Training & Gradient Descent · Evaluation Metrics · Deployment & Drift

---

## 🗺️ Where Are We in the Journey?

In [Doc 1](file:///Users/mayanksharma/Downloads/AML/docs/01_ML_Project_Lifecycle_EDA_and_Preprocessing.md), we completed the first 4 steps:
1. Define the Problem
2. Collect the Data
3. Run EDA (Check our data)
4. Clean and Preprocess (Scale numbers and encode words)

Now in **Doc 2**, we build, test, and deploy the model:
* **Phase 5:** Data Splitting *(Don't cheat on the exam!)*
* **Phase 6:** Feature Engineering *(Helping the model see patterns)*
* **Phase 7:** Baseline Establishment *(The bare minimum score)*
* **Phase 8:** The Modeling Loop *(Training with Gradient Descent)*
* **Phase 9:** Offline Evaluation *(Grading the final exam)*
* **Phase 10:** Model Deployment *(Putting the model into the real world)*
* **Phase 11:** Post-Deployment *(Watching out for Model Drift)*

---

## Core Prerequisite: Parameters vs. Hyperparameters

Before touching anything, you must understand the difference between these two words. They sound similar, but they are completely different!

### The Simple Analogy: Baking a Cake
* **Hyperparameters (The Dials You Set):**  
  Before you bake, you decide: *"I will bake this at 180°C for 30 minutes."*  
  **You (the human)** manually choose these settings *before* starting.
* **Parameters (The Internal Reactions):**  
  Inside the oven, the cake slowly rises, bubbles, and turns golden brown.  
  **The oven and cake** adjust these physical changes *automatically* while baking.

---

### Comparison Table

| Feature | Parameters | Hyperparameters |
| :--- | :--- | :--- |
| **Who sets it?** | **The computer model** (learns it automatically). | **The human engineer** (picks it manually). |
| **When is it set?** | **During** training (as the model sees examples). | **Before** training starts. |
| **What are examples?** | • Weights (`w1, w2`) in a line equation.<br>• Intercept / bias (`w0`). | • `max_depth` (how deep a Decision Tree can grow).<br>• `learning_rate` (how big of a step gradient descent takes).<br>• `n_estimators` (how many trees in a Random Forest). |

#### Practice P1: Classify Parameter [P] or Hyperparameter [H]
1. Weight `w1` in a regression equation → **[ P ] Parameter** *(learned automatically)*
2. `max_depth` of a Decision Tree → **[ H ] Hyperparameter** *(set by you)*
3. Intercept `w0` in `y = w0 + w1*x1` → **[ P ] Parameter** *(learned automatically)*
4. `learning_rate` for Gradient Descent → **[ H ] Hyperparameter** *(set by you)*
5. `n_estimators` in a Random Forest → **[ H ] Hyperparameter** *(set by you)*
6. Bias term learned during neural net backpropagation → **[ P ] Parameter** *(learned automatically)*

---

## Phase 5: Data Splitting (The Golden Barrier)

### The Hook: Why do we split data?
Imagine a teacher gives students 50 math questions for homework, complete with all answers.  
One lazy student simply **memorizes every single answer by heart**.  
The next day, if the teacher tests them on the *exact same 50 questions*, the student gets 100%!

Does getting 100% mean the student understands math? **No!**  
If the teacher changes even one number, the student will fail completely.  
This is called **Overfitting** (memorizing instead of learning).

---

### The 3-Way Split: Train, Validation, and Test

To prevent memorization, we split our data into 3 separate piles:

```
+------------------------------------+-----------------------+-----------------------+
|        Training Set (~70%)         | Validation Set (~15%) |    Test Set (~15%)    |
|   Used to learn model parameters   | Used to tune your HPs | The locked Final Exam |
+------------------------------------+-----------------------+-----------------------+
```

1. **Training Set (~70%):** The textbook and homework problems. The model uses this to learn its weights and parameters.
2. **Validation Set (~15%):** The practice mock tests. You use this to test different hyperparameters (like trying tree depth 3 vs depth 5) and pick the best one.
3. **Test Set (~15%):** The **FINAL EXAM**. It must stay locked away in a safe until the very end. The model is only allowed to see it once to get its true final grade.

---

### ⚠️ What is "Data Leakage"? (Exam Classic!)

> **Data Leakage** means **cheating on the exam by accidentally looking at the questions before the test**.

#### Reflect 1: The Leakage Scenario
* **The Mistake:** You have 100,000 customer rows. Before doing any splitting, you compute the average salary of all 100,000 people to fill in blank cells. Then, you split into Train and Test.
* **Is this Data Leakage?** **YES!**  
  Because your average salary calculation included people from the Test set! You accidentally leaked information from the future test into your training data.
* **The Golden Rule:** **Split FIRST, do everything else SECOND.** The test set is sacred—never touch it during training!

---

## Phase 6: Feature Engineering (Crafting Perspective)

Raw data is like a block of rough stone. Feature Engineering is sculpting that stone so the computer can easily see the statue inside.

It has two main pillars:
1. **Feature Creation** (Making new smart columns)
2. **Feature Selection** (Throwing away useless noise columns)

---

### 1. Feature Creation

#### A. Using Domain Knowledge
Instead of giving the computer an exact noisy decimal number like `41.3°C`, use human common sense to group temperatures into buckets:
* Cold (≤15°C)
* Warm (16°C – 30°C)
* Hot (>30°C)

#### B. Interaction Terms (Combining features together)
Sometimes two features only make sense when combined!
* *Example:* Suppose you want to predict a student's exam score:
  * Student A studies 10 hours, but sleeps 0 hours → **Fails** (too exhausted).
  * Student B sleeps 10 hours, but studies 0 hours → **Fails** (didn't study).
  * Neither column alone explains success! You need the combination:  
    `Study_Sleep_Score = Study_Hours × Sleep_Hours`
* *House Price Example:* If you have `Length` and `Width`, combine them into:  
  `Area = Length × Width`

---

### 2. Feature Selection (Throwing away useless junk)

If you feed 500 columns into an algorithm, 450 of them might be useless garbage (like customer shirt color when predicting bank loan defaults). Useless columns confuse the model.

* **Filter Methods:** You check columns *before training* using basic statistics.  
  *(Example: If a column has near 0.0 correlation with house price, delete it).*
* **Embedded Methods:** The model itself automatically figures out which columns were important *during training*.  
  *(Example: A Random Forest calculates an internal "Feature Importance Score" and tells you which features mattered).*

#### Practice P3: Matching
* Dropping columns with 0.05 correlation to target → **Filter Method**
* Multiplying `Rainfall × Soil_Moisture` → **Combining Features (Interaction Term)**
* Random Forest ranking features by importance → **Embedded Method**

---

## Phase 7: Baseline Establishment (Setting the Floor)

Before you waste months building a complex AI model, you must ask:  
**"What is the simplest, dumbest guess possible?"**

That dumb guess is called your **Baseline**:
* In predicting house prices, a simple baseline is just guessing the **average house price** for every house.
* In predicting if an email is spam, a baseline is guessing **"Not Spam"** for everything.

### Why do we need a Baseline?
1. **Technical Benchmark:** If your fancy AI cannot beat a simple average guess, your AI is completely useless.
2. **Business Cost-Benefit (Practice P4):**  
   * Suppose a simple Decision Tree gives you **80% accuracy** for free.
   * A huge Random Forest gives you **81% accuracy** (just 1% better), but requires buying expensive cloud servers that cost **₹20 lakh**.
   * **Is it worth it?** **NO!** A tiny 1% improvement does not justify spending ₹20 lakh. Always compare against a baseline!

---

## Phase 8: The Modeling Loop (Select, Tune & Train)

### 1. Model Selection (Picking the right tool)

There is no single "best" algorithm. It is always a trade-off:

| Trade-Off | What it means | Real-World Example |
| :--- | :--- | :--- |
| **Interpretability vs. Accuracy** | Can a human explain WHY the model made a prediction? | High accuracy models (like Deep Neural Nets) are black boxes—you don't know why they said yes or no. |

#### Practice P5: The Hospital Case Study (Interview Classic!)
* **Scenario:** A hospital is choosing an AI model to detect heart attacks:
  * **Model A:** 85% accurate, but fully explainable. It gives the doctor a clear checklist of reasons.
  * **Model B:** 92% accurate, but a complete black box. It gives no reasons at all.
* **Which one should the doctor choose?**  
  * **Choose Model A (Explainable)!**  
  * *Why?* Because in healthcare, human lives are on the line. If a black-box model makes a weird mistake, someone could die and nobody knows why. Doctors need to verify and trust the diagnosis.

---

### 2. Model Training (The Engine Room)

Let's look at a simple regression model that predicts crop yield from rainfall (`x1`) and soil moisture (`x2`):

```
Predicted_Yield = w0 + (w1 × Rainfall) + (w2 × Soil_Moisture)
```
* `w0` = Intercept (baseline yield when rain and moisture are 0).
* `w1, w2` = Weights (how strongly rain or moisture impacts the yield).

Training is the process of finding the perfect numerical values for `w0, w1, w2` so the predictions are as close to reality as possible!

---

### 3. How do we measure mistakes? The Loss Function (MSE)

Think of the Loss Function like a penalty score in sports—**the smaller the score, the better you played!**

The most common penalty score is **Mean Squared Error (MSE)**:
```
MSE = (1 / n) × Σ (Actual_Value - Predicted_Value)²
```
* Why do we square the mistakes?
  1. It turns negative mistakes into positive numbers so they don't cancel each other out.
  2. It punishes big mistakes very harshly (a mistake of 10 becomes a penalty of 100!).

---

### 4. How does the computer learn? Gradient Descent

Imagine this simple story:
> You are hiking on a mountain. Suddenly, a thick fog rolls in, and you cannot see your hands in front of your face. You want to get down to the bottom of the valley.  
> What do you do?  
> You can't see the valley, but **you can feel the ground with your feet**. You feel which way slopes downward, and you take a careful step in that downhill direction. You repeat this over and over until the ground under your feet feels completely flat.

* **Your altitude (height)** = The Loss / Error.
* **Your coordinates (where you stand)** = The Model Weights.
* **Taking a step downhill** = **Gradient Descent**.
* **When the ground flattens out** = **Convergence** (Training is done!).

---

### The 5 Steps of the Training Loop (Practice P6)
1. **Initialize:** Start with random numbers for weights (`w0, w1, w2`).
2. **Forward Pass:** Use the equation to calculate predictions (`ŷ`).
3. **Compute Loss:** Calculate how far off the predictions were using MSE.
4. **Backward Pass:** Use Gradient Descent to figure out the downhill slope and adjust weights.
5. **Check Convergence:** Did the error stop decreasing? If yes, stop! If no, repeat from Step 2.

---

## Phase 9: Offline Evaluation (The Final Exam)

Now that training is done, we bring out the locked **Test Set** and test the model.

### The 3 Big Regression Metrics

| Metric | What it measures | When should you use it? |
| :--- | :--- | :--- |
| **MAE**<br>*(Mean Absolute Error)* | The simple average mistake: `Average of \|Actual - Predicted\|`. | When you want to explain the error to a non-technical person: *"Our delivery time estimates are off by 5 minutes on average."* |
| **RMSE**<br>*(Root Mean Squared Error)* | The squared root of average squared mistakes. **Penalizes large mistakes very heavily!** | When making a big mistake is **fatal or dangerous** (e.g. medical medicine dosage or structural bridge stress). |
| **R² Score**<br>*(R-squared)* | Scores from **0 to 1** comparing your model against a dumb average guess. | When your boss asks: *"How much better is this AI model than just random guessing?"* (e.g. 0.85 means the model explains 85% of patterns). |

#### Practice P7: Pick the right metric
* (a) A medical dosage prediction error of 100 units is catastrophic (can kill a patient) → **RMSE** *(because RMSE heavily penalizes large errors)*.
* (b) A CEO asks: *"How much better is this model than a basic average guess?"* → **R² Score**.
* (c) A logistics company asks: *"On average, how many kilometers are our delivery drivers off?"* → **MAE**.

---

## Phase 10: Model Deployment (Putting It to Work)

Building a model on your laptop is like cooking a meal and leaving it in the kitchen. To serve it to customers, you must deploy it!

1. **Model Serialization (Saving it):**  
   We save the trained weights into a file using Python's `pickle` library as a **`.pkl` file**.
2. **Where does it live?**
   * **Web API (Flask / FastAPI):** You wrap the model in a web URL so other apps can send data and get predictions.
   * **Cloud (AWS SageMaker / Google Cloud):** Runs on big server clusters.
   * **Edge (Mobile / IoT):** Runs directly on an iPhone or smart watch without needing the internet.
3. **How does it make predictions?**
   * **Batch Prediction:** Runs once in a while on a huge list of data.  
     *(Example: Calculating Netflix movie recommendations once every night for all users).*
   * **Real-Time Prediction:** Makes an instant guess in milliseconds on demand.  
     *(Example: Checking if a credit card swipe is fraud at the cash register).*

---

## Phase 11: Post-Deployment & The Living System

Here is a big difference between traditional software and AI:
* Normal code (like a calculator app) runs the exact same way forever.
* **Machine Learning models decay and get worse over time!**

### What is Model Drift?
**Model Drift** happens when the real world changes, making your past training data outdated!

> **The Best Example:** Suppose in 2019 you built an AI model to predict airline flight bookings using data from 2010–2019. It had 99% accuracy!  
> In 2020, COVID-19 lockdowns hit the world.  
> Your model's accuracy instantly dropped to zero because human travel habits completely changed overnight. That is **Model Drift**.

### How do we fix Model Drift?
1. **Continuous Monitoring:** Tools like **Prometheus** and **MLflow** monitor live predictions in the background.
2. **Retraining (Practice P9):**
   * **Periodic Retraining:** Retrain on a fixed schedule (e.g., retrain automatically on the 1st of every month).
   * **Triggered Retraining:** An alarm goes off when live accuracy drops below 85%, automatically retraining the model on fresh data!

---

## 🌾 Capstone Case Study: Crop Yield System (Practice P10 & P11)

Let's test everything from start to finish on a real-world project:

1. **Which dataset stays locked until the very end?**  
   → **The Test Set**.
2. **Create an interaction feature using `Rainfall` and `Sunlight_Hours`:**  
   → `Water_Sun_Index = Rainfall × Sunlight_Hours`.
3. **Your baseline predicts yield with an error of 12. Your fancy model achieves an error of 11, but costs ₹15 lakh to build. Do you deploy it?**  
   → **NO!** Saving 1 point of error does not justify spending ₹15 lakh.
4. **An error of 50 tonnes of crops is catastrophic. Which metric should you use?**  
   → **RMSE** *(because it penalizes big errors harshly)*.
5. **You need to predict crop yields once per growing season across 10,000 farms. Which inference mode?**  
   → **Batch Inference** *(runs once per season in bulk)*.
6. **A severe drought permanently changes the region's climate. What is this called, and how do you fix it?**  
   → This is called **Model Drift**, and it is fixed via **Triggered Retraining**.

---

## 📋 Summary of Sheet 2
1. **Parameters** are learned by the computer; **Hyperparameters** are set by you.
2. **Split Data First** to avoid Data Leakage (cheating on the test).
3. **Create smart features** (like interaction terms) and **select only useful ones**.
4. **Always build a Baseline** to make sure ML is actually worth the time and money.
5. **Gradient Descent** is like walking down a foggy mountain to find the lowest error.
6. Pick **MAE** for simple average explanation, **RMSE** for dangerous big mistakes, and **R²** for baseline comparison.
7. Models decay due to **Model Drift**, so always monitor and retrain!
