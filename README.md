# Advanced Machine Learning Course Repository

This repository contains all organized lecture notes, worksheets, interactive HTML visualizations, and hands-on lab notebooks for the **Advanced Machine Learning (AML) Course (Semester 5)**.

All materials are categorized into dedicated folders and cross-referenced module-by-module. Furthermore, all lab exercises share a single unified virtual environment (`.venv`), with dedicated folder structures for datasets, student solutions, and instructor reference keys.

---

## Official Course Resources

- 📁 **AML Notes & Worksheets (Google Drive)**: [Google Drive Folder](https://drive.google.com/drive/u/2/folders/1L7aVVAE6_WF0m2sMKjc1BR6zBrrlQnLa)
- 🐙 **AML Lab Questions (GitHub Repo)**: [gourabrajak-cloud/aml_lab](https://github.com/gourabrajak-cloud/aml_lab)

---

### Directory Structure

```text
AML/
├── .gitignore                         # Git exclusion rules (.venv, checkpoints, pycache)
├── .venv/                             # Common shared Python 3.14 virtual environment
├── README.md                          # Master course documentation & module guide
├── scripts/
│   ├── launch_jupyter.sh              # 1-click launcher activating shared .venv & JupyterLab
│   └── sync_labs.sh                   # Upstream pull script for new assignments & questions
├── 01_Notes/                          # Lecture notes and foundational reference guides
│   └── AML_Foundations_Lecture_0_Guide.pdf
├── 02_Worksheets/                     # Theoretical & practical worksheets (Lectures 00 to 11)
│   ├── Worksheet_00_Foundations_From_Code_to_Learning.pdf
│   ├── Worksheet_01_ML_Project_Lifecycle_EDA_and_Preprocessing.pdf
│   ├── Worksheet_02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.pdf
│   ├── Worksheet_03_Simple_Linear_Regression_OLS_Teacher_Copy.pdf
│   ├── Worksheet_04_Multiple_Linear_Regression_OLS.pdf
│   ├── Worksheet_05_Batch_Gradient_Descent_MLR.pdf
│   ├── Worksheet_06_Stochastic_and_MiniBatch_Gradient_Descent.pdf
│   ├── Worksheet_07_Regression_and_Classification_Evaluation_Metrics.pdf
│   ├── Worksheet_08_Polynomial_Regression_and_Assumptions.pdf
│   ├── Worksheet_09_Bias_Variance_and_Tradeoff.pdf
│   ├── Worksheet_10_Feature_Selection.pdf
│   └── Worksheet_11_Dimensionality_Reduction_and_PCA.pdf
├── 03_HTML_Visualizations/            # Interactive browser-based learning explorables
│   ├── Module_06_Gradient_Descent_Interactive_Lecture.html
│   └── Module_07_Momentum_and_NAG_Interactive_Worksheet.html
├── 04_Notebooks/                      # Hands-on Jupyter lab assignments (Labs 01 to 11)
│   ├── Lab_01_EDA_and_Preprocessing/
│   │   └── Lab_01_EDA_and_Preprocessing.ipynb
│   ├── Lab_02_Feature_Engineering_and_Outliers/
│   │   └── Lab_02_Feature_Engineering_and_Outliers.ipynb
│   ├── Lab_03_Simple_Linear_Regression_OLS/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── Lab_3_student_notebook.ipynb
│   │   │   ├── slr_ols_todo.py
│   │   │   ├── student_clean_dataset.csv
│   │   │   ├── student_raw_dataset.csv
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   └── solved/                    # Your solved version & Jupyter checkpoints
│   │       ├── .ipynb_checkpoints/
│   │       ├── Lab_3_student_notebook.ipynb (24 outputs, tracking intact)
│   │       ├── slr_ols_todo.py (your solved implementation)
│   │       ├── student_clean_dataset.csv
│   │       ├── student_raw_dataset.csv
│   │       └── requirements.txt
│   ├── Lab_04_Multiple_Linear_Regression_OLS/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── Lab_4_student_notebook.ipynb
│   │   │   ├── mlr_ols_todo.py
│   │   │   ├── student_clean_dataset.csv
│   │   │   ├── student_raw_dataset.csv
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   └── solved/                    # Your solved version & Jupyter checkpoints
│   │       ├── .ipynb_checkpoints/
│   │       ├── Lab_4_student_notebook.ipynb (26 outputs, tracking intact)
│   │       ├── mlr_ols_todo.py (your solved implementation)
│   │       ├── student_clean_dataset.csv
│   │       ├── student_raw_dataset.csv
│   │       └── requirements.txt
│   ├── Lab_05_Batch_Gradient_Descent_MLR/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── Lab_5_Final_BGD_MLR_student_todo.ipynb
│   │   │   ├── MLR_Gradient_Descent_Student_TODO.txt
│   │   │   ├── gradient_descent.py
│   │   │   ├── mlr_gradient_descent.py
│   │   │   ├── slr_ols_student.py
│   │   │   ├── student_clean_dataset.csv
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   └── solved/                    # Your solved version & Jupyter checkpoints
│   │       ├── .ipynb_checkpoints/
│   │       ├── Lab_5_Final_BGD_MLR_student_todo.ipynb (21 outputs, tracking intact)
│   │       ├── MLR_Gradient_Descent_Student_TODO.txt
│   │       ├── gradient_descent.py
│   │       ├── mlr_gradient_descent.py (your solved implementation)
│   │       ├── slr_ols_student.py
│   │       ├── student_clean_dataset.csv
│   │       └── requirements.txt
│   ├── Lab_06_MiniBatch_SGD_MLR/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── Lab_6_Final_BGD_MLR_student_todo.ipynb
│   │   │   ├── mlr_gradient_descent_minibatch.py
│   │   │   ├── dfd.png
│   │   │   ├── student_clean_dataset.csv
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   └── solved/                    # Your solved version & Jupyter checkpoints
│   │       ├── .ipynb_checkpoints/
│   │       ├── Lab_6_Final_BGD_MLR_student_todo.ipynb (16 outputs, tracking intact)
│   │       ├── dfd.png
│   │       ├── mlr_gradient_descent_minibatch.py (your solved implementation)
│   │       ├── student_clean_dataset.csv
│   │       └── requirements.txt
│   ├── Lab_07_Student_Performance_and_Placement/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── lab_7_St_c.ipynb
│   │   │   └── student_raw_dataset.csv
│   │   └── solved/                    # Ready to solve in Jupyter with tracking
│   │       ├── .ipynb_checkpoints/
│   │       ├── lab_7_St_c.ipynb
│   │       └── student_raw_dataset.csv
│   ├── Lab_08_Polynomial_Regression/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── Poly_using_gradient_final.ipynb
│   │   │   ├── poly_using_ols_manual.ipynb
│   │   │   ├── poly_gradient.py
│   │   │   └── poly_ols.py
│   │   └── solved/                    # Ready to solve in Jupyter with tracking
│   │       ├── .ipynb_checkpoints/
│   │       ├── Poly_using_gradient_final.ipynb
│   │       ├── poly_using_ols_manual.ipynb
│   │       ├── poly_gradient.py
│   │       └── poly_ols.py
│   ├── Lab_09_Bias_Variance_Tradeoff/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── Lab_9_Bias_Variance.ipynb
│   │   │   ├── academic_outcomes.csv
│   │   │   └── bias_variance_tools.py
│   │   └── solved/                    # Ready to solve in Jupyter with tracking
│   │       ├── .ipynb_checkpoints/
│   │       ├── Lab_9_Bias_Variance.ipynb
│   │       ├── academic_outcomes.csv
│   │       └── bias_variance_tools.py
│   ├── Lab_10_Feature_Selection/
│   │   ├── raw/                       # Exact question from GitHub (untouched)
│   │   │   ├── AML_Lab_10_Feature_Selection_Student_TODO.ipynb
│   │   │   └── student_clean_dataset_2024.csv
│   │   └── solved/                    # Ready to solve in Jupyter with tracking
│   │       ├── .ipynb_checkpoints/
│   │       ├── AML_Lab_10_Feature_Selection_Student_TODO.ipynb
│   │       └── student_clean_dataset_2024.csv
│   └── Lab_11_PCA_Dimensionality_Reduction/
│       ├── raw/                       # Exact question from GitHub (untouched)
│       │   ├── L11_Student_PCA_TODO.ipynb
│       │   └── student_clean_dataset.csv
│       └── solved/                    # Ready to solve in Jupyter with tracking
│           ├── .ipynb_checkpoints/
│           ├── L11_Student_PCA_TODO.ipynb
│           └── student_clean_dataset.csv
└── .upstream/                         # Hidden git mirror for aml_lab (used by sync_labs.sh)
```

---

## Course Module Map

### Module 0: Foundations — From Code to Learning
*Transitioning from traditional rule-based software development to statistical learning. Data types, structured vs. unstructured data, learning paradigms (supervised, unsupervised, reinforcement), and core learning tasks.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Notes** | [AML_Foundations_Lecture_0_Guide.pdf](01_Notes/AML_Foundations_Lecture_0_Guide.pdf) | Foundational guide on AI/ML definitions, heuristics, and learning workflows |
| **Worksheets** | [Worksheet_00_Foundations_From_Code_to_Learning.pdf](02_Worksheets/Worksheet_00_Foundations_From_Code_to_Learning.pdf) | In-depth worksheet exploring traditional code vs. learning paradigms |

---

### Module 1: ML Project Lifecycle — EDA & Preprocessing
*The end-to-end Machine Learning lifecycle part 1: defining problem statements, data collection mechanisms, exploratory data analysis (EDA), handling missing values, and data cleaning strategies.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_01_ML_Project_Lifecycle_EDA_and_Preprocessing.pdf](02_Worksheets/Worksheet_01_ML_Project_Lifecycle_EDA_and_Preprocessing.pdf) | Project lifecycle worksheet covering collection, distribution checks, and cleaning |
| **Notebooks** | [Lab_01_EDA_and_Preprocessing.ipynb](04_Notebooks/Lab_01_EDA_and_Preprocessing/Lab_01_EDA_and_Preprocessing.ipynb) | Hands-on exploratory data analysis, visualizations, and preprocessing pipeline |

---

### Module 2: ML Project Lifecycle — Feature Engineering & Evaluation
*The Machine Learning lifecycle part 2: preventing data leakage via strict isolation, feature engineering, encoding categorical variables, outlier treatments, scaling, training loops, and validation metrics.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.pdf](02_Worksheets/Worksheet_02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.pdf) | Data isolation, feature creation, scaling, evaluation metrics, and deployment loops |
| **Notebooks** | [Lab_02_Feature_Engineering_and_Outliers.ipynb](04_Notebooks/Lab_02_Feature_Engineering_and_Outliers/Lab_02_Feature_Engineering_and_Outliers.ipynb) | Feature engineering, IQR outlier filtering, Z-score transformations, and scaling |

---

### Module 3: Simple Linear Regression (SLR) via OLS
*Closed-form mathematical derivation of the Ordinary Least Squares (OLS) line of best fit: deriving slope ($m$) and intercept ($b$), calculating residuals, and understanding variance decomposition ($R^2$, MSE, RMSE).*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_03_Simple_Linear_Regression_OLS_Teacher_Copy.pdf](02_Worksheets/Worksheet_03_Simple_Linear_Regression_OLS_Teacher_Copy.pdf) | Complete mathematical OLS derivation from scatter plots to best-fit lines |
| **Solved Notebook** | [Lab_3_student_notebook.ipynb](04_Notebooks/Lab_03_Simple_Linear_Regression_OLS/solved/Lab_3_student_notebook.ipynb) | Solved notebook with 24 execution outputs and Jupyter tracking intact |
| **Raw Assignment** | [Lab_3 Raw Question](04_Notebooks/Lab_03_Simple_Linear_Regression_OLS/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 4: Multiple Linear Regression (MLR) via OLS
*Extending linear regression to multidimensional feature spaces: matrix formulation $\hat{\beta} = (X^T X)^{-1} X^T y$, assumptions of linear regression, multicollinearity, and parameter interpretation.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_04_Multiple_Linear_Regression_OLS.pdf](02_Worksheets/Worksheet_04_Multiple_Linear_Regression_OLS.pdf) | Matrix formulation, normal equations, and multiple feature interactions |
| **Solved Notebook** | [Lab_4_student_notebook.ipynb](04_Notebooks/Lab_04_Multiple_Linear_Regression_OLS/solved/Lab_4_student_notebook.ipynb) | Solved notebook with 26 execution outputs and `mlr_ols_todo.py` solution |
| **Raw Assignment** | [Lab_4 Raw Question](04_Notebooks/Lab_04_Multiple_Linear_Regression_OLS/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 5: Batch Gradient Descent (BGD) for MLR
*Why analytical OLS becomes computationally prohibitive on large datasets ($O(d^3)$ matrix inversion), loss surfaces, deriving cost gradients with respect to weights and bias, and iterative gradient descent.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_05_Batch_Gradient_Descent_MLR.pdf](02_Worksheets/Worksheet_05_Batch_Gradient_Descent_MLR.pdf) | Deriving gradients, learning rate selection, and iterative weight updates |
| **Solved Notebook** | [Lab_5_Final_BGD_MLR_student_todo.ipynb](04_Notebooks/Lab_05_Batch_Gradient_Descent_MLR/solved/Lab_5_Final_BGD_MLR_student_todo.ipynb) | Solved notebook with 21 execution outputs and `mlr_gradient_descent.py` |
| **Raw Assignment** | [Lab_5 Raw Question](04_Notebooks/Lab_05_Batch_Gradient_Descent_MLR/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 6: Stochastic & Mini-Batch Gradient Descent (SGD / MBGD)
*Overcoming memory limitations of full-batch gradient descent: Mini-batch partitioning, noisy gradients escaping saddle points, epoch vs. step updates, and computational efficiency.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_06_Stochastic_and_MiniBatch_Gradient_Descent.pdf](02_Worksheets/Worksheet_06_Stochastic_and_MiniBatch_Gradient_Descent.pdf) | Stochastic & mini-batch gradient descent derivation, update mechanics, and batch sizing |
| **Interactive Lecture** | [Module_06_Gradient_Descent_Interactive_Lecture.html](03_HTML_Visualizations/Module_06_Gradient_Descent_Interactive_Lecture.html) | Interactive 3D and 2D loss surface visualizer comparing Batch GD, SGD, and Mini-batch |
| **Solved Notebook** | [Lab_6_Final_BGD_MLR_student_todo.ipynb](04_Notebooks/Lab_06_MiniBatch_SGD_MLR/solved/Lab_6_Final_BGD_MLR_student_todo.ipynb) | Solved mini-batch SGD notebook with 16 outputs and `mlr_gradient_descent_minibatch.py` |
| **Raw Assignment** | [Lab_6 Raw Question](04_Notebooks/Lab_06_MiniBatch_SGD_MLR/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 7: Advanced Optimization & End-to-End Problem Solving
*First and second-moment optimizers (Momentum, NAG, AdaGrad, RMSProp, Adam) and applied end-to-end student performance and placement modeling.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_07_Regression_and_Classification_Evaluation_Metrics.pdf](02_Worksheets/Worksheet_07_Regression_and_Classification_Evaluation_Metrics.pdf) | Regression & classification metrics (Residuals, MAE, MAPE, MSE, RMSE, R², Adjusted R², Confusion Matrix, F1) |
| **Interactive Worksheet** | [Module_07_Momentum_and_NAG_Interactive_Worksheet.html](03_HTML_Visualizations/Module_07_Momentum_and_NAG_Interactive_Worksheet.html) | Interactive simulation of EWMA, SGD with Momentum, and Nesterov Accelerated Gradient |
| **Solved Notebook** | [lab_7_St_c.ipynb](04_Notebooks/Lab_07_Student_Performance_and_Placement/solved/lab_7_St_c.ipynb) | Working notebook ready for solving with tracking intact |
| **Raw Assignment** | [Lab_7 Raw Question](04_Notebooks/Lab_07_Student_Performance_and_Placement/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 8: Polynomial Regression (OLS Manual & Gradient Descent)
*Nonlinear feature transformations, higher-degree polynomial feature matrices, closed-form manual OLS implementation vs. gradient descent optimization on polynomials.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_08_Polynomial_Regression_and_Assumptions.pdf](02_Worksheets/Worksheet_08_Polynomial_Regression_and_Assumptions.pdf) | Polynomial regression, nonlinear mappings, overfitting, and the five linear regression assumptions |
| **Solved Notebook (OLS)** | [poly_using_ols_manual.ipynb](04_Notebooks/Lab_08_Polynomial_Regression/solved/poly_using_ols_manual.ipynb) | Manual OLS implementation for polynomial models with `poly_ols.py` |
| **Solved Notebook (GD)** | [Poly_using_gradient_final.ipynb](04_Notebooks/Lab_08_Polynomial_Regression/solved/Poly_using_gradient_final.ipynb) | Gradient descent optimization for polynomial regression with `poly_gradient.py` |
| **Raw Assignment** | [Lab_8 Raw Question](04_Notebooks/Lab_08_Polynomial_Regression/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 9: Bias-Variance Tradeoff & Model Complexity
*Understanding overfitting vs. underfitting across model degrees, decomposing expected prediction error into bias², variance, and irreducible noise, and empirical evaluation.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_09_Bias_Variance_and_Tradeoff.pdf](02_Worksheets/Worksheet_09_Bias_Variance_and_Tradeoff.pdf) | Bias, variance, irreducible error, decomposition derivation, and complexity curves |
| **Solved Notebook** | [Lab_9_Bias_Variance.ipynb](04_Notebooks/Lab_09_Bias_Variance_Tradeoff/solved/Lab_9_Bias_Variance.ipynb) | In-depth bias-variance decomposition notebook with `bias_variance_tools.py` |
| **Dataset** | [academic_outcomes.csv](04_Notebooks/Lab_09_Bias_Variance_Tradeoff/solved/academic_outcomes.csv) | Academic performance dataset for bias-variance complexity curves |
| **Raw Assignment** | [Lab_9 Raw Question](04_Notebooks/Lab_09_Bias_Variance_Tradeoff/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 10: Feature Selection
*Filter methods (variance threshold, correlation analysis), wrapper methods (Recursive Feature Elimination - RFE), and embedded methods for feature relevance and multicollinearity mitigation.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_10_Feature_Selection.pdf](02_Worksheets/Worksheet_10_Feature_Selection.pdf) | Filter, wrapper, and embedded feature selection techniques, correlation heatmaps, and RFE |
| **Solved Notebook** | [AML_Lab_10_Feature_Selection_Student_TODO.ipynb](04_Notebooks/Lab_10_Feature_Selection/solved/AML_Lab_10_Feature_Selection_Student_TODO.ipynb) | Working notebook ready for solving with tracking intact |
| **Dataset** | [student_clean_dataset_2024.csv](04_Notebooks/Lab_10_Feature_Selection/solved/student_clean_dataset_2024.csv) | Cleaned student dataset for feature selection experiments |
| **Raw Assignment** | [Lab_10 Raw Question](04_Notebooks/Lab_10_Feature_Selection/raw/) | Clean unmodified question set directly from upstream GitHub |

---

### Module 11: Dimensionality Reduction & Principal Component Analysis (PCA)
*Curse of dimensionality, geometric projections, variance maximization, covariance matrix eigendecomposition, singular value decomposition (SVD), and scree plots.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_11_Dimensionality_Reduction_and_PCA.pdf](02_Worksheets/Worksheet_11_Dimensionality_Reduction_and_PCA.pdf) | Mathematical foundations of PCA, covariance, eigenvectors, eigenvalue decomposition, and variance explained |
| **Solved Notebook** | [L11_Student_PCA_TODO.ipynb](04_Notebooks/Lab_11_PCA_Dimensionality_Reduction/solved/L11_Student_PCA_TODO.ipynb) | Working notebook ready for solving with tracking intact |
| **Dataset** | [student_clean_dataset.csv](04_Notebooks/Lab_11_PCA_Dimensionality_Reduction/solved/student_clean_dataset.csv) | Student dataset for dimensionality reduction and projection |
| **Raw Assignment** | [Lab_11 Raw Question](04_Notebooks/Lab_11_PCA_Dimensionality_Reduction/raw/) | Clean unmodified question set directly from upstream GitHub |

---

## Environment & Workflow Guide

### 1. Shared Virtual Environment (`.venv`) & Modular Requirements
All labs execute against the unified virtual environment located at `.venv/` using the registered **`Python 3 (AML Course)`** kernel (`aml-env`). 

To prevent version conflicts across different assignments, **each lab folder maintains its own independent `requirements.txt`** containing only the packages needed for that specific module (e.g. `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`).

### 2. Launching JupyterLab
To start JupyterLab with the AML environment pre-configured:
```bash
./scripts/launch_jupyter.sh
```
When opening any notebook, select the **`Python 3 (AML Course)`** kernel (`aml-env`).

### 3. Syncing New Lab Questions from Upstream
When the instructor releases new assignments or questions to the GitHub repository:
```bash
./scripts/sync_labs.sh
```
This script will:
1. Pull the latest commits from `https://github.com/gourabrajak-cloud/aml_lab`.
2. Automatically discover new labs (e.g. `Lab_8`, `Lab_9`, etc.).
3. Set up the dedicated folder inside `04_Notebooks/` with datasets, instructions, starter notebooks, and the lab's specific `requirements.txt`.
