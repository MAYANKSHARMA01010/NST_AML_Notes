# Advanced Machine Learning Course Repository

This repository contains all organized lecture notes, worksheets, interactive HTML visualizations, and hands-on lab notebooks for the **Advanced Machine Learning (AML) Course (Semester 5)**.

All materials are categorized into dedicated folders and cross-referenced module-by-module. Furthermore, all lab exercises share a single unified virtual environment (`.venv`), with dedicated folder structures for datasets, student solutions, and instructor reference keys.

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
├── 02_Worksheets/                     # Theoretical & practical worksheets (Lectures 00 to 06)
│   ├── Worksheet_00_Foundations_From_Code_to_Learning.pdf
│   ├── Worksheet_01_ML_Project_Lifecycle_EDA_and_Preprocessing.pdf
│   ├── Worksheet_02_ML_Project_Lifecycle_Feature_Engineering_and_Evaluation.pdf
│   ├── Worksheet_03_Simple_Linear_Regression_OLS_Teacher_Copy.pdf
│   ├── Worksheet_04_Multiple_Linear_Regression_OLS.pdf
│   ├── Worksheet_05_Batch_Gradient_Descent_MLR.pdf
│   └── Worksheet_06_Optimizers_With_Memory.pdf
├── 03_HTML_Visualizations/            # Interactive browser-based learning explorables
│   ├── Module_06_Gradient_Descent_Interactive_Lecture.html
│   └── Module_07_Momentum_and_NAG_Interactive_Worksheet.html
├── 04_Notebooks/                      # Hands-on Jupyter lab assignments (Labs 01 to 07)
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
│   └── Lab_07_Student_Performance_and_Placement/
│       ├── raw/                       # Exact question from GitHub (untouched)
│       │   ├── lab_7_St_c.ipynb
│       │   └── student_raw_dataset.csv
│       └── solved/                    # Ready to solve in Jupyter with tracking
│           ├── .ipynb_checkpoints/
│           ├── lab_7_St_c.ipynb
│           └── student_raw_dataset.csv
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
| **Student Solution** | [Lab_03_student_notebook.ipynb](04_Notebooks/Lab_03_Simple_Linear_Regression_OLS/Student_Solution/Lab_03_student_notebook.ipynb) | Student implementation of `slr_ols_todo.py` and regression experiments |
| **Solved Reference** | [Lab_03 Solved Reference](04_Notebooks/Lab_03_Simple_Linear_Regression_OLS/Solved_Reference/) | Instructor's reference implementation and completed notebook |

---

### Module 4: Multiple Linear Regression (MLR) via OLS
*Extending linear regression to multidimensional feature spaces: matrix formulation $\hat{\beta} = (X^T X)^{-1} X^T y$, assumptions of linear regression, multicollinearity, and parameter interpretation.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_04_Multiple_Linear_Regression_OLS.pdf](02_Worksheets/Worksheet_04_Multiple_Linear_Regression_OLS.pdf) | Matrix formulation, normal equations, and multiple feature interactions |
| **Student Solution** | [Lab_04_student_notebook.ipynb](04_Notebooks/Lab_04_Multiple_Linear_Regression_OLS/Student_Solution/Lab_04_student_notebook.ipynb) | Student implementation of `mlr_ols_todo.py` for multivariate regression |
| **Solved Reference** | [Lab_04_Complete.ipynb](04_Notebooks/Lab_04_Multiple_Linear_Regression_OLS/Solved_Reference/Lab_4_Complete.ipynb) | Instructor's complete multivariate regression solution and code |

---

### Module 5: Batch Gradient Descent (BGD) for MLR
*Why analytical OLS becomes computationally prohibitive on large datasets ($O(d^3)$ matrix inversion), loss surfaces, deriving cost gradients with respect to weights and bias, and iterative gradient descent.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Worksheets** | [Worksheet_05_Batch_Gradient_Descent_MLR.pdf](02_Worksheets/Worksheet_05_Batch_Gradient_Descent_MLR.pdf) | Deriving gradients, learning rate selection, and iterative weight updates |
| **Student Solution** | [Lab_05_Final_BGD_MLR_student_todo.ipynb](04_Notebooks/Lab_05_Batch_Gradient_Descent_MLR/Student_Solution/Lab_05_Final_BGD_MLR_student_todo.ipynb) | Student implementation of `mlr_gradient_descent.py` and convergence checks |
| **Solved Reference** | [Lab_05 Solved Reference](04_Notebooks/Lab_05_Batch_Gradient_Descent_MLR/Solved_Reference/) | Instructor's verified gradient descent code and convergence graphs |

---

### Module 6: Stochastic & Mini-Batch Gradient Descent (SGD / MBGD)
*Overcoming memory limitations of full-batch gradient descent: Mini-batch partitioning, noisy gradients escaping saddle points, epoch vs. step updates, and computational efficiency.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Interactive Lecture** | [Module_06_Gradient_Descent_Interactive_Lecture.html](03_HTML_Visualizations/Module_06_Gradient_Descent_Interactive_Lecture.html) | Interactive 3D and 2D loss surface visualizer comparing Batch GD, SGD, and Mini-batch |
| **Worksheets** | [Worksheet_06_Optimizers_With_Memory.pdf](02_Worksheets/Worksheet_06_Optimizers_With_Memory.pdf) | Optimization worksheet covering memory, momentum, and adaptive rates |
| **Student Solution** | [Lab_06_Final_BGD_MLR_student_todo.ipynb](04_Notebooks/Lab_06_MiniBatch_SGD_MLR/Student_Solution/Lab_06_Final_BGD_MLR_student_todo.ipynb) | Student solution for mini-batch SGD implementation (`mlr_gradient_descent_minibatch.py`) |
| **Template** | [Lab_06 Template](04_Notebooks/Lab_06_MiniBatch_SGD_MLR/Template/) | Clean starter code and dataflow architecture diagram (`dfd.png`) |

---

### Module 7: Advanced Optimization & End-to-End Problem Solving
*First and second-moment optimizers (Momentum, NAG, AdaGrad, RMSProp, Adam) and applied end-to-end student performance and placement modeling.*

| Category | File | Description |
| :--- | :--- | :--- |
| **Interactive Worksheet** | [Module_07_Momentum_and_NAG_Interactive_Worksheet.html](03_HTML_Visualizations/Module_07_Momentum_and_NAG_Interactive_Worksheet.html) | Interactive simulation of EWMA, SGD with Momentum, and Nesterov Accelerated Gradient |
| **Notebooks** | [Lab_07_Student_Performance_and_Placement.ipynb](04_Notebooks/Lab_07_Student_Performance_and_Placement/Lab_07_Student_Performance_and_Placement.ipynb) | End-to-end story notebook analyzing student factors, performance metrics, and placement |

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
