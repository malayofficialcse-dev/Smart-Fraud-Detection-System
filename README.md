# 💳 Credit Card Fraud Detection using Machine Learning

![Fraud Detection Banner](./assets/fraud-detection-banner.png)

A complete Machine Learning project for detecting fraudulent credit card transactions using the Random Forest Classifier. This project focuses on handling highly imbalanced datasets, analyzing transaction behavior, and evaluating model performance using multiple classification metrics.

---

# 📌 Project Overview

Credit card fraud has become one of the biggest challenges in digital financial systems. The objective of this project is to build an intelligent fraud detection model capable of identifying suspicious transactions with high accuracy and low false-positive rates.

The project uses:

- Python
- Pandas & NumPy for data processing
- Matplotlib & Seaborn for visualization
- Scikit-learn for machine learning

The model is trained on the popular Kaggle Credit Card Fraud Detection dataset.

---

# 🚀 Workflow of the Project

## 1️⃣ Importing Required Libraries

The project begins by importing important Python libraries for:

- Data handling
- Data visualization
- Machine learning
- Model evaluation

### Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
```

---

## 2️⃣ Loading the Dataset

The dataset is loaded using Pandas:

```python
data = pd.read_csv("creditcard.csv")
```

The dataset contains anonymized transaction features (`V1` to `V28`), along with:

- `Time`
- `Amount`
- `Class`

Where:

- `Class = 0` → Valid Transaction
- `Class = 1` → Fraudulent Transaction

---

## 3️⃣ Exploratory Data Analysis (EDA)

The project performs detailed data analysis to understand the dataset structure.

### Dataset Preview

```python
print(data.head())
```

### Summary Statistics

```python
print(data.describe())
```

### Fraud vs Valid Transactions

The dataset is highly imbalanced:

- Very few fraud cases
- Majority are valid transactions

This imbalance is one of the biggest challenges in fraud detection systems.

---

# 📊 Data Visualization

Several visualizations are generated to better understand the dataset.

---

## 🔥 Correlation Heatmap

A correlation matrix is created using Seaborn:

```python
sns.heatmap(corrmat)
```

### Purpose

- Identify feature relationships
- Detect highly correlated variables
- Understand data patterns

### Preview

![Correlation Matrix](./assets/correlation-matrix.png)

---

## 📉 Confusion Matrix

Used to evaluate prediction performance.

It shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

This helps analyze how effectively the model identifies fraud transactions.

### Preview

![Confusion Matrix](./assets/confusion-matrix.png)

---

## 📈 ROC Curve

The ROC Curve measures classification performance across thresholds.

The project calculates:

- False Positive Rate (FPR)
- True Positive Rate (TPR)
- ROC-AUC Score

A higher ROC-AUC score indicates a better model.

### Preview

![ROC Curve](./assets/roc-curve.png)

---

## 📌 Feature Importance Graph

Random Forest provides feature importance scores.

Top important features are visualized using a bar chart to understand which variables contribute most to fraud detection.

### Preview

![Feature Importance](./assets/feature-importance.png)

---

# ⚙️ Data Preprocessing

## Feature & Target Separation

```python
X = data.drop(["Class"], axis=1)
y = data["Class"]
```

---

## Feature Scaling

Standardization is applied using:

```python
StandardScaler()
```

### Why Scaling is Important

- Improves model performance
- Normalizes feature values
- Prevents dominance of larger numerical ranges

---

## Train-Test Split

The dataset is split into:

- 80% Training Data
- 20% Testing Data

Using:

```python
train_test_split()
```

Stratified splitting ensures proper fraud class distribution.

---

# 🤖 Machine Learning Model

## Random Forest Classifier

The project uses:

```python
RandomForestClassifier()
```

### Why Random Forest?

- Handles large datasets efficiently
- Works well with imbalanced data
- Reduces overfitting
- Provides feature importance

### Model Parameters

```python
n_estimators=100
class_weight="balanced"
random_state=42
```

---

# 📊 Model Evaluation Metrics

The model performance is evaluated using multiple metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Precision | Fraud prediction reliability |
| Recall | Ability to detect fraud |
| F1-Score | Balance between precision & recall |
| MCC | Balanced evaluation metric |
| ROC-AUC | Overall classification quality |

---

# 🧠 Key Learnings from the Project

✅ Handling Imbalanced Datasets  
✅ Data Visualization Techniques  
✅ Feature Scaling  
✅ Random Forest Classification  
✅ Fraud Detection Systems  
✅ Model Evaluation Techniques  
✅ Confusion Matrix Analysis  
✅ ROC Curve Analysis  

---

# 📂 Technologies Used

| Technology | Usage |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |

---

# 📷 Project Visualization

The project includes:

- Correlation Heatmap
- ROC Curve
- Confusion Matrix
- Feature Importance Graph
- Fraud Distribution Analysis

These visualizations help explain the model behavior clearly.

---

# 📈 Sample Output

```text
Accuracy: 0.9991
Precision: 0.9326
Recall: 0.8673
F1-Score: 0.8984
ROC AUC: 0.9775
```

---

# 🎯 Conclusion

The Random Forest model performs exceptionally well in detecting fraudulent credit card transactions. Despite the dataset being highly imbalanced, the model achieves strong performance by effectively distinguishing between normal and fraudulent transactions.

This project demonstrates how Machine Learning can be applied in real-world financial security systems to reduce fraud and improve transaction safety.

---

# ⭐ Future Improvements

- Implement Deep Learning models
- Use XGBoost or LightGBM
- Deploy using Flask/Django
- Real-time fraud detection system
- Streamlit Dashboard Integration
- Hyperparameter Optimization

---

# 🛠️ Installation & Setup

## Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
```

## Navigate to Project Folder

```bash
cd credit-card-fraud-detection
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python fraud_detection.py
```

---

# 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── creditcard.csv
├── fraud_detection.py
├── README.md
├── requirements.txt
│
├── assets/
│   ├── fraud-detection-banner.png
│   ├── correlation-matrix.png
│   ├── confusion-matrix.png
│   ├── roc-curve.png
│   └── feature-importance.png
```

---

# 👨‍💻 Author

## Malay Maity

Machine Learning & Full Stack Developer

- Passionate about AI & Data Science
- Interested in Cyber Security & Fraud Detection
- Exploring Deep Learning & Real-Time ML Systems

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---