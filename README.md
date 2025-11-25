# Cohort-Analysis-Customer-Churn-Predictions-using-Tree-Methods-
A supervised machine learning project that predicts customer churn using cohort analysis, EDA, and tuned tree-based models to identify key churn drivers.

This project builds a machine learning model to predict **customer churn** using the Telco Customer Churn dataset.  
It includes complete data preprocessing, exploratory data analysis (EDA), cohort analysis, multiple ML models, hyperparameter tuning, and feature importance evaluation.

---

## 🔍 Project Overview
The goal of this project is to analyze customer behavior and predict whether a customer is likely to churn.  
The notebook includes:

- Data cleaning & preprocessing  
- Cohort analysis (Tenure Cohort)  
- Exploratory Data Analysis (visual insights)  
- Model building with tree-based ML algorithms  
- Hyperparameter tuning using GridSearchCV  
- Performance evaluation with confusion matrices & classification reports  
- Feature importance interpretation  

---

## 📁 Dataset

The dataset used in this project is the **Telco Customer Churn Dataset**, obtained from Kaggle.

### **📌 Dataset Source**
Kaggle Link:  
https://www.kaggle.com/code/danishmubashar/telco-customer-churn-97-acc/input

### **Dataset Includes:**

- Customer demographics  
- Account and contract details  
- Services subscribed  
- Monthly and total charges  
- Tenure  
- Churn status (target variable)

Target variable: **Churn (Yes/No)**

---

## 🧹 Data Preprocessing
Key steps:

- Handle missing values  
- Convert categorical features  
- Fix incorrect data types (`TotalCharges`)  
- Create Tenure Cohort  
- Train/Validation/Test split  
- Encode categorical columns where needed  

---

## 📊 Exploratory Data Analysis (EDA)
The notebook includes multiple visualizations such as:

- Countplots  
- Boxplots  
- Violin plots  
- Scatterplots  
- Cohort-wise histograms  
- Correlation barplots  
- Contract type vs Churn comparison  
- MonthlyCharges vs TotalCharges analysis  

These help reveal:

- Contract type strongly affects churn  
- High MonthlyCharges correlates with churn  
- Low TotalCharges (short tenure) customers churn more  
- Electronic check users have highest churn rate  

---

## 🤖 Machine Learning Models
The following models were trained:

- Decision Tree Classifier  
- Random Forest Classifier  
- AdaBoost Classifier  
- Gradient Boosting Classifier  

Each model was evaluated using:

- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  
- Classification Report  

`plot_confusion_matrix` was replaced with `ConfusionMatrixDisplay` due to scikit-learn deprecation.

---

## 🔧 Hyperparameter Tuning
GridSearchCV was used to tune:

### Random Forest
- `n_estimators`: multiple ranges  
- `max_features`: `sqrt`, `log2` (removed deprecated `auto`)  

### AdaBoost
- Tuned `n_estimators`  
- Used `cv=3` and `n_jobs=-1` for speed  

### Gradient Boosting
- Tuned `n_estimators`  

---

## 📈 Feature Importance
Tree-based models were used to extract the most important churn predictors:

- Tenure  
- MonthlyCharges  
- TotalCharges  
- Contract Type  
- InternetService  
- PaymentMethod  

A barplot visualizes the most influential features.

---

## ⚠️ Problems Faced
Only major issues:

1. **`plot_confusion_matrix` deprecated**  
   - Replaced with `ConfusionMatrixDisplay`.

2. **RandomForest parameter `'auto'` removed**  
   - Caused GridSearchCV failures; fixed by removing `"auto"`.

3. **`y_train.ravel()` FutureWarning**  
   - Replaced with `y_train.to_numpy()`.

4. **Plot labels turned black due to Matplotlib 3.9+ changes**  
   - Fixed by manually setting `color='black'` for axis labels/ticks.

5. **AdaBoost & GradientBoosting GridSearch took too long**  
   - Fixed with reduced parameter ranges, fewer CV folds, `n_jobs=-1`.

6. **Incorrect use of `confusion_matrix(model, X, y)`**  
   - Correct format used: `confusion_matrix(y_true, model.predict(X))`.

---

## 🛠 Tech Stack
- Python  
- Pandas  
- NumPy  
- Seaborn  
- Matplotlib  
- Scikit-learn  
- Google Colab  

---

## 🚀 Conclusion
This project demonstrates a complete end-to-end churn prediction workflow.  
Tree-based models (especially Random Forest and Gradient Boosting) deliver strong performance, and the analysis highlights key drivers behind customer churn.

---

## 📂 Notebook
Main notebook: **Customer_Churn_Predictions.ipynb**

---

**Ankit Kumar Upadhyay**  
