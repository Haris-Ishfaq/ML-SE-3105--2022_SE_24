# Survival Prediction Post-HCT Using Machine Learning

<p align="center">
  <strong>The University of Azad Jammu and Kashmir, Muzaffarabad</strong><br>
  <em>Department of Software Engineering</em><br>
  <em>Bachelor of Science in Software Engineering (2022-2026)</em>
</p>

---

## 📚 Course Details  

**Course Title:** Machine Learning  
**Course Code:** SE-3105  
**Instructor:** Engr. Ahmed Khawaja  
**Semester:** Fall 2024  
**Session:** 2022 – 2026  

## 📝 Submission Details  

**Submitted By:**  
**Roll Numbers:** 2022-SE-19, 2022-SE-21, 2022-SE-24  
**Degree Program:** BS Software Engineering  
**Submitted To:** Engr. Ahmed Khawaja  
**Date:** March 3, 2025  

---

## 📖 Overview
This project implements an **ensemble learning approach** to improve predictive accuracy using multiple machine learning models. The workflow includes **data preprocessing, model training, ensemble learning, and inference**. The final trained model is evaluated for accuracy and deployed for real-world predictions.

### 📂 Project Files
- **Training Code:** `Training Code.ipynb` - Preprocesses data and trains machine learning models.
- **Inference Code:** `Inference Code.ipynb` - Loads trained models and makes predictions.
- **Ensemble Model:** `ensemble_models.pkl` - Stores the trained ensemble classifier for inference.

### 📊 Dataset
- **Source:**  Provided dataset (`train.csv`, `test.csv`) .  
- **Size:** Large-scale dataset.  
- **Target Variable:** `efs` (Binary: `0 = event`, `1 = survival`). 
- ***Key Features:*** 
List of essential dataset attributes used for model training.  


---

## 🚀 Project Workflow
This project follows five key steps:

- **Data Preprocessing**  
- **Model Training**  
- **Ensemble Model Creation**  
- **Model Evaluation**  
- **Inference & Predictions**  

### 🔹 Data Preprocessing
✔ Handle missing values and normalize data.  
✔ Encode categorical variables.  
✔ Split dataset into training and testing sets.  

```python
import pandas as pd  

# Load dataset
df = pd.read_csv(train_file_path)

test_file_path = "test.csv"  
df_test = pd.read_csv(test_file_path)  
```


```

---

### 🔹 Model Training
✔ Train individual machine learning models.  
✔ Optimize hyperparameters for improved accuracy.  

```python
# Training a RandomForest model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
import joblib
joblib.dump(model, "trained_model.pkl")
```

---

### 🔹 Ensemble Learning
✔ Combine multiple models to enhance accuracy.  
✔ Use techniques like **VotingClassifier** for aggregation.  

```python
# Implementing an ensemble model
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define individual models
model1 = LogisticRegression()
model2 = RandomForestClassifier(n_estimators=100)
model3 = SVC(probability=True)

# Create an ensemble model
ensemble_model = VotingClassifier(estimators=[('lr', model1), ('rf', model2), ('svc', model3)], voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Save the ensemble model
joblib.dump(ensemble_model, "ensemble_model.pkl")
```

---

### 🔹 Model Evaluation
✔ Evaluate performance using metrics like **accuracy, precision, recall, and confusion matrix**.  

```python
# Model evaluation
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

---

### 🔹 Inference & Predictions
✔ Load trained models and make predictions on new data.  

```python
# Load trained model and make predictions
import numpy as np

# Load the model
model = joblib.load("trained_model.pkl")

# Example input for prediction
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Adjust based on dataset

# Make a prediction
prediction = model.predict(sample_input)
print("Predicted Class:", prediction)
```

---

## 🔧 Improvements & Future Work

**1. Data Enhancement**  
✔ Collect more diverse data to improve generalization.  
✔ Perform advanced feature engineering.  

**2. Model Optimization**  
✔ Experiment with different ensemble techniques like **stacking and boosting**.  
✔ Fine-tune hyperparameters for better performance.  

**3. Performance Evaluation**  
✔ Use advanced metrics like **ROC-AUC and F1-score**.  
✔ Implement **cross-validation** for better model validation.  

**4. Deployment & Automation**  
✔ Deploy the model as an **API** for real-time predictions.  
✔ Automate the **training and inference pipeline** for efficiency.  

By implementing these improvements, the model can become **more accurate, efficient, and scalable** for real-world applications. 🚀  

---
## 📁 Files in This Repository  
- `train_efs_model.py` → The main script for training the model.  
- `README.md` → This file with workflow details.  
- `/preprocessor/` → Folder containing saved preprocessing objects (imputer, encoder, scaler).  
- `efs_model.pth` → Saved PyTorch model (after training). 

---


## 🛠️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Train the Model
```bash
jupyter notebook "Training Code.ipynb"
```
### 3️⃣ Run Inference
```bash
jupyter notebook "Inference Code.ipynb"
```
### 4️⃣ Modify and Experiment  
Try different datasets, tweak hyperparameters, or change the models for further improvement.


