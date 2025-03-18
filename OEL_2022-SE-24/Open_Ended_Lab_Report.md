# MNIST Classification Report

## Introduction
The MNIST dataset consists of handwritten digits (0-9), represented as 28x28 grayscale images. These images are flattened into 784-dimensional vectors with pixel intensities ranging from 0 to 255. The dataset is pre-split into a training set (60,000 samples) and a testing set (10,000 samples). The objective of this lab is to implement and evaluate multiple classification models to determine their effectiveness in recognizing handwritten digits.

## Dataset Preparation
### **Loading the Data**
- The dataset is loaded from CSV files (`mnist_train.csv` and `mnist_test.csv`) into Pandas DataFrames.
- The training dataset contains 60,000 rows and 785 columns (784 features + 1 label column).
- The testing dataset contains 10,000 rows and 785 columns.

### **Feature and Label Separation**
- `X_train`: 60,000 samples × 784 features (pixel values).
- `y_train`: 60,000 labels.
- `X_test`: 10,000 samples × 784 features.
- `y_test`: 10,000 labels.

### **Data Preprocessing**
1. **Handling Missing Values:** Missing values were checked, and appropriate strategies such as mean imputation were applied.
2. **Normalization:** Pixel values were scaled using `StandardScaler` to improve model performance.
3. **Data Splitting:** The dataset was divided into training and testing sets as provided.

## Models Used
### **Logistic Regression**  
- A linear model that estimates class probabilities using the logistic function.
- Works well for linearly separable data and serves as a baseline model.
- Fast and computationally efficient.

### **K-Nearest Neighbors (KNN)**  
- A non-parametric, instance-based learning algorithm.
- Classifies new data points based on the majority vote of their k-nearest neighbors.
- Effective for pattern recognition but computationally expensive for large datasets.

### **Naïve Bayes (GaussianNB)**  
- A probabilistic classifier based on Bayes' theorem.
- Assumes feature independence, which may not be ideal for MNIST but still provides a useful comparison.
- Computationally efficient and works well with small datasets.

## Model Training and Hyperparameter Tuning
- KNN was optimized using `GridSearchCV` to find the best value for `n_neighbors`.
- Logistic Regression and Naïve Bayes were trained using default settings, as they perform well with minimal tuning.

## Model Evaluation Metrics
To assess model performance, the following metrics were used:
- **Accuracy:** Measures the percentage of correctly classified instances.
- **Precision:** The proportion of true positive predictions among all predicted positives.
- **Recall:** The proportion of true positive predictions among actual positives.
- **F1-Score:** The harmonic mean of precision and recall, providing a balanced metric.
- **Confusion Matrix:** Provides insight into misclassification patterns.

## Results
1. **Logistic Regression**  
   - Accuracy: **92.5%**  
   - Classification Report:
     - Precision: Ranges from **87% to 97%**
     - Recall: Ranges from **87% to 98%**
     - F1-score: Averages around **92%**
   - Confusion Matrix: Shows minor misclassification errors, particularly between similar digits (e.g., 3 and 5).

2. **K-Nearest Neighbors (KNN)**  
   - Accuracy: **96.9%**  
   - Classification Report:
     - Precision: Ranges from **95% to 99%**
     - Recall: Ranges from **94% to 100%**
     - F1-score: Averages around **97%**
   - Confusion Matrix: Shows improved classification with fewer misclassifications compared to Logistic Regression.

3. **Naïve Bayes (GaussianNB)**  
   - Accuracy: **83.4%**  
   - Classification Report:
     - Precision: Ranges from **80% to 89%**
     - Recall: Ranges from **79% to 88%**
     - F1-score: Averages around **83%**
   - Confusion Matrix: Shows significant misclassification due to the assumption of feature independence.

## Model Accuracy Comparison

  ![image](https://github.com/user-attachments/assets/12d2ed69-b1cc-4bb1-8c07-9da52a3862a5)


## Conclusion
This study evaluated three classification models on the MNIST dataset. The key findings are:
- **Logistic Regression** is a strong baseline but struggles with complex digit patterns.
- **KNN** performed the best, capturing non-linear patterns effectively but at a higher computational cost.
- **Naïve Bayes** showed lower accuracy due to its assumption of feature independence, making it less effective for digit classification.

Overall, KNN emerged as the best performer for this dataset, while Logistic Regression provided a fast and efficient baseline. Future improvements could include using deep learning models such as Convolutional Neural Networks (CNNs) for even higher accuracy.

