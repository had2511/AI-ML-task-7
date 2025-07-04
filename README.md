# AI-ML-task-7
# 🧠 Breast Cancer Classification using SVM

This project demonstrates a machine learning pipeline to classify breast cancer as **benign** or **malignant** using the Breast Cancer Wisconsin dataset. We explore **Support Vector Machines (SVM)** with **linear** and **RBF** kernels, visualize decision boundaries, perform **hyperparameter tuning**, and evaluate the model using **cross-validation**.

---

## 📁 Dataset

- **Source:** [Kaggle - Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Features:** 30 numeric features (mean, standard error, and worst of tumor measurements)
- **Target:** Diagnosis (Malignant = 1, Benign = 0)

---

## ✅ Project Steps

### 1. Data Loading & Preprocessing
- Upload and load the dataset in Google Colab.
- Drop unnecessary columns (like ID, unnamed).
- Encode the target column (`diagnosis`) as binary.
- Standardize features using `StandardScaler`.

### 2. Model Training
- Train Support Vector Machine with:
  - `linear` kernel
  - `rbf` kernel

### 3. 2D Visualization
- Select two features (`radius_mean` and `texture_mean`).
- Visualize decision boundary of linear SVM in 2D space.

### 4. Hyperparameter Tuning
- Use `GridSearchCV` to tune `C` and `gamma` for the RBF kernel.
- Evaluate the best model.

### 5. Cross-Validation
- Use 5-fold cross-validation to assess model generalization.

---

## 📊 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🧪 How to Run

1. Upload the dataset to Colab using:

   ```python
   from google.colab import files
   files.upload()
