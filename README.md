
# 🧠 Employee Insurance Enrollment Prediction

**Goal**: Predict whether an employee will opt into a new voluntary insurance product using census-style employment data.

---

## 📁 Dataset Overview

The dataset contains **10,000 synthetic employee records** with demographic and employment-related features.

| Column             | Description                                  |
|--------------------|----------------------------------------------|
| `employee_id`      | Unique ID (excluded from modeling)           |
| `age`              | Age of the employee                          |
| `gender`           | Gender (Male, Female, Other)                 |
| `marital_status`   | Marital status                               |
| `salary`           | Annual salary (continuous)                   |
| `employment_type`  | Type of employment (Full-time, Part-time...) |
| `region`           | Work region (e.g., South, West)              |
| `has_dependents`   | Whether employee has dependents (Yes/No)     |
| `tenure_years`     | Years at current job                         |
| `enrolled`         | Target: 1 = Enrolled, 0 = Not Enrolled       |

---

## 🔍 Detailed Data Observations

### ✅ Clean Data
- **No missing values**
- All columns had valid data types
- No severe outliers after clipping extreme `tenure_years`

### 📊 Class Imbalance
- **61.74%** of employees enrolled (`enrolled=1`)
- **38.26%** did not enroll (`enrolled=0`)
- Mitigated during modeling with:
  - `class_weight='balanced'` in classifiers
  - Stratified train-validation-test splits

### 📈 Feature Relationships
- Employees with **dependents**, **higher salaries**, and **full-time employment** had a higher likelihood of enrollment
- **Tenure** had a moderate effect — more years at the company generally increased enrollment probability
- Regions showed some variation in enrollment rates

### 🔗 Correlations
- `salary` and `enrolled`: **0.36**
- `age` and `enrolled`: **0.27**
- `tenure_years` and `enrolled`: slightly negative, but interaction features were more useful

---

## 🔍 Overfitting Mitigation

### Problem:
- Initial models (Random Forest & Gradient Boosting) achieved **100% accuracy**, indicating **overfitting**, especially on synthetic data

### Solutions Implemented:

1. **Reduced Feature Set**:  
   Used only the **top 8 most important features**, selected using Random Forest importance scores.

2. **Simplified Models**:  
   - Random Forest: `max_depth=30`, `n_estimators=100`  
   - Gradient Boosting: `max_depth=5`, `n_estimators=100`  
   Reducing depth and tree count helped improve generalizability.

3. **Cross-validation**:  
   Used `RandomizedSearchCV` with 5-fold CV to avoid overfitting on a single train-test split.

---

## 🔝 Top 8 Features Used

- `age`
- `salary`
- `tenure_years`
- `employment_type_Full-time`
- `region_Northeast`
- `marital_status_Married`
- `gender_Male`
- `has_dependents_Yes`

---

## 🤖 Model Development

### Models Tried:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Final Choice: **Random Forest**

**Rationale**:
- Robust to noisy features
- Handles categorical and numerical data well
- Supports feature importance
- Works well with `class_weight='balanced'` for imbalance handling

---

## 🧪 Evaluation Strategy

### Data Splits:
- Train: 70%
- Validation: 15%
- Test: 15%
- Stratified sampling to preserve class ratio

### Metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC

---

## 🎯 Model Performance (Top 8 Features)

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 99.93%  |
| Precision   | 99.89%  |
| Recall      | 100.00% |
| F1-Score    | 99.95%  |
| ROC AUC     | 100.00% |

---

## ⚙️ Model Saving

Saved with `joblib`:

- `best_random_forest_model.pkl`
- `best_random_forest_model_top8.pkl`
- `best_gradient_boosting_model.pkl`

---

## 🔬 API with Flask

### Start API:
```bash
python app.py
```

### Endpoint:
```http
POST /predict
```

### Sample Payload:
```json
{
  "age": 30,
  "salary": 50000,
  "tenure_years": 2,
  "gender_Male": 1,
  "marital_status_Married": 1,
  "employment_type_Full-time": 1,
  "region_Northeast": 1,
  "has_dependents_Yes": 0,
  "model": "random_forest"
}
```

### Response:
```json
{
  "prediction": 1,
  "probability": 0.9819
}
```

---

## ✅ Other Features Implemented

- ✅ Randomized Hyperparameter Tuning
- ✅ Overfitting Mitigation
- ✅ Feature Selection via Feature Importances
- ✅ Flask REST API for real-time inference
- ✅ Data Imbalance Handling

---

## 🛠️ How to Run

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/insurance-enrollment-predictor.git
cd insurance-enrollment-predictor
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Run Notebook (Training)
```bash
jupyter notebook assignment.ipynb
```

### 4. Start the Flask API
```bash
python app.py
```

---

## 📁 Project Structure

```
.
├── assignment.ipynb                  # Full notebook with analysis and model training
├── employee_data.csv                 # Input dataset
├── best_random_forest_model.pkl      # Full model (all features)
├── best_gradient_boosting_model.pkl  # Gradient boosting model
├── app.py                            # Flask API
├── requirements.txt
└── README.md                         # This file 📘
```


---

## 👨‍💻 Author

Ankit Chauhan
