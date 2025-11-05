# Loan Approval Prediction with Neural Networks

This project builds an end-to-end **loan approval prediction** pipeline using a feed-forward neural network.  
Given historical loan application data, the model predicts whether a loan should be **approved (1)** or **rejected (0)**.

Main file: `LoanApprovalCode.ipynb`

---

## 📂 Dataset

- **File used:** `loan_data.csv`
- **Target column:** `loan_status` (0 = reject, 1 = approve)
- **Feature types:**
  - **Categorical:**
    - `person_gender`
    - `person_education`
    - `person_home_ownership`
    - `loan_intent`
    - `previous_loan_defaults_on_file`
  - **Numerical:**
    - All remaining columns (age, income, employment experience, loan amount, credit score, etc.)

> Place `loan_data.csv` in the same directory as the notebook, or update the path in the first cell.

---

## 🔄 Data Preprocessing Pipeline

1. **Train / Val / Test Split**
   - Stratified split to maintain the `loan_status` distribution.
   - Final split:
     - **Train:** 60%
     - **Validation:** 20%
     - **Test:** 20%

2. **Feature Transformation**
   - Uses `ColumnTransformer` from `scikit-learn`:
     - `MinMaxScaler` → numerical features  
     - `OneHotEncoder(drop='first')` → categorical features  
   - Output is a fully numeric feature matrix suitable for neural networks.

3. **Class Imbalance Handling**
   - Computed class weights using `class_weight.compute_class_weight`.
   - Learned weights:
     - Class **0** (non-approved): **0.6429**
     - Class **1** (approved): **2.25**
   - Passed to `model.fit(..., class_weight=class_weights)` to counter imbalance.

---

## 🧠 Baseline Neural Network Model

### Architecture

Implemented in Keras (`tf.keras`):

- `Dense(64, activation='relu', input_shape=(n_features,))`
- `Dropout(0.3)`
- `Dense(32, activation='relu')`
- `Dropout(0.3)`
- `Dense(1, activation='sigmoid')`  → binary output (approval probability)

**Total parameters:** 3,585 (all trainable)

### Training Configuration

- **Loss:** `binary_crossentropy`
- **Optimizer:** `Adam`
- **Metric:** `accuracy`
- **Batch size:** `32`
- **Max epochs:** `50`
- **Callbacks:**
  - `EarlyStopping`  
    - `monitor='val_loss'`  
    - `patience=5`  
    - `restore_best_weights=True`
  - `ModelCheckpoint`  
    - Saves `best_model.h5` based on lowest `val_loss`.

---

## 📊 Baseline Model Results (Test Set)

After training with early stopping and class weights:

- **✅ Test Accuracy:** `0.8886` (≈ 88.86%)

### Confusion Matrix

\[
\begin{bmatrix}
6214 & 786 \\
217 & 1783
\end{bmatrix}
\]

- True Negative (0 → 0): **6214**
- False Positive (0 → 1): **786**
- False Negative (1 → 0): **217**
- True Positive (1 → 1): **1783**

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| **Accuracy** | — | — | **0.89** | 9000 |
| **Macro avg** | 0.83 | 0.89 | 0.85 | 9000 |
| **Weighted avg** | 0.91 | 0.89 | 0.89 | 9000 |

**Interpretation:**

- The model is very strong at identifying **non-approved** loans (high precision & recall for class 0).
- For the **approved** class (1), recall is high (0.89) — it catches most approved loans — but precision is lower (0.69), meaning more false approvals.

---

## 🧪 Model Interpretability (LIME)

- Used **LIME (Local Interpretable Model-agnostic Explanations)**:
  - `LimeTabularExplainer` on the transformed training data.
  - Feature names from `preprocessor.get_feature_names_out()`.
- For random test instances, LIME shows:
  - Which features **push the prediction toward approval**.
  - Which features **push toward rejection**.
- This helps explain individual loan decisions and builds trust with stakeholders.

---

## 🎯 Hyperparameter Tuning with Keras Tuner

Tuning is done using a custom `LoanApprovalHyperModel` and **three** search strategies:
- **Random Search**
- **Bayesian Optimization**
- **Hyperband**

### Hypermodel Search Space (Proper Hyperparameters)

For each trial, Keras Tuner selects from these hyperparameter ranges:

- **First hidden layer**
  - Units: `units_1 ∈ {32, 64, 96, ..., 256}` (step 32)
  - Dropout: `dropout_1 ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}`

- **Second hidden layer**
  - Units: `units_2 ∈ {16, 32, 48, ..., 128}` (step 16)
  - Dropout: `dropout_2 ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}`

- **Output**
  - `Dense(1, activation='sigmoid')`

- **Optimizer & Learning Rate**
  - Optimizer: `Adam`
  - Learning rate: `learning_rate ∈ {1e-2, 1e-3, 1e-4}`

> The **exact best combination** (chosen units, dropouts, and LR) for each tuner is stored in  
> `tuner_random.get_best_hyperparameters(1)[0].values` (and similarly for the others) inside the notebook.  
> You can print those dictionaries in a new cell if you want them directly in the README.

### Tuning Setup

Common tuning settings:

- `max_trials = 5`
- `executions_per_trial = 2`
- `epochs = 20`
- `validation_split = 0.2`

Search methods:

- `kt.RandomSearch(...)`
- `kt.BayesianOptimization(...)`
- `kt.Hyperband(...)`

Each tuner:
1. Searches over the hyperparameter space using the defined strategy.
2. Selects the best hyperparameters based on **validation accuracy** (`objective='val_accuracy'`).
3. Returns the best model via `get_best_models(1)[0]`.

### 📈 Tuned Model Performance (Test Set)

Best models from each tuning method were evaluated on the **same test set**:

| Tuning Method           | Test Accuracy |
|-------------------------|--------------:|
| Random Search           | **0.9157** |
| Bayesian Optimization   | **0.9178** |
| Hyperband               | **0.9186** |

- The **best-performing model** comes from **Hyperband**, with ≈ **91.86%** test accuracy.
- All tuned models improved significantly over the **baseline accuracy of ~88.86%**.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:**
  - Data & ML: `pandas`, `numpy`, `scikit-learn`
  - Deep Learning: `tensorflow`, `keras`
  - Visualization: `matplotlib`
  - Explainability: `lime`
  - Hyperparameter Tuning: `keras-tuner`

---

## ▶️ How to Run

1. **Clone the repo**

   ```bash
   git clone <your-repo-url>.git
   cd <your-repo-folder>
