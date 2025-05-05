# Loan_Approval_Classification
1. Introduction
Objective: Build a predictive model to classify loan applications as approved (no default) or rejected (default risk), enabling lenders to identify high‐risk loans before disbursal.
Motivation: Early detection of potential defaults helps financial institutions reduce losses, optimize credit underwriting, and allocate resources more efficiently.
________________________________________

2. Data Description
•	Source: loan_data.csv containing attributes of past loan applicants.
•	Target Variable: loan_status (binary: Approved = 1, Rejected/Default = 0).
•	Features:
o	Categorical (5): person_gender, person_education, person_home_ownership, loan_intent, previous_loan_defaults_on_file
o	Numerical (~20+): Income, annual debt, credit scores, loan amount, interest rate, etc.

Exploratory Analysis:
•	Dataset shape confirmed (N records × M features).
•	Class imbalance observed: roughly X% approved vs. Y% rejected.
•	Basic statistics (means, counts) and feature distributions inspected.
________________________________________

3. Data Preprocessing
1.	Train/Val/Test Split:
o	60% training, 20% validation, 20% test (stratified by loan_status).
2.	Pipeline (ColumnTransformer):
o	One‐hot encoding of categorical features (drop='first' to avoid multicollinearity).
o	Min–Max scaling of numerical features into [0,1].
3.	Class Weights: Computed via sklearn.utils.class_weight to counteract imbalance during model training.
________________________________________

4. Model Development
4.1 Baseline Neural Network
•	Architecture (Keras Sequential):
1.	Dense(64, relu) → Dropout(0.3)
2.	Dense(32, relu) → Dropout(0.3)
3.	Dense(1, sigmoid) for binary classification
•	Compilation:
o	Optimizer: Adam
o	Loss: Binary Crossentropy
o	Metric: Accuracy
•	Training:
o	Epochs: up to 50
o	Batch size: 32
o	EarlyStopping (patience=5, restore best weights)
o	ModelCheckpoint to save best model
o	Class weights applied
4.2 Hyperparameter Tuning (Keras Tuner)
•	Methods compared:
o	Random Search
o	Bayesian Optimization
o	Hyperband
•	Hyperparameters tuned: number of units in hidden layers (32–256), dropout rates (0.0–0.5), learning rate, etc.
•	Evaluation: Best model from each tuner evaluated on the held‐out test set; accuracies compared in a summary table.
________________________________________

5. Results
•	Baseline Model Performance:
o	Test accuracy: ≈A.BC%
o	Confusion matrix and classification report showed precision/recall trade‐offs.
•	Learning Curves:
o	Training vs. validation accuracy and loss plotted over epochs; early stopping prevented overfitting.
•	Hyperparameter Tuning:
o	Random Search: ≈ 91.56%
o	Bayesian Opt.: ≈91.77%
o	Hyperband: ≈91.9%
o	Best overall: Hyperband with 92% test accuracy.
•	Model Interpretability:
o	LIME explanations generated for sample test instances, highlighting top features driving approval/rejection decisions.
________________________________________

6. Conclusions & Future Work
•	The deep‐learning classifier achieved strong predictive performance on the loan default dataset, with Hyperband‐tuned models offering the best accuracy. (91.9 percent)
•	LIME analyses provided human‐interpretable insights into individual predictions, which is crucial for credit‐decision transparency.
________________________________________

Key Takeaway:
A structured deep‐learning pipeline—combining proper preprocessing, class‐imbalance handling, early stopping, and hyperparameter tuning—can effectively predict loan defaults, while interpretability tools like LIME ensure that model decisions remain transparent to stakeholders.
