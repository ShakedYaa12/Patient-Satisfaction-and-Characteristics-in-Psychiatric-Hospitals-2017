
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, f1_score
import re


def XGBoost_Classifier_Improved(path, k=5):
    
    df = pd.read_csv(path)

    # Perform all column name modifications at once
    df.columns = [
        re.sub(r'\s+', '_', col)  # Replacing spaces with `_`
        .replace('\xa0', '_')  # Removing invisible spaces
        .replace('(', '')  # Removing left parentheses
        .replace(')', '')  # Removing right parentheses
        .replace("'", '')  # Removing single quotes
        .replace('"', '')  # Removing double quotes (for additional cases)
        .replace('[','').replace(']','').replace('<','').replace('>','')  # Removing square brackets and special characters
        .replace('.','')  # Removing periods
        .replace('–','_')  # Removing non-standard hyphens and replacing them with `_`
        .replace('-','')  # Removing hyphens
        .replace('__', '_')  # Preventing double underscores
        for col in df.columns
    ]

    # Re-checking if there are any invalid columns
    print([col for col in df.columns if not col.isidentifier()])

    # --- Step 1: Define Columns ---
    survey_columns = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9',
                      'q10', 'q12', 'q13', 'q14','q15', 'q16', 'q17', 'q18', 'q19',
                      'q20', 'q21', 'q22', 'q23', 'q24', 'q25', 'q26', 'q27','q28', 'q29',
                      'q30','q31', 'q33', 'q34', 'q35', 'q36', 'q37', 'q71', 'q72', 'q73', 'q74',
                      'q75', 'q76', 'forced_during_agreed']

    # --- Step 2: Prepare Data ---
    X = df.drop(columns=survey_columns + ["kmeans_cluster"])  
    y = df["kmeans_cluster"]  

    # Define K-Fold Cross Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Lists to store metrics for each fold
    recall_scores = []
    precision_scores = []
    f1_scores = []

    # --- Step 3: Train and Evaluate Performance for Each Fold ---
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Training on Fold {fold_idx+1}/{k}...")

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Calculate scale_pos_weight (class imbalance ratio)
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # Create XGBoost model
        model = xgb.XGBClassifier(
            eval_metric="logloss",
            n_estimators=100,
            random_state=42,
            scale_pos_weight=pos_weight  
        )

        model.fit(X_train, y_train)

        # --- Step 4: Predict and Evaluate Performance ---
        y_pred_prob = model.predict_proba(X_test)[:, 1]  
        threshold = 0.15  # Decision threshold
        y_pred = (y_pred_prob >= threshold).astype(int)

        # Calculate performance metrics
        recall_class_1 = recall_score(y_test, y_pred, pos_label=1)
        precision_class_1 = precision_score(y_test, y_pred, pos_label=1)
        f1_class_1 = f1_score(y_test, y_pred)

        recall_scores.append(recall_class_1)
        precision_scores.append(precision_class_1)
        f1_scores.append(f1_class_1)

        print(f"Fold {fold_idx+1} - Recall: {recall_class_1:.2f}, Precision: {precision_class_1:.2f}, F1: {f1_class_1:.2f}")

    # --- Step 5: Calculate Average Metrics ---
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)
    avg_f1 = np.mean(f1_scores)

    print("\n📊 Average metrics across all folds:")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average F1-Score: {avg_f1:.2f}")

    return model  # Returns the last trained model