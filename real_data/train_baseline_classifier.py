import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics as skm
from fairlearn.metrics import MetricFrame, count

# 1. Data Preparation (Matching your CompasCategoricalDataset logic)
def load_compas_data(csv_path='datasets_arbma/compas-scores-two-years-violent.csv'):
    df = pd.read_csv(csv_path)
    cols = ['sex', 'race', 'c_charge_degree', 'decile_score', 'age_cat',
            'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
    
    # Filtering and Cleaning
    df = df[cols]
    df = df[df['race'].isin(['Caucasian', 'African-American'])].copy()
    df = df[(df['decile_score'] >= 1) & (df['decile_score'] <= 10)].copy()
    
    # Encoding
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    df['race_label'] = df['race'].map({'Caucasian': 0, 'African-American': 1})
    
    cols_to_encode = ['c_charge_degree', 'age_cat']
    df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    df = df.dropna()
    
    # Separate Features, Target, and Sensitive Attribute
    # Target is 0-9 (consistent with your PyTorch script)
    y = df['decile_score'].values.astype(int) - 1
    S = df['race_label'] 
    X = df.drop(['decile_score', 'race', 'race_label'], axis=1)
    
    return train_test_split(X, y, S, test_size=0.2, random_state=42)

# 2. Training the Baseline
X_train, X_test, y_train, y_test, S_train, S_test = load_compas_data()

print(f"Training Random Forest on {len(X_train)} samples...")
rf_baseline = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10, 
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train, y_train)

# 3. Predictions
y_pred = rf_baseline.predict(X_test)

# 4. Fairlearn Evaluation
def mean_prediction(y_true, y_pred):
    return np.mean(y_pred)

metric_frame = MetricFrame(
    metrics={
        "accuracy": skm.accuracy_score,
        "f1_macro": lambda y_true, y_pred: skm.f1_score(y_true, y_pred, average="macro"),
        "mean_decile": mean_prediction,
        "count": count
    },
    sensitive_features=S_test,
    y_true=y_test,
    y_pred=y_pred
)

print("\n--- Baseline Results by Race (0: Caucasian, 1: African-American) ---")
print(metric_frame.by_group)

print("\n--- Overall Metrics ---")
print(metric_frame.overall)