import pandas as pd
import numpy as np
import glob
import joblib
from fairlearn.adversarial import AdversarialFairnessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cloudpickle

def main():
    print("Loading training data...")
    csv_files = glob.glob("data_baseline/*.csv")
    

    full_df = pd.concat([pd.read_csv(file) for file in csv_files[:10]], ignore_index=True)

    full_df = full_df.replace([np.inf, -np.inf], np.nan)
    full_df = full_df.dropna()
    full_df['premium'] = full_df['premium'].astype(int)

    full_df = full_df[['price', 'installments', 'delivery_time', 'score', 'premium']]

    full_df.to_csv("teste.csv", header=True, index=False)

    # 1. Features
    feature_cols = ['price', 'installments', 'delivery_time']
    X = full_df[feature_cols]
    
    # 2. Target
    y = full_df['score']
    
    # 3. Protected attribute (False when protected; mapping boolean to int 0/1)
    Z = full_df['premium']

    print("Scaling features and target...")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
        X_scaled, y_scaled, Z, test_size=0.2, random_state=42
    )

    print("Initializing AdversarialFairnessRegressor...")
    mitigator = AdversarialFairnessRegressor(
        backend="torch",
        predictor_model=[256, 128, 64],
        adversary_model=[64, 32],
        epochs=1,
        batch_size=64,
        alpha=91.57221742511838,  # Fairness penalty strength
        random_state=42
    )

    print("Training model...")
    mitigator.fit(X_train, y_train, sensitive_features=Z_train)

    print("Evaluating model...")
    predictions = mitigator.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    print(f"Validation MSE (Scaled): {mse:.4f}")

    print("Saving model and scalers...")
    with open('fairlearn_adv_regressor.pkl', 'wb') as f:
        cloudpickle.dump(mitigator, f)
    joblib.dump(scaler_X, 'fairlearn_scaler_X.pkl')
    joblib.dump(scaler_y, 'fairlearn_scaler_y.pkl')
    print("Artifacts saved successfully.")

if __name__ == "__main__":
    main()