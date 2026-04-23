import pandas as pd
import numpy as np
import joblib  # Or use pickle to load the model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def main():
    # --- 1. Load and Preprocess the Data ---
    print("Loading and formatting dataset...")
    # Using the same path and logic as your NN script
    df = pd.read_csv('datasets_arbma/compas-scores-two-years-violent.csv', header=0)
    
    cols = ['sex', 'race', 'c_charge_degree', 'decile_score', 'age_cat',
            'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_violent_recid']
    df = df[cols]
    
    # Filter for target groups and ensure valid deciles (1-10)
    df = df[df['race'].isin(['Caucasian', 'African-American'])].copy()
    df = df[(df['decile_score'] >= 1) & (df['decile_score'] <= 10)]
    
    # Save original values for final report
    df['Race_Label'] = df['race']
    df['Original_Decile'] = df['decile_score']
    
    # Encoding (Matching your training script)
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    df['race_bin'] = df['race'].map({'Caucasian': 0, 'African-American': 1})
    
    cols_to_encode = ['c_charge_degree', 'age_cat']
    df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    df = df.dropna()

    # --- 2. Format Data for the Model ---
    exclude_cols = ['decile_score', 'race', 'Race_Label', 'Original_Decile', 'is_violent_recid', 'race_bin']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values

    # --- 3. Run Inference ---
    print("Loading Random Forest model and generating predictions...")
    # Note: Assuming you saved your model using joblib.dump(model, 'rf_baseline.joblib')
    # If you haven't saved it yet, you can initialize and fit a quick one here for testing
    import joblib
    try:
        model = joblib.load('rf_baseline.joblib')
    except:
        print("Model file not found. Training a quick baseline for demonstration...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, df['decile_score'] - 1)
    
    preds = model.predict(X) 
    preds_1_to_10 = preds + 1 # Convert back to 1-10 decile range
    df['Predicted_Decile'] = preds_1_to_10

    # --- 4. Generate Fairness Analysis ---
    print("\n" + "="*60)
    print("      RANDOM FOREST BASELINE FAIRNESS ANALYSIS")
    print("="*60)
    
    df['Truth'] = df['is_violent_recid']
    df['Orig_Pred_Pos'] = (df['Original_Decile'] >= 5).astype(int)
    df['New_Pred_Pos'] = (df['Predicted_Decile'] >= 5).astype(int)

    # --- 5. Generate Distribution Graphs ---
    print("\nGenerating distribution graphs...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    palette = {'Caucasian': '#1f77b4', 'African-American': '#ff7f0e'}
    
    sns.countplot(data=df, x='Original_Decile', hue='Race_Label', palette=palette, ax=axes[0])
    axes[0].set_title('ORIGINAL: COMPAS Deciles')
    
    sns.countplot(data=df, x='Predicted_Decile', hue='Race_Label', palette=palette, ax=axes[1])
    axes[1].set_title('BASELINE: Random Forest Predictions')
    
    plt.tight_layout()
    plt.savefig('rf_baseline_comparison.png', dpi=300)
    print("Graph saved as 'rf_baseline_comparison.png'")

    # --- 6. Metrics Calculation ---
    def calculate_rates(group):
        def get_fpr_fnr(pred_col):
            fp = ((group[pred_col] == 1) & (group['Truth'] == 0)).sum()
            tn = ((group[pred_col] == 0) & (group['Truth'] == 0)).sum()
            fn = ((group[pred_col] == 0) & (group['Truth'] == 1)).sum()
            tp = ((group[pred_col] == 1) & (group['Truth'] == 1)).sum()
            
            fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
            fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
            return fpr, fnr

        orig_fpr, orig_fnr = get_fpr_fnr('Orig_Pred_Pos')
        rf_fpr, rf_fnr = get_fpr_fnr('New_Pred_Pos')

        return pd.Series({
            'Orig_FPR (%)': round(orig_fpr, 2),
            'RF_FPR (%)': round(rf_fpr, 2),
            'Orig_FNR (%)': round(orig_fnr, 2),
            'RF_FNR (%)': round(rf_fnr, 2)
        })

    rates_df = df.groupby('Race_Label').apply(calculate_rates)
    
    print("\nFALSE POSITIVE RATES (FPR) - Higher is worse for the defendant")
    print(rates_df[['Orig_FPR (%)', 'RF_FPR (%)']])
    print("-" * 60)
    print("FALSE NEGATIVE RATES (FNR) - Higher is worse for public safety")
    print(rates_df[['Orig_FNR (%)', 'RF_FNR (%)']])
    print("="*60)

if __name__ == "__main__":
    main()