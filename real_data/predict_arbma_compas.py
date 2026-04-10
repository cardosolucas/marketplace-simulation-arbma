import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Network Definition (Required to load the model) ---
class PredictorClassificationNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sizes, seq_len=14, num_classes=10):
        super(PredictorClassificationNetwork, self).__init__()
        input_size = input_shape[0] * input_shape[1]
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = size
            
        layers.append(nn.Linear(prev_size, seq_len * num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = self.model(x)
        return out.view(-1, self.num_classes, self.seq_len)

def main():
    # --- 2. Load and Preprocess the Data ---
    print("Loading and formatting dataset...")
    df = pd.read_csv('datasets_arbma/compas-scores-two-years-violent.csv', header=0)
    
    # FIXED: Added 'is_violent_recid' back to the columns list to use as ground truth
    cols = ['sex', 'race', 'c_charge_degree', 'decile_score', 'age_cat',
            'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'is_violent_recid']
    df = df[cols]
    
    # Filter for target groups and ensure valid deciles (1-10)
    df = df[df['race'].isin(['Caucasian', 'African-American'])].copy()
    df = df[(df['decile_score'] >= 1) & (df['decile_score'] <= 10)]
    
    # Save original values for our final comparison report
    df['Race_Label'] = df['race']
    df['Original_Decile'] = df['decile_score']
    
    # Encoding
    df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
    df['race'] = df['race'].map({'Caucasian': 0, 'African-American': 1})
    
    cols_to_encode = ['c_charge_degree', 'age_cat']
    df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    df = df.dropna()

    # --- 3. Format Data for the Model ---
    seq_len = 14
    num_groups = len(df) // seq_len
    df_eval = df.iloc[:num_groups * seq_len].copy() 
    
    # FIXED: Added 'is_violent_recid' to exclude_cols so the model doesn't cheat!
    exclude_cols = ['decile_score', 'race', 'Race_Label', 'Original_Decile', 'is_violent_recid']
    feature_cols = [c for c in df_eval.columns if c not in exclude_cols]
    
    X = df_eval[feature_cols].values.astype(np.float32)
    mask = np.ones((len(X), 1), dtype=np.float32)
    X = np.hstack((X, mask))
    
    num_features = X.shape[1]
    X_tensor = torch.tensor(X.reshape(num_groups, seq_len, num_features))

    # --- 4. Run Inference ---
    print("Loading model and generating predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = X_tensor.to(device)
    
    model = torch.load('adversarial_categorical_predictor.pth', map_location=device)
    model.eval()
    
    with torch.no_grad():
        out = model(X_tensor) 
        preds = torch.argmax(out, dim=1) 
        preds_1_to_10 = preds.flatten().cpu().numpy() + 1
        
    df_eval['Predicted_Decile'] = preds_1_to_10

    # --- 5. Generate Ground Truth Fairness Analysis ---
    print("\n" + "="*60)
    print("      COMPAS GROUND TRUTH FAIRNESS ANALYSIS")
    print("="*60)
    
    # Define "Predicted Positive" (Medium or High Risk: Deciles 5-10)
    df_eval['Truth'] = df_eval['is_violent_recid']
    df_eval['Orig_Pred_Pos'] = (df_eval['Original_Decile'] >= 5).astype(int)
    df_eval['New_Pred_Pos'] = (df_eval['Predicted_Decile'] >= 5).astype(int)

    # --- 6. Generate Distribution Graphs ---
    print("\nGenerating distribution graphs...")
    
    # Set up the figure with 2 subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Define a consistent color palette for the races
    palette = {'Caucasian': '#1f77b4', 'African-American': '#ff7f0e'}
    
    # Graph 1: Original COMPAS Deciles
    sns.countplot(
        data=df_eval, 
        x='Original_Decile', 
        hue='Race_Label', 
        palette=palette, 
        ax=axes[0]
    )
    axes[0].set_title('BEFORE: Original COMPAS Deciles')
    axes[0].set_xlabel('Decile Score (1-10)')
    axes[0].set_ylabel('Number of Individuals')
    axes[0].legend(title='Race')
    
    # Graph 2: Predicted Deciles (Adversarial Model)
    sns.countplot(
        data=df_eval, 
        x='Predicted_Decile', 
        hue='Race_Label', 
        palette=palette, 
        ax=axes[1]
    )
    axes[1].set_title('AFTER: Predicted Deciles (Debiased)')
    axes[1].set_xlabel('Decile Score (1-10)')
    axes[1].set_ylabel('') # Hide y-label on the second graph for clean look
    axes[1].legend(title='Race')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('decile_distribution_comparison.png', dpi=300)
    print("Graph successfully saved as 'decile_distribution_comparison.png'")

    def calculate_rates(group):
        # Original Metrics
        orig_fp = ((group['Orig_Pred_Pos'] == 1) & (group['Truth'] == 0)).sum()
        orig_tn = ((group['Orig_Pred_Pos'] == 0) & (group['Truth'] == 0)).sum()
        orig_fpr = (orig_fp / (orig_fp + orig_tn)) * 100 if (orig_fp + orig_tn) > 0 else 0

        orig_fn = ((group['Orig_Pred_Pos'] == 0) & (group['Truth'] == 1)).sum()
        orig_tp = ((group['Orig_Pred_Pos'] == 1) & (group['Truth'] == 1)).sum()
        orig_fnr = (orig_fn / (orig_fn + orig_tp)) * 100 if (orig_fn + orig_tp) > 0 else 0

        # New Debiased Metrics
        new_fp = ((group['New_Pred_Pos'] == 1) & (group['Truth'] == 0)).sum()
        new_tn = ((group['New_Pred_Pos'] == 0) & (group['Truth'] == 0)).sum()
        new_fpr = (new_fp / (new_fp + new_tn)) * 100 if (new_fp + new_tn) > 0 else 0

        new_fn = ((group['New_Pred_Pos'] == 0) & (group['Truth'] == 1)).sum()
        new_tp = ((group['New_Pred_Pos'] == 1) & (group['Truth'] == 1)).sum()
        new_fnr = (new_fn / (new_fn + new_tp)) * 100 if (new_fn + new_tp) > 0 else 0

        return pd.Series({
            'Orig_FPR (%)': round(orig_fpr, 2),
            'New_FPR (%)': round(new_fpr, 2),
            'Orig_FNR (%)': round(orig_fnr, 2),
            'New_FNR (%)': round(new_fnr, 2)
        })

    rates_df = df_eval.groupby('Race_Label').apply(calculate_rates)
    
    print("\nFALSE POSITIVE RATES (FPR)")
    print("Flagged as High Risk, but DID NOT violently recidivate:")
    print(rates_df[['Orig_FPR (%)', 'New_FPR (%)']])
    
    print("-" * 60)
    
    print("FALSE NEGATIVE RATES (FNR)")
    print("Flagged as Low Risk, but DID violently recidivate:")
    print(rates_df[['Orig_FNR (%)', 'New_FNR (%)']])
    
    print("="*60)

if __name__ == "__main__":
    main()