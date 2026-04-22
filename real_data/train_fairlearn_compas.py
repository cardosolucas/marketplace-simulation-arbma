import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skm
import optuna

from fairlearn.metrics import (
    MetricFrame,
    count,
    plot_model_comparison,
    selection_rate,
    selection_rate_difference,
    false_positive_rate_difference,
    false_positive_rate
)
from fairlearn.reductions import DemographicParity, ErrorRate, GridSearch

# --- Dataset ---

class CompasCategoricalDataset(Dataset):
    def __init__(self, csv_path='datasets_arbma/compas-scores-two-years-violent.csv', val_split=0.1):
        super(CompasCategoricalDataset, self).__init__()
        
        # 1. Load and Filter
        df = pd.read_csv(csv_path, header=0)
        cols = ['sex', 'race', 'c_charge_degree', 'decile_score', 'age_cat',
                'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        df = df[cols]
        df = df[df['race'].isin(['Caucasian', 'African-American'])].copy()

        df = df[(df['decile_score'] >= 1) & (df['decile_score'] <= 10)].copy()
        
        # 2. Encoding
        df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
        df['race'] = df['race'].map({'Caucasian': 0, 'African-American': 1})
        
        cols_to_encode = ['c_charge_degree', 'age_cat']
        df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
        df = df.dropna()
        
        # 3. Shuffle
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        target_col = 'decile_score'
        sens_col = 'race'
        
        feature_cols = [c for c in df.columns if c not in [target_col, sens_col]]
        self.num_features = len(feature_cols)
        
        # Target is now exactly 0 to 9 for CrossEntropy classification
        X = df[feature_cols].values.astype(np.float32)
        y = (df[target_col].values.astype(np.int64) - 1) 
        sens = df[sens_col].values.astype(np.float32)
        
        # No more reshaping or masking needed for single instances
        self.inputs = X
        self.targets = y
        self.sensitive_attributes = sens
        
        # Train/Val Split
        self.dataset_size = len(self.inputs)
        indices = torch.randperm(self.dataset_size).tolist()
        val_size = int(self.dataset_size * val_split)
        self.train_indices = indices[val_size:]
        self.val_indices = indices[:val_size]

    def __len__(self): return len(self.inputs)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.inputs[idx]), 
                torch.tensor(self.targets[idx], dtype=torch.long), 
                torch.tensor(self.sensitive_attributes[idx]))

# --- Networks ---

class PredictorClassificationNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=10):
        super(PredictorClassificationNetwork, self).__init__()
        self.num_classes = num_classes
        
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = size
            
        # Output is simply the number of classes
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Input: (Batch, Features) -> Output: (Batch, Num_Classes)
        return self.model(x)

class AdversaryClassificationNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super(AdversaryClassificationNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x, prediction_probs):
        # x: (Batch, Features), prediction_probs: (Batch, NumClasses)
        combined_input = torch.cat([x, prediction_probs], dim=1)
        return self.model(combined_input)


def mean_prediction(y_true, y_pred):
    return np.mean(y_pred)

def train_adversarial_model(model_pred, model_adv, train_dl, val_dl, opt_p, opt_a, device, alpha, epochs, verbose=True):
    tiny = 1e-8
    model_pred.to(device)
    model_adv.to(device)
    
    criterion_main = nn.CrossEntropyLoss() 
    criterion_adv = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model_pred.train() 
        model_adv.train()
        
        running_p_loss = 0.0
        running_a_loss = 0.0
        
        for inputs, targets, sens in train_dl:
            inputs, targets, sens = inputs.to(device), targets.to(device), sens.to(device)
            
            # Train Adversary
            opt_a.zero_grad()
            p_out = model_pred(inputs).detach() 
            p_probs = torch.softmax(p_out, dim=1)
            
            a_out = model_adv(inputs, p_probs).squeeze(1) 
            a_loss = criterion_adv(a_out, sens)
            a_loss.backward()
            opt_a.step()
            
            # Train Predictor
            opt_p.zero_grad()
            p_out = model_pred(inputs)
            p_probs = torch.softmax(p_out, dim=1)
            a_out = model_adv(inputs, p_probs).squeeze(1)
            
            p_loss_val = criterion_main(p_out, targets)
            a_loss_val = criterion_adv(a_out, sens)
            
            dW_LP = torch.autograd.grad(p_loss_val, model_pred.parameters(), retain_graph=True)
            dW_LA = torch.autograd.grad(a_loss_val, model_pred.parameters())
            
            # Gradient Projection
            for i, p in enumerate(model_pred.parameters()):
                unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + tiny)
                proj = torch.sum(dW_LP[i] * unit_dW_LA)
                new_grad = dW_LP[i] - (proj * unit_dW_LA) - (alpha * dW_LA[i])
                p.grad = new_grad.contiguous()
                
            torch.nn.utils.clip_grad_norm_(model_pred.parameters(), max_norm=1.0)
            opt_p.step()
            
            running_p_loss += p_loss_val.item()
            running_a_loss += a_loss_val.item()
            
        if verbose:
            avg_p = running_p_loss / len(train_dl)
            avg_a = running_a_loss / len(train_dl)
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Pred Loss: {avg_p:.6f} | Adv BCE: {avg_a:.6f}")
            if (epoch + 1) % 10 == 0:
                validate(model_pred, val_dl, device, criterion_main, verbose=True)
                
    return model_pred

def validate(model_pred, val_dl, device, criterion_main, verbose=True):
    model_pred.eval()
    val_p_loss = 0.0
    
    with torch.no_grad():
        predictions_list = []
        targets_list = []
        sens_list = []
        for inputs, targets, sens in val_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            
            out = model_pred(inputs)
            loss = criterion_main(out, targets)
            val_p_loss += loss.item()
            
            predictions_list.append(torch.argmax(out, dim=1))
            targets_list.append(targets)
            sens_list.append(sens)
            
        predictions_list = torch.cat(predictions_list)
        targets_list = torch.cat(targets_list)
        sens_list = torch.cat(sens_list)
            
        metric_frame = MetricFrame(
            metrics={
                "accuracy": skm.accuracy_score,
                "mean_prediction": mean_prediction,
                "count": count,
            },
            sensitive_features=sens_list.cpu().numpy(),
            y_true=targets_list.cpu().numpy(),
            y_pred=predictions_list.cpu().numpy(),
        )

        binary_metric_frame = MetricFrame(
            metrics={
                "false_positive_rate": false_positive_rate,
                "selection_rate": selection_rate,
                "count": count
            },
            sensitive_features=sens_list.cpu().numpy(),
            y_true=np.where(targets_list.cpu().numpy() >= 5, 1, 0),
            y_pred=np.where(predictions_list.cpu().numpy() >= 5, 1, 0),
        )
        
        if verbose:
            print(metric_frame.by_group)
            print(binary_metric_frame.by_group)

        df_binary_metrics = binary_metric_frame.by_group
        false_positive_protected = df_binary_metrics[df_binary_metrics.index == 1].iloc[0].tolist()[0]
        false_positive_non_protected = df_binary_metrics[df_binary_metrics.index == 0].iloc[0].tolist()[0]
        false_positive_diff = abs(false_positive_protected - false_positive_non_protected)

        return metric_frame, binary_metric_frame, false_positive_diff
    
def objective(trial, train_dl, val_dl, device, input_dim_predictor, input_dim_adversary, num_classes):
    # Suggest an alpha between 0.1 and 1.0
    alpha = trial.suggest_float('alpha', 0.1, 100.0)
    
    # Initialize fresh models for each trial
    predictor = PredictorClassificationNetwork(input_dim_predictor, [256, 48, 144], num_classes=num_classes)
    adversary = AdversaryClassificationNetwork(input_dim_adversary, [48], output_size=1)

    opt_p = optim.Adam(predictor.parameters(), lr=0.0046)
    opt_a = optim.Adam(adversary.parameters(), lr=0.0046)

    # Train for fewer epochs during the study to save time (e.g., 40), or keep it at 100
    # verbose=False prevents console flooding during tuning
    predictor = train_adversarial_model(predictor, adversary, train_dl, val_dl, opt_p, opt_a, device, alpha, epochs=40, verbose=False)
    
    criterion_main = nn.CrossEntropyLoss()
    _, _, false_positive_diff = validate(predictor, val_dl, device, criterion_main, verbose=False)
    
    return false_positive_diff

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = CompasCategoricalDataset(val_split=0.2)
    train_ds = torch.utils.data.Subset(dataset, dataset.train_indices)
    val_ds = torch.utils.data.Subset(dataset, dataset.val_indices)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    num_classes = 10
    num_features = dataset.num_features 
    
    input_dim_predictor = num_features
    input_dim_adversary = num_features + num_classes 
    
    print("Starting Optuna Hyperparameter Optimization...")
    # Create study optimizing for the minimum difference
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dl, val_dl, device, input_dim_predictor, input_dim_adversary, num_classes), n_trials=500)
    
    best_alpha = study.best_params['alpha']
    print(f"\n--- Optimization Complete ---")
    print(f"Best alpha found: {best_alpha:.4f} with FPR diff: {study.best_value:.4f}")
    
    print(f"\nTraining final model with best alpha = {best_alpha:.4f} for 100 epochs...")
    final_predictor = PredictorClassificationNetwork(input_dim_predictor, [256, 48, 144], num_classes=num_classes)
    final_adversary = AdversaryClassificationNetwork(input_dim_adversary, [48], output_size=1)

    opt_p_final = optim.Adam(final_predictor.parameters(), lr=0.0046)
    opt_a_final = optim.Adam(final_adversary.parameters(), lr=0.0046)

    # Train final model with full epochs and verbosity enabled
    final_predictor = train_adversarial_model(final_predictor, final_adversary, train_dl, val_dl, opt_p_final, opt_a_final, device, best_alpha, epochs=100, verbose=True)
    
    # Save the final optimized model
    torch.save(final_predictor, 'best_adversarial_categorical_predictor.pth')
    print("Best model saved as 'best_adversarial_categorical_predictor.pth'")

if __name__ == "__main__":
    main()

