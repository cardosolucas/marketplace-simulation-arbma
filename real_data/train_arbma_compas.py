import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn import metrics as skm

from fairlearn.metrics import (
    MetricFrame,
    count,
    selection_rate,
    false_positive_rate
)

# --- Dataset ---

class CompasCategoricalDataset(Dataset):
    def __init__(self, csv_path='datasets_arbma/compas-scores-two-years-violent.csv', val_split=0.1, seq_len=14):
        super(CompasCategoricalDataset, self).__init__()
        
        # 1. Load and Filter
        df = pd.read_csv(csv_path, header=0)
        cols = ['sex', 'race', 'c_charge_degree', 'decile_score', 'age_cat',
                'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
        df = df[cols]
        df = df[df['race'].isin(['Caucasian', 'African-American'])].copy()
        
        # 3. Encoding
        df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
        df['race'] = df['race'].map({'Caucasian': 0, 'African-American': 1})
        
        cols_to_encode = ['c_charge_degree', 'age_cat']
        df = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
        df = df.dropna()
        
        # 4. Shuffle and Group
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        target_col = 'decile_score'
        sens_col = 'race'
        
        feature_cols = [c for c in df.columns if c not in [target_col, sens_col]]
        self.num_features = len(feature_cols) + 1 # +1 for the mask column
        
        num_groups = len(df) // seq_len
        df = df.iloc[:num_groups * seq_len].copy()
        
        # Target is now exactly 0 to 9 for CrossEntropy classification
        X = df[feature_cols].values.astype(np.float32)
        y = (df[target_col].values.astype(np.int64) - 1) 
        sens = df[sens_col].values.astype(np.float32)
        
        mask = np.ones((len(X), 1), dtype=np.float32)
        X = np.hstack((X, mask))
        
        self.inputs = X.reshape(num_groups, seq_len, self.num_features)
        self.targets = y.reshape(num_groups, seq_len)
        self.sensitive_attributes = sens.reshape(num_groups, seq_len)
        
        # Train/Val Split
        self.dataset_size = len(self.inputs)
        indices = torch.randperm(self.dataset_size).tolist()
        val_size = int(self.dataset_size * val_split)
        self.train_indices = indices[val_size:]
        self.val_indices = indices[:val_size]

    def __len__(self): return len(self.inputs)
    
    def __getitem__(self, idx):
        # Targets are cast to torch.long because CrossEntropyLoss requires it
        return (torch.tensor(self.inputs[idx]), 
                torch.tensor(self.targets[idx], dtype=torch.long), 
                torch.tensor(self.sensitive_attributes[idx]))

# --- Networks ---

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
            
        # Output is Sequence Length * Number of Classes (14 * 10 = 140)
        layers.append(nn.Linear(prev_size, seq_len * num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = self.model(x)
        # Reshape to (Batch_size, Num_Classes, Sequence_Length)
        # This exact shape is required by PyTorch's CrossEntropyLoss
        return out.view(-1, self.num_classes, self.seq_len)

class AdversaryClassificationNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size=14):
        super(AdversaryClassificationNetwork, self).__init__()
        # input_shape[1] now includes the 10 class probabilities instead of 1 score
        input_size = input_shape[0] * input_shape[1] 
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
        # prediction_probs comes in as (Batch, SeqLen, NumClasses)
        combined_input = torch.cat([x, prediction_probs], dim=2)
        combined_input = combined_input.flatten(start_dim=1)
        return self.model(combined_input)

# --- Training Loop ---

def train_adversarial_model(model_pred, model_adv, train_dl, val_dl, opt_p, opt_a, device, alpha, epochs, verbose=True):
    tiny = 1e-8
    model_pred.to(device)
    model_adv.to(device)
    
    # Changed to CrossEntropy for Multi-class
    criterion_main = nn.CrossEntropyLoss(reduction='none') 
    criterion_adv = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(epochs):
        model_pred.train()
        model_adv.train()
        
        running_p_loss = 0.0
        running_a_loss = 0.0
        
        for inputs, targets, sens in train_dl:
            inputs, targets, sens = inputs.to(device), targets.to(device), sens.to(device)
            mask = inputs[:, :, -1]
            
            # Train Adversary
            opt_a.zero_grad()
            p_out = model_pred(inputs).detach() 
            
            # Convert logits to probabilities and transpose to (Batch, SeqLen, NumClasses) for adversary
            p_probs = torch.softmax(p_out, dim=1).transpose(1, 2)
            
            a_out = model_adv(inputs, p_probs)
            a_loss = (criterion_adv(a_out, sens) * mask).sum() / mask.sum()
            a_loss.backward()
            opt_a.step()
            
            # Train Predictor
            opt_p.zero_grad()
            p_out = model_pred(inputs)
            p_probs = torch.softmax(p_out, dim=1).transpose(1, 2)
            a_out = model_adv(inputs, p_probs)
            
            p_loss_val = (criterion_main(p_out, targets) * mask).sum() / mask.sum()
            a_loss_val = (criterion_adv(a_out, sens) * mask).sum() / mask.sum()
            dW_LP = torch.autograd.grad(p_loss_val, model_pred.parameters(), retain_graph=True)
            dW_LA = torch.autograd.grad(a_loss_val, model_pred.parameters())
            
            for i, p in enumerate(model_pred.parameters()):
                unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + tiny)
                proj = torch.sum(dW_LP[i] * unit_dW_LA)
                p.grad = dW_LP[i] - (proj * unit_dW_LA) - (alpha * dW_LA[i])
                
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
    
    preds_list = []
    targets_list = []
    sens_list = []
    mask_list = []
    
    with torch.no_grad():
        for inputs, targets, sens in val_dl:
            inputs, targets, sens = inputs.to(device), targets.to(device), sens.to(device)
            mask = inputs[:, :, -1]
            out = model_pred(inputs)
            
            loss = (criterion_main(out, targets) * mask).sum() / mask.sum()
            val_p_loss += loss.item()
            
            predictions = torch.argmax(out, dim=1)
            
            preds_list.append(predictions.cpu())
            targets_list.append(targets.cpu())
            sens_list.append(sens.cpu())
            mask_list.append(mask.cpu())
            
    # Flatten the sequences to evaluate overall fairness
    preds_flat = torch.cat(preds_list).flatten().numpy()
    targets_flat = torch.cat(targets_list).flatten().numpy()
    sens_flat = torch.cat(sens_list).flatten().numpy()
    mask_flat = torch.cat(mask_list).flatten().numpy()
    
    # Filter out masked paddings
    valid_idx = mask_flat == 1.0
    preds_valid = preds_flat[valid_idx]
    targets_valid = targets_flat[valid_idx]
    sens_valid = sens_flat[valid_idx]

    acc = np.mean(preds_valid == targets_valid) * 100 if len(preds_valid) > 0 else 0

    binary_metric_frame = MetricFrame(
        metrics={
            "false_positive_rate": false_positive_rate,
            "selection_rate": selection_rate,
            "count": count
        },
        sensitive_features=sens_valid,
        y_true=np.where(targets_valid >= 5, 1, 0),
        y_pred=np.where(preds_valid >= 5, 1, 0),
    )

    df_binary_metrics = binary_metric_frame.by_group
    
    # Fallback in case a batch is missing a demographic group during tuning
    try:
        fp_protected = df_binary_metrics[df_binary_metrics.index == 1].iloc[0]['false_positive_rate']
    except IndexError: fp_protected = 0.0
    
    try:
        fp_non_protected = df_binary_metrics[df_binary_metrics.index == 0].iloc[0]['false_positive_rate']
    except IndexError: fp_non_protected = 0.0
        
    false_positive_diff = abs(fp_protected - fp_non_protected)

    if verbose:
        print(f"    >>> Validation Loss: {val_p_loss / len(val_dl):.6f} | Accuracy: {acc:.2f}% | FPR Diff: {false_positive_diff:.6f}")
        
    return false_positive_diff

# --- Optuna Optimization ---

def objective(trial, train_dl, val_dl, device, input_dim_predictor, input_dim_adversary, seq_len, num_classes):
    alpha = trial.suggest_float('alpha', 0.1, 100.0)
    
    predictor = PredictorClassificationNetwork(input_dim_predictor, [256, 48, 144], seq_len=seq_len, num_classes=num_classes)
    adversary = AdversaryClassificationNetwork(input_dim_adversary, [48], output_size=seq_len)

    opt_p = optim.Adam(predictor.parameters(), lr=0.0046)
    opt_a = optim.Adam(adversary.parameters(), lr=0.0046)

    predictor = train_adversarial_model(predictor, adversary, train_dl, val_dl, opt_p, opt_a, device, alpha, epochs=40, verbose=False)
    
    criterion_main = nn.CrossEntropyLoss(reduction='none')
    false_positive_diff = validate(predictor, val_dl, device, criterion_main, verbose=False)
    
    return false_positive_diff if false_positive_diff > 0 else float('inf')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = CompasCategoricalDataset(val_split=0.2)
    train_ds = torch.utils.data.Subset(dataset, dataset.train_indices)
    val_ds = torch.utils.data.Subset(dataset, dataset.val_indices)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    seq_len = 14
    num_classes = 10
    num_features = dataset.num_features 
    
    input_dim_predictor = (seq_len, num_features)
    input_dim_adversary = (seq_len, num_features + num_classes) 
    
    #print("Starting Optuna Hyperparameter Optimization...")
    #study = optuna.create_study(direction="minimize")
    #study.optimize(lambda trial: objective(trial, train_dl, val_dl, device, input_dim_predictor, input_dim_adversary, seq_len, num_classes), n_trials=500)
    
    #best_alpha = study.best_params['alpha']
    #print(f"\n--- Optimization Complete ---")
    #print(f"Best alpha found: {best_alpha:.4f} with FPR diff: {study.best_value:.4f}")
    
    #print(f"\nTraining final model with best alpha = {best_alpha:.4f} for 100 epochs...")
    final_predictor = PredictorClassificationNetwork(input_dim_predictor, [256, 48, 144], seq_len=seq_len, num_classes=num_classes)
    final_adversary = AdversaryClassificationNetwork(input_dim_adversary, [48], output_size=seq_len)

    opt_p_final = optim.Adam(final_predictor.parameters(), lr=0.0046)
    opt_a_final = optim.Adam(final_adversary.parameters(), lr=0.0046)

    final_predictor = train_adversarial_model(final_predictor, final_adversary, train_dl, val_dl, opt_p_final, opt_a_final, device, 40.5100, epochs=100, verbose=True)
    
    torch.save(final_predictor, 'best_arbma_categorical_predictor.pth')
    print("Best model saved as 'best_arbma_categorical_predictor.pth'")

if __name__ == "__main__":
    main()