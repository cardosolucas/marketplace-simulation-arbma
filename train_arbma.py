import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import joblib

def make_identifier(df):
    str_id = df.apply(lambda x: '_'.join(map(str, x)), axis=1)
    return pd.factorize(str_id)[0]

def sort_by_timestamp_and_natural(file_list):
    def sort_key(text):
        ts_match = re.search(r'(\d{8})_(\d{6})', text)
        if ts_match:
            try:
                ts_str = f"{ts_match.group(1)}_{ts_match.group(2)}"
                return (0, datetime.strptime(ts_str, '%Y%m%d_%H%M%S'), text.lower())
            except ValueError:
                pass
        convert = lambda segment: int(segment) if segment.isdigit() else segment.lower()
        alphanum_parts = [convert(c) for c in re.split(r'([0-9]+)', text)]
        return (1, alphanum_parts)
    return sorted(file_list, key=sort_key)

def arrange_data_by_id(df, id_col, columns, control_var, target, sensitive_attribute):
    data_list, sensitive_att_list, target_list = [], [], []
    iteration_list = sorted(df[id_col].unique())
    expected_rows = 14

    for i in iteration_list:
        sample_rows = df[(df[id_col] == i)].copy()

        if not (sample_rows[control_var] > 0).all():
            sample_rows = sample_rows[sample_rows["capital"] > 0].copy()
        
        sample_rows['mask'] = 1.0

        if len(sample_rows) < expected_rows:
            missing_rows = expected_rows - len(sample_rows)
            pad_data = pd.DataFrame(np.full((missing_rows, sample_rows.shape[1]), -1.0), columns=sample_rows.columns)
            pad_data['mask'] = 0.0 
            sample_rows = pd.concat([sample_rows, pad_data], ignore_index=True)
        elif len(sample_rows) > expected_rows:
            sample_rows = sample_rows.head(expected_rows)

        target_list.append(sample_rows[target].to_numpy())
        sensitive_att_list.append(sample_rows[sensitive_attribute].to_numpy())

        feat_cols = [c for c in columns if c not in [target, sensitive_attribute]] + ['mask']
        sample_rows_features = sample_rows[feat_cols]
        
        data_list.append(sample_rows_features.to_numpy())

    return np.array(data_list), np.array(target_list), np.array(sensitive_att_list)


class PredictorNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size=14):
        super(PredictorNetwork, self).__init__()
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

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.model(x)

class AdversaryNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size=14):
        super(AdversaryNetwork, self).__init__()
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

    def forward(self, x, prediction):
        prediction = prediction.unsqueeze(-1)
        combined_input = torch.cat([x, prediction], dim=2)
        combined_input = combined_input.flatten(start_dim=1)
        return self.model(combined_input)

# --- Dataset ---

class RegressionDataset(Dataset):
    def __init__(self, val_split=0.2):
        super(RegressionDataset, self).__init__()
        csv_files = glob.glob("data_train/*.csv")
        csv_files = sort_by_timestamp_and_natural(csv_files)

        if len(csv_files) > 0:
            full_df = pd.concat([pd.read_csv(file) for file in csv_files[:100]], ignore_index=True)
        else:
            raise FileNotFoundError("No CSV files found.")

        full_df['id_transaction'] = make_identifier(full_df[['timestamp_id', 'step']])
        feature_cols = ['price', 'installments', 'delivery_time', 'score', 'premium']

        self.inputs, self.targets, self.sensitive_attributes = arrange_data_by_id(
            full_df, 'id_transaction', feature_cols, 'capital', 'score', 'premium'
        )
        
        self.inputs = self.inputs.astype(np.float32)
        self.targets = self.targets.astype(np.float32)
        self.sensitive_attributes = self.sensitive_attributes.astype(np.float32)

        self.target_scaler =  StandardScaler()
        valid_targets = self.targets[self.targets != -1].reshape(-1, 1)
        self.target_scaler.fit(valid_targets)
        self.targets = self.target_scaler.transform(self.targets.reshape(-1, 1)).reshape(self.targets.shape)

        self.dataset_size = len(self.inputs)
        indices = torch.randperm(self.dataset_size).tolist()
        val_size = int(self.dataset_size * val_split)
        self.train_indices = indices[val_size:]
        self.val_indices = indices[:val_size]

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx]), torch.tensor(self.sensitive_attributes[idx])

def train_adversarial_model(model_pred, model_adv, train_dl, val_dl, opt_p, opt_a, device, alpha, epochs):
    tiny = 1e-8
    model_pred.to(device); model_adv.to(device)
    
    criterion_main = nn.MSELoss(reduction='none')
    criterion_adv = nn.BCEWithLogitsLoss(reduction='none')
    try:
        for epoch in range(epochs):
            model_pred.train(); model_adv.train()
            
            running_p_loss = 0.0
            running_a_loss = 0.0
            
            for inputs, targets, sens in train_dl:
                inputs, targets, sens = inputs.to(device), targets.to(device), sens.to(device)
                mask = inputs[:, :, -1]

                opt_a.zero_grad()
                p_out = model_pred(inputs).detach() 
                a_out = model_adv(inputs, p_out)
                a_loss = (criterion_adv(a_out, sens) * mask).sum() / mask.sum()
                a_loss.backward()
                opt_a.step()

                opt_p.zero_grad()
                p_out = model_pred(inputs)
                a_out = model_adv(inputs, p_out)
                
                p_loss_val = (criterion_main(p_out, targets) * mask).sum() / mask.sum()
                a_loss_val = (criterion_adv(a_out, sens) * mask).sum() / mask.sum()

                dW_LP = torch.autograd.grad(p_loss_val, model_pred.parameters(), retain_graph=True)
                dW_LA = torch.autograd.grad(a_loss_val, model_pred.parameters())

                for i, p in enumerate(model_pred.parameters()):
                    unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + tiny)
                    proj = torch.sum(dW_LP[i] * unit_dW_LA)

                    p.grad = dW_LP[i] - (proj * unit_dW_LA) - (alpha * dW_LA[i])

                opt_p.step()

                running_p_loss += p_loss_val.item()
                running_a_loss += a_loss_val.item()

            avg_p = running_p_loss / len(train_dl)
            avg_a = running_a_loss / len(train_dl)
            
            print(f"Epoch [{epoch+1:03d}/{epochs}] | Pred MSE: {avg_p:.6f} | Adv BCE: {avg_a:.6f}")

            if (epoch + 1) % 10 == 0:
                validate(model_pred, model_adv, val_dl, device, criterion_main, criterion_adv)
    except KeyboardInterrupt:
        torch.save(model_pred, 'adversarial_new_predictor_final_aa.pth')
        print("Modelo salvo 'adversarial_new_predictor_final_aa.pth'")

def validate(model_pred, model_adv, val_dl, device, criterion_main, criterion_adv):
    model_pred.eval()
    val_p_loss = 0.0
    with torch.no_grad():
        for inputs, targets, _ in val_dl:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = inputs[:, :, -1]
            out = model_pred(inputs)
            loss = (criterion_main(out, targets) * mask).sum() / mask.sum()
            val_p_loss += loss.item()
    print(f"    >>> Validation MSE: {val_p_loss / len(val_dl):.6f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RegressionDataset()
    train_ds = torch.utils.data.Subset(dataset, dataset.train_indices)
    val_ds = torch.utils.data.Subset(dataset, dataset.val_indices)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32)

    input_dim = (14, 4) 
    
    predictor = PredictorNetwork(input_dim, [256, 48, 144])
    adversary = AdversaryNetwork((14, 5), [48], 14)

    opt_p = optim.Adam(predictor.parameters(), lr=0.0004677101596098702)
    opt_a = optim.Adam(adversary.parameters(), lr=0.0004677101596098702)

    train_adversarial_model(predictor, adversary, train_dl, val_dl, opt_p, opt_a, device, 91.57221742511838, 500)

if __name__ == "__main__":
    main()