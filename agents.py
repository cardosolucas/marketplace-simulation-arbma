import time
from abc import ABC

import os
import numpy
import pandas as pd
import numpy as np
from scipy.stats import rankdata, binom, percentileofscore
import optimization
import scipy.optimize as optim
from model import OptimizationModel, PredictorNetwork
import joblib
import random
import torch
import cloudpickle
import math
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

KL_DIVERGENCE = "rKL"
ND_DIFFERENCE = "rND"
RD_DIFFERENCE = "rRD"

class Agent(ABC):

    def __init__(self):
        pass

    def rank(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FeldmanAgent(Agent):
    def __init__(self, protected_feature: str = 'premium'):
        self.protected_feature = protected_feature

    def rank(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        prot_mask = df[self.protected_feature] == True
        unprot_mask = df[self.protected_feature] == False
        
        prot_scores = df.loc[prot_mask, 'scores'].values
        unprot_scores = df.loc[unprot_mask, 'scores'].values
        
        fair_scores = df['scores'].copy()
        
        # Garante que existem membros em ambos os grupos antes de calcular os percentis
        if len(prot_scores) > 0 and len(unprot_scores) > 0:
            for idx in df[prot_mask].index:
                orig_score = df.loc[idx, 'scores']
                # Descobre o percentil do candidato protegido dentro do seu próprio grupo
                p = percentileofscore(prot_scores, orig_score)
                # Pega a qualificação correspondente ao mesmo percentil no grupo não protegido
                new_score = np.percentile(unprot_scores, p)
                fair_scores[idx] = new_score
                
        df['fair_scores'] = fair_scores.tolist()
        
        rank_list = rankdata(fair_scores, method='ordinal')
        df['fair_rank'] = rank_list.tolist()
        
        return df

class OptimizerAgent(Agent):
    def __init__(self,
                 base_demand: float,
                 demand_elasticity: float,
                 sensitivity_installments: float,
                 sensitivity_delivery: float,
                 marketplace_tax: float):
        self.base_demand = base_demand
        self.demand_elasticity = demand_elasticity
        self.sensitivity_installments = sensitivity_installments
        self.sensitivity_delivery = sensitivity_delivery
        self.marketplace_tax = marketplace_tax
        self.max_demand = self.base_demand - (self.demand_elasticity * 0.0001 + \
                   self.sensitivity_delivery * np.log(0.0001) + \
                   self.sensitivity_installments * np.log(0.0001))
        self.min_demand = self.base_demand - (self.demand_elasticity * 1 + \
                   self.sensitivity_delivery * np.log(1) + \
                   self.sensitivity_installments * np.log(1))
        self.max_margin = (1 * self.max_demand) + (self.marketplace_tax * self.max_demand)
        self.min_margin = (0.0001 * self.min_demand)
        #print(f"Min margin: {self.min_margin} Max margin: {self.max_margin}")

    def _normalize_np(self, data, min_val, max_val):
        if max_val == min_val:
            raise ValueError("max_val and min_val cannot be the same.")
        return (data - min_val) / (max_val - min_val)

    def rank(self, data: pd.DataFrame) -> pd.DataFrame:
        scores = []
        scores_baseline = []
        for i in range(len(data.index)):
            row = data.loc[i, :].values.flatten().tolist()
            #row = row[:-1]
            #assert len(self.weights) == len(row)
            t_price = abs((row[0] - 1))
            demand = self.base_demand - (self.demand_elasticity * t_price + \
                     self.sensitivity_delivery * np.log(row[1]) + \
                     self.sensitivity_installments * np.log(row[2]))
            margin = (t_price * demand) + \
                     ((float(row[len(row) - 1]) * self.marketplace_tax) * 
                     demand)
            margin = self._normalize_np(margin, self.min_margin, self.max_margin)
            margin_baseline = t_price * demand
            margin_baseline = self._normalize_np(margin_baseline, self.min_margin, self.max_margin)
            scores.append(((2 * row[0] + margin)/3) * 1000)
            scores_baseline.append(((2 * row[0] * margin_baseline)/3) * 1000)
        scores = self._normalize_np(np.array(scores), self.min_margin, self.max_margin)
        scores_baseline = self._normalize_np(np.array(scores_baseline), self.min_margin, self.max_margin)
        ranking_reference = list(reversed(range(1, len(scores) + 1)))
        rank_list = rankdata(scores, method='ordinal')
        rank_baseline_list = rankdata(scores_baseline, method='ordinal')
        #reversed_rank_list = []
        #reversed_rank_baseline_list = []
        #for i in range(len(scores)):
        #    reversed_rank_list.append(ranking_reference[rank_list[i] - 1])
        #    reversed_rank_baseline_list.append(ranking_reference[rank_baseline_list[i] - 1])
        data['scores'] = scores
        data['rank'] = rank_list
        data['scores_baseline'] = scores_baseline
        data['rank_baseline'] = rank_baseline_list
        return data


class FairAgent(Agent):
    def __init__(self, model: OptimizationModel):
        self.model = model

    def rank(self, data: pd.DataFrame):
        prep_data = data.drop(['premium', 'rank', 'scores'], axis=1)
        prep_data = prep_data.to_numpy()
        print(prep_data.shape)
        predictions = self.model.predict(prep_data)
        #ranking_reference = list(reversed(range(1, len(predictions) + 1)))
        rank_list = rankdata(predictions, method='ordinal')
        #reversed_rank_list = []
        #for i in range(len(predictions)):
        #    reversed_rank_list.append(ranking_reference[rank_list[i] - 1])
        data['fair_scores'] = predictions
        data['fair_rank'] = rank_list
        return data


class FairAgentOptimizer(Agent):
    def __init__(self):
        pass

    def rank(self, data: pd.DataFrame):
        sensi_att = data['premium'].to_numpy()
        input_scores = data['scores'].to_numpy()
        prep_data = data.drop(['premium', 'rank', 'scores'], axis=1)
        prep_data = prep_data.to_numpy()
        pro_index = np.array(np.where(sensi_att == False))[0].flatten()
        unpro_index = np.array(np.where(sensi_att != False))[0].flatten()
        pro_data = prep_data[pro_index, :]
        unpro_data = prep_data[unpro_index, :]
        _accmeasure = "scoreDiff"
        _k = 14

        start_time = time.time()
        print("Starting optimization @ ", _k, "ACCM ", _accmeasure, " time: ", start_time)
        rez, bnd = optimization.initOptimization(prep_data, _k)
        rez = optim.fmin_l_bfgs_b(optimization.lbfgsOptimize, x0=rez, disp=1, epsilon=1e-5,
                                  args=(prep_data, pro_data, unpro_data, input_scores, _accmeasure, _k, 0.01,
                                             1, 100, 0), bounds=bnd, approx_grad=True, factr=1e12, pgtol=1e-04, maxfun=15,
                                       maxiter=50)
        end_time = time.time()
        print("Ending optimization ", "@ ", " warnflag ", rez[2]['warnflag'], _k, "ACCM ", _accmeasure, " time: ", end_time)
        model = OptimizationModel(rez, _k)
        predictions = model.predict(prep_data)
        data['fair_scores'] = predictions
        ranking_reference = list(reversed(range(1, len(predictions) + 1)))
        rank_list = rankdata(predictions, method='ordinal')
        reversed_rank_list = []
        for i in range(len(predictions)):
            reversed_rank_list.append(ranking_reference[rank_list[i] - 1])
        data['fair_rank'] = reversed_rank_list
        return data

class AdversarialMitigationModel:
    def __init__(self, model_path: str):
        input_dim = (14, 4) 
        hidden_sizes = [176, 48, 144]
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.eval()

    def rank(self, data: pd.DataFrame):
        feature_cols = ['price', 'installments', 'delivery_time']
        prep_data = data[feature_cols].copy()
        
        original_len = len(prep_data)
        expected_rows = 14
        
        prep_data['mask'] = 1.0

        if original_len < expected_rows:
            missing_rows = expected_rows - original_len
            pad_data = pd.DataFrame(np.full((missing_rows, prep_data.shape[1]), -1.0), columns=prep_data.columns)
            pad_data['mask'] = 0.0
            prep_data = pd.concat([prep_data, pad_data], ignore_index=True)
        elif original_len > expected_rows:
            prep_data = prep_data.head(expected_rows)

        features_only = prep_data[feature_cols].to_numpy()
        mask_only = prep_data[['mask']].to_numpy()
        
        final_input = np.concatenate([features_only, mask_only], axis=1)
        
        input_tensor = torch.tensor(final_input, dtype=torch.float32).unsqueeze(0) # Batch dimension

        with torch.no_grad():
            predictions = self.model(input_tensor).squeeze().numpy()
        
        if original_len < expected_rows:
            predictions = predictions[:original_len]
        
        rank_list = rankdata(predictions, method='ordinal')

        data['fair_scores'] = predictions
        data['fair_rank'] = rank_list

        return data

class FairlearnAdversarialAgent(Agent):
    def __init__(self, model_path='fairlearn_adv_regressor.pkl', scaler_x_path='fairlearn_scaler_X.pkl', scaler_y_path='fairlearn_scaler_y.pkl'):
        with open(model_path, 'rb') as f:
            self.model = cloudpickle.load(f)
        self.scaler_X = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)

    def rank(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_cols = ['price', 'installments', 'delivery_time']
        features = data[feature_cols].copy()
        
        # Transform features based on the saved scaler
        features_scaled = self.scaler_X.transform(features)
        
        predictions_scaled = self.model.predict(features_scaled)
            
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        rank_list = rankdata(predictions, method='ordinal')
        
        data['fair_scores'] = predictions
        data['fair_rank'] = rank_list
        return data