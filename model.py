import numpy as np
import torch.nn as nn
from optimization import distances, M_nk, calculateWeightedScores

#Mnk_x, _inputscores, clusters, user_N, _k, _accmeasure
#_M_nk_x, _inputscores, _clusters, _N, _k, _accmeasure

class OptimizationModel:
    def __init__(self, rez: tuple, k: int):
        self.x = rez[0]
        self.f = rez[1]
        self.warnflag = rez[2]['warnflag']
        self.grad = rez[2]['grad']
        self.nit = rez[2]['nit']
        self.k = k

    def predict(self, data: np.array):
        user_N, att_N = data.shape
        clusters = np.matrix(self.x[(2 * att_N) + self.k:]).reshape((self.k, att_N))
        alpha1 = self.x[att_N: 2 * att_N]
        # get the distance between input user X and intermediate clusters Z
        dists_x = distances(data, clusters, alpha1, user_N, att_N, self.k)
        # compute the probability of each X maps to Z
        Mnk_x = M_nk(dists_x, user_N, self.k)
        score_hat = np.zeros(user_N)  # initialize the estimated scores
        # calculate estimate score of each user by mapping probability between X and Z
        for ui in range(user_N):
            score_hat_u = 0.0
            for ki in range(self.k):
                score_hat_u += (Mnk_x[ui, ki] * clusters[ki])
            score_hat[ui] = calculateWeightedScores(score_hat_u)

        score_hat = list(score_hat)

        return score_hat

class PredictorNetwork(nn.Module):
    def __init__(self, input_shape, hidden_sizes, output_size=14):
        super(PredictorNetwork, self).__init__()
        input_size = input_shape[0] * input_shape[1]  # Flattened input size
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Dropout for regularization
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))  # Final layer with no activation
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.model(x)
