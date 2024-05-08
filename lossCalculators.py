import torch
from torch.nn import functional as F
from abstractjepa.loss import VICRegLoss

class VICRegLossDotCalculator(VICRegLoss):

    def calculate_variance(self, normalized_expanded_representation):
        std = torch.sqrt(normalized_expanded_representation.var(dim=0) + 0.0001)
        return torch.mean(F.relu(1 - std))

    def calculate_covariance(self, normalized_expanded_representation):
        batch_size = normalized_expanded_representation.shape[0]
        num_features = normalized_expanded_representation.shape[-1]
        cov = (normalized_expanded_representation.T @ normalized_expanded_representation) / (batch_size - 1)
        return off_diagonal(cov).pow_(2).sum().div(num_features)

    def calculate_invariance(self, expanded_representation_yhat, expanded_representation_y):
        return F.mse_loss(expanded_representation_yhat, expanded_representation_y)

    def get_variance_and_covariance_loss(self, expanded_representation):
        normalized_x = expanded_representation - expanded_representation.mean(dim=0)
        variance_loss = self.calculate_variance(normalized_x)
        covariance_loss = self.calculate_covariance(normalized_x)
        return variance_loss, covariance_loss

    def calculate_VICReg_loss(self, encoded_states, predicted_encoded_states):

        expanded_encoded_states = self.expander(encoded_states)
        expanded_predicted_encoded_states = self.expander(predicted_encoded_states)

        num_timesteps = encoded_states.shape[0]

        variance_loss = torch.tensor(0.0).to('mps')
        covariance_loss = torch.tensor(0.0).to('mps')
        invariance_loss = torch.tensor(0.0).to('mps')

        timestep_variance, timestep_covariance = self.get_variance_and_covariance_loss(expanded_encoded_states[0])
        variance_loss += timestep_variance
        covariance_loss += timestep_covariance

        for i in range(num_timesteps-1):
            timestep_variance, timestep_covariance = self.get_variance_and_covariance_loss(expanded_encoded_states[i+1])
            timestep_invariance = self.calculate_invariance(expanded_predicted_encoded_states[i], expanded_encoded_states[i+1])
            variance_loss += timestep_variance
            covariance_loss += timestep_covariance
            invariance_loss += timestep_invariance

        return variance_loss / num_timesteps, covariance_loss / num_timesteps, invariance_loss / (num_timesteps - 1)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



