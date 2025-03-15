import numpy as np
import torch
import torch.nn as nn
from models.diff_models import diff_CD2_TSI


class CD2_TSI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim # 20

        self.freq_interpolation = config["model"]["freq_interpolation"]
        
        config_diff = config["diffusion"]
        config_diff["shared_side_dim"] = config["model"]["timeemb"] + config["model"]["featureemb"] # 128 + 16 
        config_diff["specific_side_dim"] = 1
        config_diff["device"] = device

        input_dim = 2
        self.diffmodel = diff_CD2_TSI(config_diff, inputdim=input_dim)
    
        self.num_steps = config_diff["num_steps"]
        self.beta = np.linspace(
            config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
        ) ** 2
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, shared_side_info, specific_side_info, itp_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss, predicted = self.calc_loss(
                observed_data, cond_mask, observed_mask, shared_side_info, specific_side_info, itp_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps, predicted

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, shared_side_info, specific_side_info, itp_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        if self.config['model']['freq_interpolation']:
            extended_obsedved_data = observed_data * observed_mask + (1-observed_mask)*itp_info
            noisy_data = (current_alpha ** 0.5) * extended_obsedved_data + (1.0 - current_alpha) ** 0.5 * noise
        else:
            noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, self.shared_input_projection, shared_side_info, specific_side_info, t) 

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss, predicted
    
    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, shared_side_info, specific_side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device) 

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data) # noise
            for t in range(self.num_steps - 1, -1, -1): 
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

                predicted = self.diffmodel(diff_input, self.shared_input_projection, shared_side_info, specific_side_info, torch.tensor([t]).to(self.device)) # (B, K, L)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples
    

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp, 
            _,
            _,
            itp_data,
            cond_mask,
        ) = self.process_train_val_data(batch)
        shared_side_info = self.shared_time_feature_embed(observed_tp, observed_mask)
        specific_side_info = cond_mask.unsqueeze(1)
        
        itp_info = None
        if self.freq_interpolation:
            itp_info = itp_data

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss, predicted = loss_func(
            observed_data, cond_mask, observed_mask, shared_side_info, specific_side_info, itp_info, is_train
        ) 
        
        return loss, predicted

    def evaluate(self, batch, n_samples):
        (
            observed_data, 
            observed_mask, 
            observed_tp, 
            gt_mask,
            cut_length,
            _,
        ) = self.process_test_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            
            shared_side_info = self.shared_time_feature_embed(observed_tp, observed_mask)
            specific_side_info = cond_mask.unsqueeze(1)

            samples = self.impute(observed_data, cond_mask, shared_side_info, specific_side_info, n_samples)

            for i in range(len(cut_length)): 
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CD2_TSI(CD2_TSI_base):
    def __init__(self, config, device, shared_input_projection, shared_time_feature_embed, target_dim=20):
        super(CD2_TSI, self).__init__(target_dim, config, device)
        self.config = config
        self.shared_input_projection = shared_input_projection
        self.shared_time_feature_embed = shared_time_feature_embed

    def process_train_val_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        cond_mask = batch["cond_mask"].to(self.device).float()

        itp_data = None
        if self.config['model']['freq_interpolation']:
            itp_data = batch["freq_itp"].to(self.device).float()
            itp_data = itp_data.permute(0, 2, 1)
        
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
            itp_data,
            cond_mask
        )

    def process_test_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        cond_mask = batch["cond_mask"].to(self.device).float()
     
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
            cond_mask
        )
