import numpy as np
import torch
import torch.nn as nn
from models.diff_models import diff_CSDI


class TSI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim 

        self.emb_time_dim = config["model"]["timeemb"] 
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"] 
        self.target_strategy = config["model"]["target_strategy"]

        self.freq_interpolation = config["model"]["freq_interpolation"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim 
        if self.is_unconditional == False:
            self.emb_total_dim += 1  
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        ) 

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim 
        config_diff["device"] = device

        input_dim = 1 if self.is_unconditional == True else 2 
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"] 
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat) 
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device) 
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        ) # 64
        pe[:, :, 0::2] = torch.sin(position * div_term) 
        pe[:, :, 1::2] = torch.cos(position * div_term)

        return pe


    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) 

        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1) 

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  
        side_info = side_info.permute(0, 3, 2, 1) 
        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  
            side_info = torch.cat([side_info, side_mask], dim=1) 

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  
            loss, predicted, features = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps, predicted, features

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t] 
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, observed_mask, cond_mask, itp_info)
        predicted, features = self.diffmodel(total_input, side_info, t) 

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss, predicted, features

    def set_input_to_diffmodel(self, noisy_data, observed_data, observed_mask, cond_mask, itp_info):
        if self.config['model']['time_interpolation'] or self.config['model']['freq_interpolation']:
            cond_obs = (cond_mask * observed_data + (1-observed_mask)*itp_info).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
        
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)

        return total_input

    def impute(self, observed_data, observed_mask, cond_mask, side_info, n_samples, itp_info):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device) 

        for i in range(n_samples):            
            current_sample = torch.randn_like(observed_data) 

            for t in range(self.num_steps - 1, -1, -1):
                if self.config['model']['freq_interpolation']:
                    cond_obs = (cond_mask * observed_data + (1-observed_mask)*itp_info).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  
                predicted, _ = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device)) # (B, K, L)

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
            gt_mask,
            _,
            coeffs,
            cond_mask,
        ) = self.process_data(batch)

        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None
        if self.freq_interpolation:
            itp_info = coeffs

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        loss, predicted, features = loss_func(
            observed_data, cond_mask, observed_mask, side_info, itp_info, is_train
        )

        return loss, predicted, features, cond_mask

    def evaluate(self, batch, n_samples):
        (
            observed_data, 
            observed_mask, 
            observed_tp, 
            gt_mask,  
            cut_length, 
            coeffs,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask 

            side_info = self.get_side_info(observed_tp, cond_mask) 

            itp_info = None
            if self.freq_interpolation:
                itp_info = coeffs

            samples = self.impute(observed_data, observed_mask, cond_mask, side_info, n_samples, itp_info)


            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CD2_TSI(TSI_base):
    def __init__(self, config, device, target_dim=27):
        super(CD2_TSI, self).__init__(target_dim, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float() 
        observed_mask = batch["observed_mask"].to(self.device).float() 
        observed_tp = batch["timepoints"].to(self.device).float() 
        gt_mask = batch["gt_mask"].to(self.device).float() 
        cut_length = batch["cut_length"].to(self.device).long()

        coeffs = None
        if self.config['model']['freq_interpolation']:
            coeffs = batch["interpolated_data"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)

        if self.config['model']['freq_interpolation']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            cut_length,
            coeffs,
            cond_mask,
        )