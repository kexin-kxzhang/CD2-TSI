import torch
import torch.nn as nn

class SharedEmbedding(nn.Module):
    def __init__(self, config, device, target_dim):
        super(SharedEmbedding, self).__init__()
        self.device = device
        self.target_dim = target_dim    
        self.emb_feature_dim = config["model"]["featureemb"] 
        self.emb_time_dim = config["model"]["timeemb"] 
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        ) 

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device) 
        position = pos.unsqueeze(2) 
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        ) # 64
        pe[:, :, 0::2] = torch.sin(position * div_term) 
        pe[:, :, 1::2] = torch.cos(position * div_term) 

        return pe
    
    def forward(self, observed_tp, observed_mask):
        B, K, L = observed_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B, L, emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)  # (B, L, K, emb_time_dim)
        
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))  # (K, emb_feature_dim)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # (B, L, K, emb_feature_dim)
        
        shared_side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B, L, K, emb_time_dim + emb_feature_dim)
        shared_side_info = shared_side_info.permute(0, 3, 2, 1)  # (B, emb_time_dim + emb_feature_dim, K, L)
        
        return shared_side_info

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer