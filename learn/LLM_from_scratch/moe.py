import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 4*dim),
                nn.ReLU(),
                nn.Linear(4*dim, dim)
            ) for _ in range(num_experts)
        ])
        self.centroids = nn.Parameter(torch.randn(num_experts, dim))  # e_i

    def forward(self, u_t):
        B, S, dim = u_t.shape
        K = self.k
        N = self.num_experts
        # Total tokens
        T = B * S

        # Flatten tokens
        u_flat = u_t.reshape(-1, dim)  # [B*S, dim]

        # Gating scores
        scores = torch.matmul(u_flat, self.centroids.T)   # [B*S, N]
        gate_scores = torch.softmax(scores, dim=-1)       # g_i,t

        # Top-k experts per token
        topk_vals, topk_idx = torch.topk(gate_scores, K, dim=-1)  # both [B*S, K]

        # Repeat tokens for each selected expert
        flat_idx = topk_idx.reshape(-1)                    # [B*S*K]
        flat_inputs = u_flat.repeat_interleave(K, dim=0)   # [B*S*K, dim]
        flat_outputs = torch.zeros_like(flat_inputs)
        
        # perfect assignment load will be 
        perfect_load=(K*T)/N


        aux_loss = None
        if self.training:
            flat_vals = topk_vals.reshape(-1)
            # One-hot encoding: [B*S*K, N]
            one_hot = F.one_hot(flat_idx, num_classes=N).float()
            # f_i: fraction of tokens assigned to each expert
            f_i = one_hot.sum(dim=0) * N / (K * T)   # [N]
            # P_i: average gating score per expert
            P_i = (one_hot * flat_vals.unsqueeze(1)).sum(dim=0) / T  # [N]
            # Auxiliary loss
            alpha = 1e-2
            aux_loss = alpha * (f_i * P_i).sum()

        # Process per expert
        for expert_id in torch.unique(flat_idx):
            mask = flat_idx == expert_id
            if mask.any():
                flat_outputs[mask] = self.experts[expert_id](flat_inputs[mask])

        # Reshape to [B, S, K, dim]
        expert_outs = flat_outputs.view(B, S, K, dim)
        g_vals = topk_vals.view(B, S, K).unsqueeze(-1)     # [B, S, K, 1]

        # Weighted sum over experts
        out = (g_vals * expert_outs).sum(dim=2)            # [B, S, dim]

        return out, aux_loss


moe=MoELayer(dim=512,num_experts=5,k=2)
x=torch.tensor(torch.rand(2,128,512))
y,aux_loss=moe(x)