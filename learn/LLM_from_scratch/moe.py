import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer_Loss(nn.Module):
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

        aux_loss = None
        maxvio = None

        if self.training:
            flat_vals = topk_vals.reshape(-1)
            one_hot_for_loss = F.one_hot(flat_idx, num_classes=N).float()
            f_i = one_hot_for_loss.sum(dim=0)
            f_i_norm = (f_i * N) / (K * T)
            P_i = (one_hot_for_loss * flat_vals.unsqueeze(1)).sum(dim=0) / T
            alpha = 1e-2
            aux_loss = alpha * (f_i_norm * P_i).sum()

            with torch.no_grad():
                perfect_load = (K * T) / N
                # We can reuse f_i from above, detached from the graph
                maxfi, _ = torch.max(f_i, dim=0)
                maxvio = (maxfi - perfect_load) / perfect_load

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
        return out, maxvio, aux_loss
    

class MoELayer_LossLess(nn.Module):
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
        self.centroids = nn.Parameter(torch.randn(num_experts, dim))
        # A buffer is part of the model's state but not considered a model parameter
        # by the optimizer. We initialize it with zeros for a neutral start.
        self.register_buffer("score_biases", torch.zeros(num_experts))
        self.bias_lr = 0.01

    def forward(self, u_t):
        B, S, dim = u_t.shape
        K = self.k
        N = self.num_experts
        T = B * S

        u_flat = u_t.reshape(-1, dim)

        scores = torch.matmul(u_flat, self.centroids.T)
        scores = scores + self.score_biases
        gate_scores = torch.softmax(scores, dim=-1)

        topk_vals, topk_idx = torch.topk(gate_scores, K, dim=-1)

        flat_idx = topk_idx.reshape(-1)
        flat_inputs = u_flat.repeat_interleave(K, dim=0)
        flat_outputs = torch.zeros_like(flat_inputs)

        # The `no_grad` block is correct, as we are doing a manual update
        # that should not be tracked by autograd.
        with torch.no_grad():
            perfect_load = (K * T) / N
            one_hot = F.one_hot(flat_idx, num_classes=N).float()
            # f_i is the actual load for each expert in this batch
            f_i = one_hot.sum(dim=0)
            error = perfect_load - f_i 
            # We use -= because a positive error (overloaded) should decrease the bias.
            self.score_biases += self.bias_lr * torch.sign(error)
            # Calculation of maxvio for monitoring purposes
            maxfi,_ = torch.max(f_i,dim=0)
            maxvio = (maxfi - perfect_load) / perfect_load

        for expert_id in torch.unique(flat_idx):
            mask = flat_idx == expert_id
            if mask.any():
                flat_outputs[mask] = self.experts[expert_id](flat_inputs[mask])

        expert_outs = flat_outputs.view(B, S, K, dim)
        g_vals = topk_vals.view(B, S, K).unsqueeze(-1)

        out = (g_vals * expert_outs).sum(dim=2)

        # The second return value is now just for monitoring, not for a loss function
        return out, maxvio

if __name__=="__main__":
    moe=MoELayer_Loss(dim=512,num_experts=5,k=2)
    #moe=MoELayer_LossLess(dim=512,num_experts=5,k=2)
    x=torch.tensor(torch.rand(2,128,512))
    y,max_vio,aus_loss=moe(x)
    #y,max_vio,aus_loss=moe(x)