import numpy as np

class DropOut(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape) < self.p).to(x.device)
            x = x * mask / (1 - self.p)
        return x
    

class mlp:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.W1 =np.array(in_dim, hidden_dim)
        self.b1 =np.array(hidden_dim)
        self.W2 =np.array(hidden_dim, out_dim)
        self.b2 =np.array(out_dim)
    
    def forward(self, x):
        x=self.W1 @ x + self.b1
        x=max(0,x)
        x=self.W2 @ x + self.b2
        return x
    

class Softmax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.softmax(x, dim=self.dim)
    



class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1/(1+torch.exp(x))
    

class softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = torch.exp(x - torch.max(x,dim=1, keepdim=True))
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)
    

class MHA(nn.Module):
    def __init__(dmodel,dhidden, head):
        super().__init__()
        self.head=head
        self.dmodel=dmodel
        self.dhidden=dhidden
        self.kqv = nn.Linear(d_model,3*d_hidden,bias=False)
        assert self.dhidden%self.head==0, "Hidden dimensions should be divisible by number of heads"
        self.head_dim = dhidden//head

    def forward(self,x):
        # x has shape of B, S, dmodel

        B,S,_ = x.shape
        q, k, v = self.kqv(x).chunk(3, dim=-1)
        q= q.view(B, S, head,self.head_dim)
        k= k.view(B, S, head,self.head_dim)
        v = v.view(B, S, head,self.head_dim)

        q= q.permute(0,2,1,3) # B,head,S,dhidden//head
        k= k.permute(0,2,1,3)

        q= q.view(B*self.head,S,self.d_head).contiguous()
        k= k.view(B*self.head,S,self.d_head)

        att = torch.bmm(q,k.transpose(-1,-2))/torch.sqrt(torch.tensor(self.head_dim).float())

        mask= torch.tril(torch.ones(S, S)).to(x.device).unsqueeze(0)
        attn = toch.masked_fill(mask==0,float('inf'))
        attn = att.softmax(dim=-1,keep)

        out= attn.view(B,self.dim_head,-1,-1) v.permute(0,2,1)
        out = out.view(B, S, -1)

        return out
    


## serialize and deserialize binary tree
def encode(root):
    if root is None:
        return "#,"
    return str(root.val) + "," + encode(root.left) + encode(root.right)




