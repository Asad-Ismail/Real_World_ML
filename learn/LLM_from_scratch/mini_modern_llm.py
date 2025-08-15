import torch
import torch.nn as nn

## LLM is (Prenorm->Self Attention ->Prenorm(skip add next and add) -> FFN ->) X N times -> (MOE)*m -> softmax ->Cross entropy

class MHA(nn.Module):
    def __init__(self, dmodel, dhidden, n_head):
        super.__init__(self)
        #q,k,v
        # atten_weights =(q.kT)/torch.sqrt(d)
        # attention = atten_weights @ values
        assert dhidden%n_head==0, "hidden dimension must be dimvisble ny number of heads"
        self.q = nn.Linear(dmodel,dhidden)
        self.k = nn.Linear(dmodel, dhidden)
        self. v = nn.Linear(dmodel, dhidden)
        self.head_dim = dhidden//n_head
        self.heads= n_head
        self.norm= nn.LayerNorm(dmodel)
        self.proj = nn.Linear(dhidden,dmodel)
    def forward(self, x):
        # x is of shape (B,seqLength,dmodel)
        B=x.shape(0)
        x=self.norm(x)
        q= self.q(x)
        k= self.k(x)
        v = self.k(x)
        q=q.reshape(-1,self.head,self.head_dim).transpose(1,2)
        # q is (B, heads,seqLength, perheaddim )
        k= k.reshape(B,-1,self.head,self.head_dim).transpose(1,2)
        # k is (B, heads, seqLength, perheaddim )
        v= v.reshape(B,-1,self.head,self.head_dim).transpose(1,2)
        # v is (B, heads, seqLength, perheaddim )
        atten_weights = (q @ k.transpose(-2,-1))/ (self.head_dim**0.5)
        # attention weights are (B,heads,seq_length,seq_length)
        atten_weights = torch.softmax(atten_weights,dim=1)
        causal_mask = torch.tril((seq_len,seq_len), devide=atten_weights.device)
        atten_weights=atten_weights*causal_mask
        # padding mask
        atten_weights= torch.maskfill(atten_weights, padding_mask,0.0)
        attention = atten_weights @ v
        # attention is (B,heads,seq_length, head_dimension)
        attention= attention.transpose(1,2),reshape(B,seq_len,-1)
        x=self.proj(attention)
        return x



class transofrmer(nn.Module):
    def __init__(self,maxseqlength,dmodel, dhidden, n_head, expansion_factor=4):
        super().__init__()
        self.mha = MHA(dmodel, dhidden, n_head )
        self.ffd = nn.Sequential(nn.Linear(dmodel, dmodel*expansion_factor), nn.GELU(), nn.Linear(dmodel*expansion_factor,dmodel))
        self.dropout =0.1
        self.layernorm1= nn.LayerNorm(dmodel)

    def forward(self,x):
        B,seqlen,_= x.shape
        x_pos= torch.arange(seqlen,device=x.device).unsqueeze(0).expand(B,dim=0)
        # Broadcoast addition
        x=x+x_pos
        mha_out= MHA(self.layernorm1(x))
        x = x + self.dropout(mha_out)
        ff_out = self.ffd(x)
        x=x+self.dropout(ff_out)
        return x

