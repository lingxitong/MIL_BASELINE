import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
class AMD_Layer(nn.Module):
    def __init__(
        self,
        dim,
        agent_num=256,
        heads = 8,
    ):
        super().__init__()
        self.dim_head = dim//heads
        self.agent_num = agent_num
        self.denoise = nn.Linear(self.dim_head,self.dim_head)
        self.mask = nn.Linear(self.dim_head,self.dim_head)
        self.get_thresh = nn.Linear(dim,1)
        self.heads = heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.agent = nn.Parameter(torch.randn(heads, agent_num, self.dim_head))

    def forward(self, x , return_WSI_attn = False):
        forward_return = {}
        b, _, _, h = *x.shape, self.heads
        # obtain the qkv matrix
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        agent = self.agent.unsqueeze(0).expand(b,-1,-1,-1)
        # Perform agent calculations
        q = torch.matmul(q,agent.transpose(-1,-2))
        k = torch.matmul(agent,k.transpose(-1,-2))
        softmax = nn.Softmax(dim=-1)
        q *= self.scale
        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v) 
        kv_c = kv.reshape(b,self.agent_num,-1)
        thresh = self.get_thresh(kv_c).squeeze().mean()
        thresh = F.sigmoid(thresh)
        # Perform mask and denoise operations
        denoise = self.denoise(kv)
        denoise = torch.sigmoid(denoise)
        mask = self.mask(kv)
        mask = torch.sigmoid(mask)
        mask = torch.where(mask > thresh, torch.ones_like(mask), torch.zeros_like(mask))
        kv = kv * mask + denoise
        # Obtain weighted features
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        forward_return['amd_out'] = out
        if return_WSI_attn:
            WSI_attn = torch.matmul(q,k)
            forward_return['WSI_attn'] = WSI_attn 
        return forward_return
    
    
if __name__ == "__main__":
    agent = AMD_Layer(dim=512).cuda()
    x = torch.randn(1, 1000, 512).cuda()
    out = agent(x)
    print(out.shape)