import torch
import torch.nn as nn
from .nystrom_attention import NystromAttention
from .multiheadatt import MultiheadLinearAttention
from .linearatt import MultiheadLinearAttention

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout= d
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class CrossLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,d=0.3):
        super().__init__()
        self.attn = MultiheadLinearAttention(embed_dim=dim,num_heads=8,dropout=d)

    def forward(self,q,k,v):

        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        x,attention= self.attn(q,k,v)

        return x.permute(1,0,2), attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout1d(drop)
        self.act2 = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.act2(x)
        return x

class optimizer_triple(nn.Module):
    def __init__(self, in_feature,out_feature,drop=0.):
        super().__init__()
        self.infeature = in_feature
        self.outfeature = out_feature
        self.drop_rate = drop
        #print(self.infeature)
        self.fc1 = nn.Linear(self.infeature,self.outfeature)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout1d(self.drop_rate)

        self.fc2 = nn.Linear(self.outfeature,self.outfeature)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout1d(self.drop_rate)

    def forward(self, x, mode):
        if mode == 'global':
            x = self.fc1(x)
            x = self.act1(x)
            x = self.fc2(x)
            x = self.act2(x)

        else:
            x = self.fc1(x)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.act2(x)
            x = self.drop2(x)
        
        return x

class DGR_MIL(nn.Module):
    def __init__(self, in_dim, num_classes=2, L=512, D=128, n_lesion = 11, attn_mode="gated", dropout_node=0.0,dropout_patch=0.0,initialize=False):
        super().__init__()
        self.L = L
        self.D = D
        self.n_lesion = n_lesion
        self.attn_mode = attn_mode
        self.initialize = initialize
        # global lesion representation learning 
        self.m = 0.4
 
        self.lesionRrepresentation = nn.Parameter(torch.randn(1,self.n_lesion, in_dim))
        self.normalcenter = nn.Parameter(torch.randn(1, self.L),requires_grad=False)
        self.postivecenter = nn.Parameter(torch.randn(1, self.L),requires_grad=False)
        # encoder instances -> 

        self.token = nn.Parameter(torch.randn(1, 1, L))


        self.triple_optimizer = optimizer_triple(in_feature=in_dim,out_feature=self.L,drop=dropout_patch)

        self.encoder_instances = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            nn.LayerNorm(self.L),
        )
        
        # encoder global lesion representation -> 
        self.encoder_globalLesion = nn.Sequential(
            TransLayer(dim=self.L,d=dropout_node),
            nn.LayerNorm(self.L),
        )
              
        
        self.crossffn = nn.Sequential(
            nn.Linear(self.L,self.L),
            nn.LayerNorm(self.L),
        )
 
        self.crossattention =  CrossLayer(dim=self.L,d=dropout_node)
        self.classifier = nn.Sequential(
            nn.Linear(self.L,num_classes)
        )


    def forward(self, x, bag_mode='normal', return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        x = self.triple_optimizer(x,mode='instances')

        H = self.encoder_instances(x)

        lesion_enhacing = self.triple_optimizer(self.lesionRrepresentation,mode='global')

        #x = self.triple_optimizer(x)
        #H = self.encoder_instances(x)
        #lesion_enhacing = self.triple_optimizer(self.lesionRrepresentation)

        lesion_token =  torch.cat((self.token,lesion_enhacing), dim=1)

        lesion = self.encoder_globalLesion(lesion_token)
                                
        

        out,A = self.crossattention(lesion,H,H) # 1 x n x L -> 1 x 5 x n 
        out = self.crossffn(out)
        out = out[:,0,:]
        if return_WSI_attn:
            WSI_attn = A[:,0,:].transpose(0,1)
            print(WSI_attn.shape)
            forward_return['WSI_attn'] = WSI_attn
        if return_WSI_feature:
            forward_return['WSI_feature'] = out
        # print(cls.shape)
        if self.training:
            logits = self.classifier(out)
            forward_return['logits'] = logits
            with torch.no_grad(): 
                if bag_mode == 'normal': 
                    x = x.squeeze(0)  
                    negative_instances = torch.mean(x,dim=0,keepdim=True)
                    self._momentum_update_nc(negative_instances)

                else:
                    x = x.squeeze(0)  
                    postive_instances = torch.mean(x,dim=0,keepdim=True)
                    self._momentum_update_p(postive_instances)
            forward_return['A'] = A
            forward_return['H'] = H
            forward_return['postivecenter'] = self.postivecenter
            forward_return['normalcenter'] = self.normalcenter
            forward_return['lesion_enhacing'] = lesion_enhacing
            return forward_return
        else: 
            logits = self.classifier(out)
            forward_return['logits'] = logits
            return forward_return

    
    @torch.no_grad()
    def _momentum_update_p(self,postive):
        self.postivecenter.data = self.postivecenter.data * self.m + postive.data * (1. - self.m)
    
    @torch.no_grad()
    def _momentum_update_nc(self,negative):
        
        self.normalcenter.data = self.normalcenter.data * self.m + negative.data * (1. - self.m)
            


@torch.no_grad()
def concat_all_gather(tensor):
    
    tensor_gather = [torch.ones_like(tensor) 
                     for _ in range(torch.distributions.get_world.size())]
    
    torch.distributions.all_gather(tensor_gather,tensor,async_op = False)
    
    output = torch.cat(tensor_gather,dim=0)
    
    return output     

if __name__ == "__main__":
    milnet = DGR_MIL(512, attn_mode='linear', dropout_node=0.1)
    logits, A, _, postive, negative,lesion= milnet(torch.randn(1,100, 512),bag_mode='lesion')
    
