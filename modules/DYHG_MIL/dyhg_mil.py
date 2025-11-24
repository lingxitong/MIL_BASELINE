import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  
        return A, x


class DYHG_MIL(nn.Module):
    def __init__(self, in_dim=1024, emb_dim=256, num_classes=2, dropout=0.25, hyper_num=20, num_layers=1, tau=0.05, act=nn.ReLU()):
        super().__init__()
        self.message_dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hyper_num = hyper_num
        self.act = nn.LeakyReLU()
        self.norm = nn.LayerNorm(emb_dim)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, emb_dim), nn.ReLU(), nn.Dropout(dropout))
        self.tau = tau  
        self.num_classes = num_classes
        size = [emb_dim, emb_dim, 256]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], num_classes)

        self.construct_hyper = nn.Sequential(
            nn.Linear(emb_dim, self.hyper_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.hyper_num)
        )
        
        self.apply(initialize_weights)
    
    def gumbel_softmax_sample(self, logits, temperature=1.0, hard=False):
        return F.gumbel_softmax(logits, tau=temperature, hard=hard)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        # Handle input format: x should be (1, N, D) or (N, D)
        if isinstance(x, dict):
            x = x.get("feature", x)
        
        # Handle 2D input (N, D) -> (1, N, D)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        x = self._fc1(x)  # [B, N, C]
        features = [x]
        B, N, C = x.shape

        patchs_hyper = self.construct_hyper(x)
        patchs_hyper = self.gumbel_softmax_sample(patchs_hyper, temperature=self.tau, hard=False)  # [N, H]
        hyperedge_feature = self.act(patchs_hyper.permute(0, 2, 1) @ x)
        x = self.act(patchs_hyper @ hyperedge_feature)
        x = self.message_dropout(x)
        x = F.normalize(x, p=2, dim=1)
        features.append(x)       

        features = torch.stack(features, 1)
        features = torch.sum(features, dim=1).squeeze(1)
        features = features / (self.num_layers + 1)
        features = self.message_dropout(features)
        features = self.norm(features)

        A, h = self.attention_net(features.squeeze(0))
        A_ori = A.clone()
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, h)  # n_classes, dim
        
        logits = self.classifiers(M)
        forward_return["logits"] = logits
        if return_WSI_attn:
            forward_return["WSI_attn"] = A_ori
        if return_WSI_feature:
            forward_return["WSI_feature"] = M

        return forward_return

