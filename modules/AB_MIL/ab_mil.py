import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()

        self.model = list(models.resnet50(pretrained = True).children())[:-1]
        self.features = nn.Sequential(*self.model)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.classifier = nn.Linear(512,1)
        initialize_weights(self.feature_extractor_part2)
        initialize_weights(self.classifier)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x=self.feature_extractor_part2(x)
        # feat = torch.mean(x,dim=0)
        x1 = self.classifier(x)
        # x2 = torch.mean(x1, dim=0).view(1,-1)
        x2,_ = torch.max(x1, dim=0)
        x2=x2.view(1,-1)
        return x2,x

class ABMIL(nn.Module):
    def __init__(self,args,n_classes=1,dropout=True,act='relu',in_dim = 512):
        super(ABMIL, self).__init__()
        n_classes = args.General.num_classes
        dropout = args.Model.dropout
        act = args.Model.act
        in_dim = args.Model.in_dim
        self.L = 512 #512
        self.D = 128 #128
        self.K = 1
        self.feature = [nn.Linear(in_dim, 512)]
        
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]

        if dropout:
            self.feature += [nn.Dropout(dropout)]

        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.apply(initialize_weights)
    def forward(self, x, return_attn=False,no_norm=False):
        feature = self.feature(x)

        # feature = group_shuffle(feature)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.mm(A, feature)  # KxL
        Y_prob = self.classifier(M)

        if return_attn:
            if no_norm:
                return Y_prob,A_ori
            else:
                return Y_prob,A
        else:
            return Y_prob

