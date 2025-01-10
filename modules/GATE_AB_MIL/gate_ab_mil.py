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
class GATE_AB_MIL(nn.Module):
    def __init__(self,L = 512, D = 128, num_classes=2,act=nn.ReLU(),dropout=0,bias=False,in_dim = 512):
        super(GATE_AB_MIL, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.in_dim = in_dim
        self.L = L
        self.D = D 
        self.K = 1
        self.act = act
        self.feature = [nn.Linear(in_dim, 512)]
        self.feature += [self.act]
        self.feature += [nn.Dropout(self.dropout)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.num_classes,bias=bias),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        A_ori = A.clone()

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        logits = self.classifier(x)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = x
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return




