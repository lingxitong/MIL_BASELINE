import torch.nn as nn
from utils.process_utils import get_act
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

class MEAN_MIL(nn.Module):
    def __init__(self,num_classes=2,dropout=True,act=nn.ReLU() ,in_dim = 512):
        super(MEAN_MIL, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        self.act = act
        self.in_dim = in_dim
        head = [nn.Linear(self.in_dim,512)]

        head+=[act,]

        if dropout:
            head += [nn.Dropout(self.dropout)]
            
        self.classifier = nn.Linear(512,self.num_classes)
        
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x,return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        all_feartures = self.head(x)
        features_cls = self.classifier(all_feartures)
        logits = features_cls.mean(axis=1)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = all_feartures.mean(axis=1)
        if return_WSI_attn:
            WSI_attn = features_cls.mean(axis=2).transpose(0,1)
            forward_return['WSI_attn'] = WSI_attn
        return forward_return
