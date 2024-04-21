import torch.nn as nn

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

class MeanMIL(nn.Module):
    def __init__(self,args,n_classes=1,dropout=True,act='relu',in_dim = 512):
        super(MeanMIL, self).__init__()
        n_classes = args.General.num_classes
        dropout = args.Model.dropout
        act = args.Model.act
        in_dim = args.Model.in_dim
        head = [nn.Linear(in_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(dropout)]
            
        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x):
        x = self.head(x).mean(axis=1)
        return x
