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


class MaxMIL(nn.Module):
    def __init__(self,args,n_classes=1,dropout=True,act='relu',test=False,in_dim = 512):
        super(MaxMIL, self).__init__()
        n_classes = args.General.num_classes
        dropout = args.Model.dropout.use
        dropout_rate = args.Model.dropout.dropout_rate
        act = args.Model.act
        in_dim = args.Model.in_dim

        head = [nn.Linear(in_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]

        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)

        self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x
