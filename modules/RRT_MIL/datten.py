import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D,bias=bias)]

        if act == 'gelu': 
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K,bias=bias)]

        self.attention = nn.Sequential(*self.attention)

    def forward(self,x,no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)
        
        if no_norm:
            return x,A_ori
        else:
            return x,A

class AttentionGated(nn.Module):
    def __init__(self,input_dim=512,act='relu',bias=False,dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

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

    def forward(self, x,no_norm=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        if no_norm:
            return x,A_ori
        else:
            return x,A

class DAttention(nn.Module):
    def __init__(self,input_dim=512,act='relu',gated=False,bias=False,dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim,act,bias,dropout)
        else:
            self.attention = Attention(input_dim,act,bias,dropout)


 # Modified by MAE@Meta
    def masking(self, x, ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert ids_shuffle is not None

        _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,no_norm=False,mask_enable=False):

        if mask_enable and mask_ids is not None:
            x, _,_ = self.masking(x,mask_ids,len_keep)

        x,attn = self.attention(x,no_norm)

        if return_attn:
            return x.squeeze(1),attn.squeeze(1)
        else:   
            return x.squeeze(1)
        
class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
class DSMIL(nn.Module):
    def __init__(self,n_classes=2,mask_ratio=0.,mlp_dim=512,cls_attn=True,attn_index='max'):
        super(DSMIL, self).__init__()

        self.i_classifier = nn.Sequential(
            nn.Linear(mlp_dim, n_classes))
        self.b_classifier = BClassifier(mlp_dim,n_classes)

        self.cls_attn = cls_attn
        self.attn_index = attn_index

        self.mask_ratio = mask_ratio

    def attention(self,x,no_norm=False,label=None,criterion=None,return_attn=False):
        ps = x.size(1)
        feats = x.squeeze(0)
        classes = self.i_classifier(feats)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        classes_bag,_ = torch.max(classes, 0) 

        if return_attn:
            # 通过bag和inst综合判断
            if self.attn_index == 'max':
                attn,_ = torch.max(classes,-1) if self.cls_attn else torch.max(A,-1)
                # pred = 0.5*torch.softmax(prediction_bag,dim=-1)+0.5*torch.softmax(classes_bag,dim=-1)
                # _,_attn_idx = torch.max(pred.squeeze(),0)
            #     _,_attn_idx = torch.max(classes_bag,0)
            #     attn = classes[:,int(_attn_idx)] if self.cls_attn else A[:,int(_attn_idx)]
            elif self.attn_index == 'label':
                if label is None:
                    pred = 0.5*torch.softmax(prediction_bag,dim=-1)+0.5*torch.softmax(classes_bag,dim=-1)
                    _,_attn_idx = torch.max(pred.squeeze(),0)
                    attn = classes[:,int(_attn_idx)] if self.cls_attn else A[:,int(_attn_idx)]
                else:
                    attn = classes[:,label[0]] if self.cls_attn else A[:,label[0]]
            else:
                attn = classes[:,int(self.attn_index)] if self.cls_attn else A[:,int(self.attn_index)]
            attn = attn.unsqueeze(0)
        else:
            attn = None

        if self.training and criterion is not None:
            # if isinstance(loss,nn.CrossEntropyLoss):
            max_loss = criterion(classes_bag.view(1,-1),label)
            # elif isinstance(loss,nn.BCEWithLogitsLoss):
            #     max_loss = loss(classes.view(1, -1), label.view(1, -1).float())
            # 
            return prediction_bag,attn,B,max_loss
        else:
            return prediction_bag,attn,B,classes_bag.unsqueeze(0)
    
    def random_masking(self, x, mask_ratio,ids_shuffle=None,len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        N, L, D = x.shape  # batch, length, dim
        if ids_shuffle is None:
            # sort noise for each sample
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        else:
            _,ids_restore = ids_shuffle.sort()

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False,no_norm=False,mask_enable=False,**kwargs):
  
        if mask_enable and (self.mask_ratio > 0. or mask_ids is not None):
            x, _,_ = self.random_masking(x,self.mask_ratio,mask_ids,len_keep)

        _label = kwargs['label'] if 'label' in kwargs else None    
        _criterion = kwargs['criterion'] if 'criterion' in kwargs else None

        prediction_bag, attn, B, other = self.attention(x, no_norm, _label, _criterion,return_attn=return_attn)

        logits = prediction_bag

        if return_attn:
            return B, logits, other, attn
        else:   
            return B, logits, other