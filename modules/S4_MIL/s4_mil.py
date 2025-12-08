"""
S4_MIL - S4-based Multiple Instance Learning
Reference: https://github.com/isyangshu/MambaMIL/blob/main/models/S4MIL.py
Paper: Efficiently Modeling Long Sequences with Structured State Spaces (ICLR 2022)

Note: This implementation requires the S4 library. For a simpler version without S4,
you can use MAMBA_MIL which provides similar functionality.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class CPUFFTFunction(Function):
    """
    Custom autograd Function to perform FFT on CPU to avoid cuFFT errors
    while maintaining gradient flow for GPU training.
    
    This function performs convolution via FFT on CPU, then moves results back to GPU.
    PyTorch's autograd will automatically handle the gradient computation.
    """
    @staticmethod
    def forward(ctx, u, k, L):
        """
        Args:
            u: (B, H, L) input tensor (can be on GPU)
            k: (H, L) kernel tensor (can be on GPU)
            L: sequence length
        Returns:
            y: (B, H, L) output tensor on same device as u
        """
        # Store original device and requires_grad flags
        device = u.device
        u_requires_grad = u.requires_grad
        k_requires_grad = k.requires_grad
        
        # Move to CPU for FFT
        u_cpu = u.cpu()
        k_cpu = k.cpu()
        
        # FFT on CPU
        k_f = torch.fft.rfft(k_cpu, n=2*L)
        u_f = torch.fft.rfft(u_cpu.to(torch.float32), n=2*L)
        y_cpu = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]
        
        # Save for backward
        ctx.save_for_backward(u_cpu, k_cpu, k_f, u_f)
        ctx.L = L
        ctx.device = device
        ctx.u_requires_grad = u_requires_grad
        ctx.k_requires_grad = k_requires_grad
        
        # Move back to original device
        return y_cpu.to(device)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: also use CPU for FFT to avoid cuFFT errors
        """
        u_cpu, k_cpu, k_f, u_f = ctx.saved_tensors
        L = ctx.L
        device = ctx.device
        
        # Move gradient to CPU
        grad_output_cpu = grad_output.cpu()
        
        # Backward FFT on CPU
        # For convolution y = ifft(fft(u) * fft(k)), the gradients are:
        # grad_u = ifft(fft(grad_y) * conj(fft(k)))
        # grad_k = ifft(fft(grad_y) * conj(fft(u)))
        grad_y_f = torch.fft.rfft(grad_output_cpu.to(torch.float32), n=2*L)
        
        # Compute gradients
        grad_u_f = grad_y_f * torch.conj(k_f)
        grad_k_f = grad_y_f * torch.conj(u_f)
        
        grad_u_cpu = torch.fft.irfft(grad_u_f, n=2*L)[..., :L]
        grad_k_cpu = torch.fft.irfft(grad_k_f, n=2*L)[..., :L]
        
        # Move gradients back to original device
        grad_u = grad_u_cpu.to(device) if ctx.u_requires_grad else None
        grad_k = grad_k_cpu.to(device) if ctx.k_requires_grad else None
        
        return grad_u, grad_k, None

class DropoutNd(nn.Module):
    """Multi-dimensional dropout"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # Apply dropout across all dimensions except batch
        shape = x.shape
        x = x.reshape(shape[0], -1)
        x = F.dropout(x, p=self.p, training=self.training)
        return x.reshape(shape)

class S4DKernel(nn.Module):
    """S4 Diagonal Kernel - generates diagonal SSM parameters"""
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        # Store complex C as two separate parameters (real and imag)
        C_real = torch.randn(H, N // 2)
        C_imag = torch.randn(H, N // 2)
        self.C_real = nn.Parameter(C_real)
        self.C_imag = nn.Parameter(C_imag)
        self.register_parameter('log_dt', nn.Parameter(log_dt))
        
        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * torch.arange(N//2, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        self.register_parameter('log_A_real', nn.Parameter(log_A_real))
        self.register_buffer('A_imag', A_imag)
    
    def forward(self, L):
        dt = torch.exp(self.log_dt)  # (H)
        # Reconstruct complex C
        C = torch.complex(self.C_real, self.C_imag)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        
        dtA = A * dt.unsqueeze(-1)  # (H N)
        # Use float32 for arange to avoid dtype issues
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device, dtype=torch.float32).to(dtA.dtype)  # (H N L)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K

class S4D(nn.Module):
    """S4D Layer - Diagonal State Space Model"""
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed
        
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n)
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()
        
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )
    
    def forward(self, u):
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)
        
        # Compute kernel
        k = self.kernel(L=L)  # (H, L)
        
        # Use custom CPU FFT function to avoid cuFFT errors
        # This allows the rest of the model to train on GPU
        y = CPUFFTFunction.apply(u, k, L)
        
        # Skip connection
        y = y + u * self.D.unsqueeze(-1)
        
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y

class S4_MIL(nn.Module):
    """
    S4-based Multiple Instance Learning
    
    Uses Structured State Space Models (S4) for efficient long sequence modeling.
    Linear complexity makes it suitable for long patch sequences.
    
    Reference: https://github.com/isyangshu/MambaMIL/blob/main/models/S4MIL.py
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(), 
                 d_model=512, d_state=32, n_layers=1):
        super(S4_MIL, self).__init__()
        
        # Convert act to string
        if isinstance(act, nn.ReLU):
            act_str = 'relu'
        elif isinstance(act, nn.GELU):
            act_str = 'gelu'
        else:
            act_str = 'relu'
        
        # Feature projection
        self._fc1 = [nn.Linear(in_dim, d_model)]
        if act_str.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act_str.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)
        
        # S4 layers
        self.s4_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    S4D(d_model=d_model, d_state=d_state, dropout=dropout, transposed=False)
                )
            )
        
        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        h = x.float()  # [B, n, in_dim]
        h = self._fc1(h)  # [B, n, d_model]
        
        # Apply S4 layers
        for s4_layer in self.s4_layers:
            h_ = h
            h = s4_layer(h)
            h = h + h_  # Residual connection
        
        # Attention aggregation
        A = self.attention(h)  # [B, n, 1]
        A_ori = A.clone()
        A = torch.transpose(A, 1, 2)  # [B, 1, n]
        A = F.softmax(A, dim=-1)  # [B, 1, n]
        h = torch.bmm(A, h)  # [B, 1, d_model]
        h = h.squeeze(1)  # [B, d_model] - keep batch dimension
        
        # Classification
        logits = self.classifier(h)  # [B, num_classes]
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = h.unsqueeze(0)
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori.squeeze(0).squeeze(-1)
        
        return forward_return

