import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from scipy.stats import beta
import torch.nn.functional as F




from abc import ABCMeta, abstractmethod


class MLP(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, int(in_size/2/2)),
                                nn.ReLU(True),
                                nn.Linear(int(in_size/2/2),out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return x

class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass


class ReparametrizedGaussian_VI(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """

    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, n_samples=1):
        epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        epsilon = epsilon.to(self.mean.device)
        return self.mean + self.std_dev * epsilon

    def log_prob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.std_dev)
                - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            # n_inputs, n_outputs = self.mean.shape
            dim = 1
            for d in self.mean.shape:
                dim *= d
        elif self.mean.dim() == 0:
            dim = 1
        else:
            dim = len(self.mean)
            # n_outputs = 1

        part1 = dim / 2 * (math.log(2 * math.pi) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return (part1 + part2).unsqueeze(0)

    def set_parameters(self, mu, rho):
        self.mean = mu
        self.rho = rho


class DirichletProcess_VI(nn.Module):
    def __init__(self, trunc, eta, batch_size, dim=1024, n_sample=100):
        super().__init__()
        self.T = trunc
        self.dim = dim
        self.batch_size = batch_size
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.mu = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.dim).uniform_(-0.5, 0.5)) for t in range(self.T)])
        # self.sig = nn.Parameter(torch.stack([torch.eye(self.dim) for _ in range(self.T)]))
        self.rho = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.dim).uniform_(-0.5, 0.5)) for t in range(self.T)])
        self.gaussians = [ReparametrizedGaussian_VI(self.mu[t], self.rho[t]) for t in range(self.T)]
        self.phi = torch.ones([self.dim, self.T]) / self.T

        self.eta = eta
        self.gamma_1 = torch.ones(self.T)
        self.gamma_2 = torch.ones(self.T) * eta

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        pi = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return pi

    def entropy(self):
        entropy = [self.gaussians[t].entropy() for t in range(self.T)]
        entropy = torch.stack(entropy, dim=-1)

        return entropy

    def get_log_prob(self, x):
        pdfs = [self.gaussians[t].log_prob(x) for t in range(self.T)]
        pdfs = torch.stack(pdfs, dim=-1)
        return pdfs

    def sample_beta(self, size):
        a = self.gamma_1.detach().cpu().numpy()
        b = self.gamma_2.detach().cpu().numpy()

        samples = beta.rvs(a, b, size=(size, self.T))
        samples = torch.from_numpy(samples)

        return samples

    def forward(self, x):
        x = x.squeeze(0)
        batch_size = x.shape[0]

        beta = self.sample_beta(batch_size)
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        entropy = self.entropy()
        entropy = entropy.expand(batch_size, -1)

        phi_new, kl_gaussian = self.get_phi(torch.log(pi), entropy, log_pdfs)

        self.update_gamma()

        likelihood = phi_new * kl_gaussian
        likelihood = likelihood.sum(1).mean(0)

        self.phi = phi_new.data

        return likelihood

    def infer(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """
        Get logit

        return: Logits with length T
        """
        x = x.squeeze(0)
        forward_return = {}
        beta = self.sample_beta(x.shape[0])
        pi = self.mix_weights(beta)[:, :-1]
        log_pdfs = self.get_log_prob(x)
        pi = pi.to(x.device)
        logits = torch.log(pi) + log_pdfs
        logits = logits.mean(0)
        assert not torch.isnan(logits).any()
        # logits = F.normalize(logits, dim=2)
        forward_return['logits'] = logits.unsqueeze(0)
        
        return forward_return

    def get_phi(self, log_pi, entropy, log_pdf):
        # maybe mention this in the paper we do this to improve numerical stability
        kl_gaussian = log_pdf + entropy
        kl_pi = log_pi

        N_t_gaussian = kl_gaussian.sum(0, keepdim=True)
        N_t_pi = kl_pi.sum(0, keepdim=True)
        mix = (N_t_pi / (N_t_gaussian + N_t_pi))

        kl = mix * kl_gaussian + (1 - mix) * kl_pi

        return kl.softmax(dim=1), mix * kl_gaussian

    def update_gamma(self):
        phi = self.phi

        phi_flipped = torch.flip(phi, dims=[1])
        cum_sum = torch.cumsum(phi_flipped, dim=1) - phi_flipped
        cum_sum = torch.flip(cum_sum, dims=[1])

        self.gamma_1 = 1 + phi.mean(0)
        self.gamma_2 = self.eta + cum_sum.mean(0)

    def standardize(self, x):
        x = (x - x.mean(1)) / x.std(1)
        return x

    def update_mean(self, phi):
        N = phi.sum(0)
        pass

    def update_variance(self):
        pass


class CDP_MIL(nn.Module):

    def __init__(self,in_dim,num_classes, eta, batch_size=1, n_sample=100):
        super().__init__()
        self.dp_process = DirichletProcess_VI(num_classes, eta, batch_size, in_dim)

    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        # neg_likelyhood = -self.dp_process(x)
        return self.dp_process.infer(x, return_WSI_attn, return_WSI_feature)


if __name__ == '__main__':
    cdp_mil = CDP_MIL(1024, 2, 0.1)
    x = torch.randn(10, 1024)
    y = cdp_mil(x)
