import torch.nn.functional as F
import torch.nn as nn
import torch

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_k1 = F.kl_div(p, q, size_average=False) / y.shape[0]
    return l_k1

def mixup_criterion(criterion, pred, label1, label2, batch_weight):
    return batch_weight*criterion(pred, label1) + (1-batch_weight)*criterion(pred, label2)



def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1),
                      x1,
                      x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    return res.clamp_min_(1e-30)

def gaussian_kernel(x, y):
    # gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    D = my_cdist(x, y)
    K = torch.zeros_like(D)

    for g in gamma:
        K.add_(torch.exp(D.mul(-g)))
    return K

def mmd(x, y, kernel_type='gaussian'):
    if kernel_type == "gaussian":
        Kxx = gaussian_kernel(x, x).mean()
        Kyy = gaussian_kernel(y, y).mean()
        Kxy = gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))