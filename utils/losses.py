import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXent(nn.Module):
    def __init__(self, batch_size, n_views=2, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.n_views = n_views
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits, labels, pos, neg = self.get_infoNCE_logits_labels(x, self.batch_size, self.n_views, self.temperature)
        return self.criterion(logits, labels), pos, neg

    def get_infoNCE_logits_labels(self, features, batch_size, n_views=2, temperature=0.1):
        """
            Implementation from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        # creates a vector with labels [0, 1, 2, 0, 1, 2]
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        # creates matrix where 1 is on the main diagonal and where indexes of the same intances match (e.g. [0, 4][1, 5] for batch_size=3 and n_views=2)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # computes similarity matrix by multiplication, shape: (batch_size * n_views, batch_size * n_views)
        similarity_matrix = get_cosine_sim_matrix(features)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)  # .to(self.args.device)
        # mask out the main diagonal - output has one column less
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix_wo_diag = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix_wo_diag[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives
        negatives = similarity_matrix_wo_diag[~labels.bool()].view(similarity_matrix_wo_diag.shape[0], -1)

        # reshuffles values in each row so that positive similarity value for each row is in the first column
        logits = torch.cat([positives, negatives], dim=1)
        # labels is a zero vector because all positive logits are in the 0th column
        labels = torch.zeros(logits.shape[0])

        logits = logits / temperature

        return logits, labels.long().to(logits.device), positives.mean(), negatives.mean()


def get_cosine_sim_matrix(features):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    return similarity_matrix