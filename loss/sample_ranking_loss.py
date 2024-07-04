import torch
import numpy as np
import torch.nn.functional as F


# compute ideal DCG
def torch_dcg_at_k(batch_sorted_labels, cutoff=None):
    if cutoff is None:
        cutoff = batch_sorted_labels.size(1)
    batch_numerators = torch.pow(2.0, batch_sorted_labels[:, 0:cutoff]) - 1.0
    batch_discounts = torch.log2(torch.arange(cutoff).type(torch.cuda.FloatTensor).expand_as(batch_numerators) + 2.0)
    batch_dcg_at_k = torch.sum(batch_numerators / batch_discounts, dim=1, keepdim=True)
    return batch_dcg_at_k


# compute the approximation of rank
def get_approx_ranks(input, alpha=10):
    batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)
    batch_indicators = torch.sigmoid(alpha * torch.transpose(batch_pred_diffs, dim0=1, dim1=2))
    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5
    return batch_hat_pis


def approxNDCG_loss(batch_preds=None, batch_stds=None, alpha=10):
    batch_hat_pis = get_approx_ranks(batch_preds, alpha=alpha)
    batch_idcgs = torch_dcg_at_k(batch_sorted_labels=batch_stds, cutoff=None)
    batch_gains = torch.pow(2.0, batch_stds) - 1.0
    batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1).unsqueeze(dim=1)

    no_label_index = (batch_idcgs == 0).nonzero()
    while len(no_label_index) > 0:
        i = no_label_index[0, :][0]
        batch_idcgs = torch.cat((batch_idcgs[0:i, :], batch_idcgs[i + 1:, :]))
        batch_dcg = torch.cat((batch_dcg[0:i, :], batch_dcg[i + 1:, :]))
        no_label_index = (batch_idcgs == 0).nonzero()

    batch_approx_nDCG = torch.div(batch_dcg, batch_idcgs)
    batch_loss = torch.sum(1 - batch_approx_nDCG)

    return batch_loss


class Sample_Ranking_Loss(torch.nn.Module):
    def __init__(self, model_para_dict=100):
        super(Sample_Ranking_Loss, self).__init__()
        self.alpha = model_para_dict

    def forward(self, preds, labels):
        """
        Call each function to calculate sample_ranking_loss

        Args:
            preds: The output of model in a min-batch with virtual neutral samples
            labels: Ground Truth in a min-batch

        Returns:
            batch_loss: sample_ranking_loss in a min-batch
        """
        batch_preds = torch.transpose(preds, dim0=0, dim1=1)
        batch_stds = torch.transpose(labels, dim0=0, dim1=1)
        target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
        target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)
        batch_loss = approxNDCG_loss(target_batch_preds, target_batch_stds, self.alpha)

        return batch_loss
