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

    batch_loss = 0.0
    for index in range(len(batch_idcgs)):
        cur_batch_idcgs = batch_idcgs[index, :]
        cur_batch_dcg = batch_dcg[index, :]
        cur_approx_nDCG = torch.div(cur_batch_dcg, cur_batch_idcgs)
        batch_loss += (1 - cur_approx_nDCG) / torch.sum(batch_stds[index, :] == 2)

    return batch_loss


class Label_Ranking_Loss(torch.nn.Module):
    def __init__(self, model_para_dict=100):
        super(Label_Ranking_Loss, self).__init__()
        self.alpha = model_para_dict

    def forward(self, batch_preds, batch_stds):
        """
        Call each function to calculate label_ranking_loss

        Args:
            batch_preds: The output of model in a min-batch with virtual neutral labels
            labels: Ground Truth in a min-batch

        Returns:
            batch_loss: label_ranking_loss in a min-batch
        """
        target_batch_stds, batch_sorted_inds = torch.sort(batch_stds, dim=1, descending=True)
        target_batch_preds = torch.gather(batch_preds, dim=1, index=batch_sorted_inds)
        batch_loss = approxNDCG_loss(target_batch_preds, target_batch_stds, self.alpha)

        return batch_loss