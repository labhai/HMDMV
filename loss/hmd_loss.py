import torch
import torch.nn as nn
import torch.nn.functional as F

# Uncertainty_Weighting: CrossEntropy + Entropy
class HierarchicalMutualDistillationLoss(nn.Module):
    def __init__(self, dataset, base_temp=4.0, base_lambda=0.1, alpha=1.2):
        super(HierarchicalMutualDistillationLoss, self).__init__()
        self.dataset = dataset
        self.base_temp = base_temp
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.base_lambda = base_lambda

    def forward(self, final_view_logits, cur_view_logits, targets, current_level, total_views):
        temp = self.base_temp
        lambda_hyperparam = self.base_lambda * (current_level ** self.alpha)

        q_i = torch.softmax(cur_view_logits / temp, dim=-1) # prediction distribution (softmax)
        p_target = F.one_hot(targets, num_classes=q_i.size(-1)).float() # one-hot vector of target label
        p_target = p_target.unsqueeze(1).expand(-1, q_i.size(1), -1)
        cross_uncertainty = -torch.sum(p_target * torch.log(q_i + 1e-5), dim=-1)
        entropy_uncertainty = -torch.sum(q_i * torch.log(q_i + 1e-5), dim=-1)

        uncertainty = cross_uncertainty + entropy_uncertainty
        weights = 1 / (uncertainty + 1e-5)
        normalized_weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-5)
        weighted_logits = cur_view_logits * normalized_weights.unsqueeze(-1)
        averaged_cur_logits = torch.sum(weighted_logits, dim=1)

        q = torch.softmax(averaged_cur_logits / temp, dim=1)
        p = torch.softmax(final_view_logits / temp, dim=1)

        log_q = torch.log_softmax(averaged_cur_logits / temp, dim=1)
        log_p = torch.log_softmax(final_view_logits / temp, dim=1)
            
        kl_loss = (1/2) * (self.kl_div(log_p, q.detach()).sum(dim=1).mean() + self.kl_div(log_q, p.detach()).sum(dim=1).mean())
        kl_loss_weighted = kl_loss * (temp ** 2) * lambda_hyperparam

        total_loss = kl_loss_weighted

        return total_loss