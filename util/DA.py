import torch

def compute_similarity(augmented_tgt_pred, tgt_pred, tau_l=0.2, tau_h=0.7):

    difference = (augmented_tgt_pred - tgt_pred).abs().mean()
    if difference < tau_l:
        return torch.tensor(0.0, device=augmented_tgt_pred.device)
    else:
        consistency_loss = torch.clamp(difference - tau_l, tau_h)
        return consistency_loss

def discrepancy_alignment_loss(tgt_pred, tgt_pred_from_src_model, tau_l, tau_h):

    align_loss = compute_similarity(tgt_pred, tgt_pred_from_src_model, tau_l, tau_h)

    return align_loss