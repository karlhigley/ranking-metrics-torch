import torch

from ranking_metrics_torch.precision_recall import precision_at


def avg_precision_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute average precision at K for provided cutoffs

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of average precisions at cutoffs
    """

    # Create a placeholder for the results
    avg_precisions = torch.zeros(len(ks)).to(device=scores.device)

    # Order and trim labels to top K using scores and maximum K
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 0, topk_indices)

    # Compute average precisions at K
    total_relevant = labels.sum()

    if total_relevant > 0:
        for index, k in enumerate(ks):
            relevant_pos = (topk_labels[:k] != 0).nonzero()

            precisions = precision_at(relevant_pos + 1, topk_scores, topk_labels)
            num_relevant = total_relevant if total_relevant <= k else k

            avg_precisions[index] = precisions.sum() / float(num_relevant)

    return avg_precisions
