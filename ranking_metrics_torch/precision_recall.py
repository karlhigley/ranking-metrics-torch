import torch


def precision_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute precision@K for each of the provided cutoffs

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of precisions at cutoffs
    """

    # Create a placeholder for the results
    precisions = torch.zeros(len(ks)).to(device=scores.device)

    # Order and trim labels to top K using scores and maximum K
    max_k = max(ks)
    _, topk_indices = torch.topk(scores, int(max_k))
    topk_labels = torch.gather(labels, 0, topk_indices)

    # Compute precisions at K
    for index, k in enumerate(ks):
        precisions[index] = topk_labels[: int(k)].sum() / float(k)

    return precisions


def recall_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Compute recall@K for each of the provided cutoffs

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of recalls at cutoffs
    """

    # Create a placeholder for the results
    recalls = torch.zeros(len(ks)).to(device=scores.device)

    # Order and trim labels to top K using scores and maximum K
    max_k = max(ks)
    _, topk_indices = torch.topk(scores, int(max_k))
    topk_labels = torch.gather(labels, 0, topk_indices)

    # Compute recalls at K
    num_relevant = labels.sum()

    if num_relevant > 0:
        for index, k in enumerate(ks):
            recalls[index] = topk_labels[: int(k)].sum() / num_relevant

    return recalls
