import torch


def dcg_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2
) -> torch.Tensor:
    """Compute discounted cumulative gain at K for provided cutoffs (ignoring ties)

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of discounted cumulative gains at cutoffs
    """
    # Maintain maximum precision to avoid sorting errors with small values
    scores = scores.to(torch.float64)
    labels = labels.to(torch.float64)

    # Create a placeholder for the results
    dcgs = torch.zeros(len(ks)).to(device=scores.device)

    # Order and trim labels to top K using scores and maximum K
    max_k = int(max(ks))
    _, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 0, topk_indices)

    # Compute discounts
    discount_positions = torch.arange(max_k).to(
        device=scores.device, dtype=torch.float64
    )

    discount_log_base = float(
        torch.log(
            torch.tensor([log_base]).to(device=scores.device, dtype=torch.float64)
        )
    )

    discounts = 1 / (torch.log(discount_positions + 2) / discount_log_base)

    # Compute DCGs at K
    for index, k in enumerate(ks):
        dcgs[index] = torch.dot(topk_labels[:k], discounts[:k])

    return dcgs


def ndcg_at(
    ks: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, log_base: int = 2
) -> torch.Tensor:
    """Compute normalized discounted cumulative gain at K for provided cutoffs (ignoring ties)

    Args:
        ks (torch.Tensor or list): list of cutoffs
        scores (torch.Tensor): predicted item scores
        labels (torch.Tensor): true item labels

    Returns:
        torch.Tensor: list of discounted cumulative gains at cutoffs
    """
    # Maintain maximum precision to avoid sorting errors with small values
    scores = scores.to(torch.float64)
    labels = labels.to(torch.float64)

    # Order and trim labels to top K using scores and maximum K
    max_k = int(max(ks))
    topk_scores, topk_indices = torch.topk(scores, max_k)
    topk_labels = torch.gather(labels, 0, topk_indices)

    # Compute discounted cumulative gains
    normalizing_gains = dcg_at(ks, topk_labels, topk_labels)
    gains = dcg_at(ks, topk_scores, topk_labels)

    # Prevent divisions by zero\
    relevant_pos = (normalizing_gains != 0).nonzero()
    irrelevant_pos = (normalizing_gains == 0).nonzero()

    gains[irrelevant_pos] = 0
    gains[relevant_pos] /= normalizing_gains[relevant_pos]

    return gains
