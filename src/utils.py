import math

import torch


def evaluate(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Evaluate the classification accuracy of a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    X : torch.Tensor
        Input features for the evaluation dataset.
    y : torch.Tensor
        Target labels for the evaluation dataset.
    device : str
        Device on which to perform the evaluation.

    Returns
    -------
    float
        Classification accuracy of the model on the dataset.
    """
    X = X.to(device)
    y = y.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean().item()

    return float(accuracy)


# K-Scheduling


def k_schedule(
    data_size: int,
    loc_round_alpha: float,
) -> list[int]:
    """
    Generate step sizes for local training rounds.

    The step sizes increase according to the specified growth factor until
    the total data size is reached.

    Parameters
    ----------
    data_size : int
        Total number of data samples available.
    loc_round_alpha : float
        Growth factor controlling the increase in local training steps.

    Returns
    -------
    list[int]
        Sequence of local training step sizes.
    """
    head = 0
    count = 1
    schedule: list[int] = []

    while head <= data_size:
        step = math.ceil(count**loc_round_alpha)

        if head < data_size:
            schedule.append(step)
        else:
            schedule.append(data_size)

        head += step
        count += 1

    return schedule


# ByzFL Library Compatibility


def flat_updates_avg(
    updates: list[torch.Tensor | None],
) -> torch.Tensor:
    """
    Flatten model parameter tensors into a single vector.

    Parameters
    ----------
    updates : list[torch.Tensor or None]
        Model parameter tensors to flatten. ``None`` entries are treated
        as empty tensors.

    Returns
    -------
    torch.Tensor
        One-dimensional tensor containing all flattened model parameters.
    """
    flattened = torch.cat(
        [
            update.view(-1) if update is not None else torch.tensor([])
            for update in updates
        ]
    )

    return flattened


def unflat_updates_avg(
    flattened_update: torch.Tensor,
    template_updates: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """
    Restore a flattened update to its original tensor structure.

    Parameters
    ----------
    flattened_update : torch.Tensor
        One-dimensional tensor containing the flattened model update.
    template_updates : list[torch.Tensor or None]
        Template tensors defining the shapes and structure to restore.

    Returns
    -------
    list[torch.Tensor or None]
        Reconstructed tensors with the same structure and shapes as
        ``template_updates``.
    """
    unflattened_updates: list[torch.Tensor | None] = []
    idx = 0

    for original_update in template_updates:
        if original_update is not None:
            update_size = original_update.numel()
            unflat_update = flattened_update[idx : idx + update_size].view(
                original_update.shape
            )
            unflattened_updates.append(unflat_update)
            idx += update_size
        else:
            unflattened_updates.append(None)

    return unflattened_updates


def flat_updates(
    client_gradient_lists: list[list[torch.Tensor | None]],
) -> list[torch.Tensor]:
    """
    Flatten updates from multiple clients into one-dimensional vectors.

    Each client's gradients are flattened independently.

    Parameters
    ----------
    client_gradient_lists : list[list[torch.Tensor or None]]
        Gradient tensors for each client. ``None`` entries are treated
        as empty tensors.

    Returns
    -------
    list[torch.Tensor]
        One-dimensional flattened update vector for each client.
    """
    flattened_updates: list[torch.Tensor] = []

    for client_gradients in client_gradient_lists:
        flat = torch.cat(
            [
                grad.view(-1) if grad is not None else torch.tensor([])
                for grad in client_gradients
            ]
        )
        flattened_updates.append(flat)

    return flattened_updates


def unflat_updates(
    flattened_updates: list[torch.Tensor],
    template_gradients: list[torch.Tensor | None],
) -> list[list[torch.Tensor | None]]:
    """
    Restore multiple flattened updates to their original tensor shapes.

    Parameters
    ----------
    flattened_updates : list[torch.Tensor]
        Flattened update vector for each client.
    template_gradients : list[torch.Tensor or None]
        Template gradients defining the shapes and structure to restore.

    Returns
    -------
    list[list[torch.Tensor or None]]
        Reconstructed gradient tensors for each client.
    """
    unflattened_updates: list[list[torch.Tensor | None]] = []

    for flat_update in flattened_updates:
        unflat_grads: list[torch.Tensor | None] = []
        idx = 0

        for original_grad in template_gradients:
            if original_grad is not None:
                grad_size = original_grad.numel()
                unflat_grad = flat_update[idx : idx + grad_size].view(
                    original_grad.shape
                )
                unflat_grads.append(unflat_grad)
                idx += grad_size
            else:
                unflat_grads.append(None)

        unflattened_updates.append(unflat_grads)

    return unflattened_updates
