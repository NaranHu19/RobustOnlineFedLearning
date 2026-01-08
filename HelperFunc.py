# General Imports
import torch
import math

def evaluate(model, X, y, device):
    """Computes the classification accuracy of a model on a given dataset (X,y) using the specified hardware device"""

    X = X.to(device)
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean().item()
    return accuracy

##### K_Scheduling #####

def k_schedule(data_size, loc_round_alpha): 
    """Generates a list of step sizes for local training rounds based on the total data size and a growth factor alpha."""

    head = 0
    count = 1
    k_schedule = []
    while head <= data_size:
        if head < data_size:
            k_schedule.append(math.ceil(count**loc_round_alpha))
        else:
            k_schedule.append(data_size)
        head += math.ceil(count**loc_round_alpha)
        count += 1
    return k_schedule


##### ByzFL Library Compatibability #####

def flat_updates_avg(updates):
    """"Transforms a list of model parameter tensors into a single, contiguous 1D vector to prepare for robust aggregation."""

    flattened = torch.cat([update.view(-1) if update is not None else torch.tensor([]) for update in updates])
    return flattened

def unflat_updates_avg(flattened_update, template_updates):
    """Reverses the flattening process, taking a 1D aggregated vector and reshaping it back into the original tensor structure of the model."""

    unflattened_updates = []
    idx = 0
    for original_update in template_updates:
        if original_update is not None:
            update_size = original_update.numel()
            unflat_update = flattened_update[idx : idx + update_size].view(original_update.shape)
            unflattened_updates.append(unflat_update)
            idx += update_size
        else:
            unflattened_updates.append(None)
    return unflattened_updates

def flat_updates(client_gradient_lists):
    """A batch version of the flattening function that processes updates from multiple clients into a list of 1D vectors."""

    flattened_updates = []
    for client_gradients in client_gradient_lists:
        flat = torch.cat([grad.view(-1) if grad is not None else torch.tensor([]) for grad in client_gradients])
        flattened_updates.append(flat)
    return flattened_updates


def unflat_updates(flattened_updates, template_gradients):
    """A batch version of the unflattening function that restores the original tensor shapes for a collection of client updates."""

    unflattened_updates = []
    for flat_update in flattened_updates:
        unflat_grads = []
        idx = 0
        for original_grad in template_gradients:
            if original_grad is not None:
                grad_size = original_grad.numel()
                unflat_grad = flat_update[idx : idx + grad_size].view(original_grad.shape)
                unflat_grads.append(unflat_grad)
                idx += grad_size
            else:
                unflat_grads.append(None)
        unflattened_updates.append(unflat_grads)
    return unflattened_updates