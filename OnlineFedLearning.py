# General Imports
from copy import deepcopy
import math
import torch
import gc

from HelperFunc import evaluate, k_schedule

### Federated Learning Environmment were clients communicate entiely trained models to the servers as their updates 

##### Federated Training #####

def online_federated_averaging(server, tot_num_loc_rounds, loc_round_alpha, methode, batchsize, learning_rate, X_test, y_test, 
                               output=True, history=False, attackers=0, aggeg_func='Mean', pre_aggreg=False, pre_agg_method='NNM', 
                               decay_gd=False, decay_factor_gd=0.66, decay_constant_gd=0.1):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    server.global_model.to(device)

    num_attackers = getattr(attackers, 'f', 0)
    if num_attackers > 0 and num_attackers > math.ceil((len(server.clients) + num_attackers)/2):
        print("Warning: Impossible to train, more Attackers then Honest Clients.")
        return
    
    max_datasize = max([client.features.shape[0] for client in server.clients])
    min_datasize = min([client.features.shape[0] for client in server.clients])

    k_sched = k_schedule(max_datasize, loc_round_alpha)

    if tot_num_loc_rounds >= max_datasize:
        print("Warning: Reduce the maximum amount of local steps, as there is no enough data for a complete trainig.")
        return

    k=0
    local_steps = 0
    if history:
        server.loss_history = []
        server.local_steps = []

    while local_steps + k_sched[k] <= tot_num_loc_rounds:
        if output:
            print(f"Global Round {k + 1}, Local Steps: {k_sched[k]}")

        # Server distributes the global model
        server.distribute_model()

        # Clients train locally
        client_models_updates = []
        template_state_dict = server.global_model.state_dict()

        for client in server.clients:
            update = client.online_get_model_update(client.idx, local_steps, local_steps + k_sched[k], methode, batchsize, learning_rate, decay=decay_gd, decay_factor=decay_factor_gd, decay_constant=decay_constant_gd)
            update_cpu = {k: v.cpu() for k, v in update.items()}
            client_models_updates.append(update_cpu)
            gc.collect()
            torch.cuda.empty_cache()

        if attackers != 0:
            attacked_state_dict = attackers.apply_attack_to_model(client_models_updates, template_state_dict)
            client_models_updates.extend(attacked_state_dict)

        # Server aggregates updates
        if pre_aggreg:
            client_models_updates = server.pre_aggregate_methode(client_models_updates, num_attackers, pre_agg_method)
            
        server.aggregate_model_updates(client_models_updates, num_attackers, aggeg_func)

        del client_models_updates
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate the global model on test set
        local_steps += k_sched[k]
        test_accuracy = evaluate(server.global_model, X_test, y_test, device)
        if history:
            server.loss_history.append(test_accuracy)
            server.local_steps.append(local_steps)
        if output:
            print(f"Global model accuracy {k + 1}: {test_accuracy:.4f}")
            print(f"Total local steps so far: {local_steps:.4f}, compared to {max_datasize} samples in the biggest client and {min_datasize} samples in the smallest clinet")
        k += 1
