# General Imports
from copy import deepcopy
import torch
import byzfl

from HelperFunc import flat_updates_avg, unflat_updates_avg

##### Server #####

class Server:
    def __init__(self, model):
        self.global_model = deepcopy(model)
        self.clients = []
        self.loss_history = []
        self.local_steps = []

    def add_client(self, client):
        self.clients.append(client)

    def distribute_model(self):
        for client in self.clients:
            client.set_global_model(deepcopy(self.global_model))
    
    def pre_aggregate_methode(self, client_models_updates, num_attackers=0, pre_agg_method='NNM'):
        if not client_models_updates:
            print("No client model updates received for pre-aggregation.")
            return client_models_updates 

        bucket_size = 1  # Number of vectors per bucket
        n_clients = len(client_models_updates)

        if pre_agg_method == 'NNM':
            agg = byzfl.NNM(num_attackers)
        else:
            raise ValueError(f"Pre-aggregation method '{pre_agg_method}' not supported.")

        keys = list(client_models_updates[0].keys())

        # Create list of empty dicts for the pre-aggregated updates
        pre_aggregated_updates = [dict() for _ in range(n_clients)]

        for key in keys:
            # Flatten all client tensors for this key
            flattened = [c[key].flatten() for c in client_models_updates]
            stacked = torch.stack(flattened, dim=0)

            agg_result = agg(stacked) 

            # Ensure proper shape
            if agg_result.ndim == 1:
                agg_result = agg_result.unsqueeze(0)

            n_buckets = agg_result.shape[0]
            for client_idx in range(n_clients):
                bucket_idx = client_idx % n_buckets
                pre_aggregated_updates[client_idx][key] = (agg_result[bucket_idx].view(client_models_updates[0][key].shape).detach().clone().to(client_models_updates[0][key].device).type(client_models_updates[0][key].dtype))

        # Sanity check
        for i in range(n_clients):
            for key in keys:
                assert pre_aggregated_updates[i][key].shape == client_models_updates[0][key].shape, \
                    f"Shape mismatch for key {key}, client {i}"

        return pre_aggregated_updates

    def aggregate_model_updates(self, client_models_updates, num_attackers=0, aggeg_func='Mean'):
        if not client_models_updates:
            print("No client model updates received for aggregation.")
            return

        template_state_dict = client_models_updates[0] # Use the first client as template

        # Flatten all client model parameters
        flattened_client_params = []
        for state_dict in client_models_updates:
            flattened_params = flat_updates_avg([param for param in state_dict.values()])
            flattened_client_params.append(flattened_params)

        aggregated_flattened_params = None
        if aggeg_func == 'Mean':
            agg = byzfl.Average()
            aggregated_flattened_params = agg(torch.stack(flattened_client_params))

        elif aggeg_func == 'TriMean':
            agg = byzfl.TrMean(num_attackers)
            aggregated_flattened_params = agg(torch.stack(flattened_client_params))

        elif aggeg_func == 'GeoMed':
            agg = byzfl.GeometricMedian(nu=0.0, T=100)
            aggregated_flattened_params = agg(torch.stack(flattened_client_params))

        elif aggeg_func == 'MultiKrum':
            agg = byzfl.MultiKrum(num_attackers)
            aggregated_flattened_params = agg(torch.stack(flattened_client_params))

        else:
            raise ValueError(f"Aggregation function '{aggeg_func}' not supported for model aggregation.")

        # Unflatten the aggregated parameters
        aggregated_parameters = unflat_updates_avg(aggregated_flattened_params, [param for param in template_state_dict.values()])

        # Update the global model
        aggregated_state_dict = {}
        for i, key in enumerate(template_state_dict.keys()):
            aggregated_state_dict[key] = aggregated_parameters[i]

        self.global_model.load_state_dict(aggregated_state_dict)