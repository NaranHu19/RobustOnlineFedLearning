# General Imports
from copy import deepcopy
import torch
import byzfl

from HelperFunc import flat_updates_avg, unflat_updates_avg

##### Client #####

class Client:
    def __init__(self, dataloader, model, optimizer, idx, device):
        self.idx = idx
        self.device = device
        self.dataloader = dataloader
        self.local_model = deepcopy(model).to(self.device)
        self.optimizer = optimizer
        self.optimizer.model = self.local_model
        self.criterion = self.optimizer.loss
        self.global_model = None
        self.features, self.targets = self._fetch_all_data()
        self.features = self.features.to(self.device)
        self.targets = self.targets.to(self.device)

    def _fetch_all_data(self):
      all_features = []
      all_targets = []
      for inputs, targets in self.dataloader:
        all_features.append(inputs.to(self.device))
        all_targets.append(targets.to(self.device))
      return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

    def set_global_model(self, global_model):
            self.global_model = global_model
            self.local_model.load_state_dict(global_model.state_dict())
            self.optimizer.model = self.local_model

    def get_model_update(self, methode, batchsize, num_local_rounds, learning_rate, decay=False, decay_factor=1.0, decay_constant=1):
        if self.global_model is None:
            print("Debug (Client.get_model_update): Global model not set for the client.")
            raise ValueError("Global model not set for the client.")

        initial_global_model_state_dict = deepcopy(self.global_model.state_dict())

        self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.to(self.device)

        X = self.features
        y = self.targets

        if X.numel() == 0 or y.numel() == 0:
            print("Debug (Client.get_model_update): Warning: Client has empty data!")
            # Return a zero update vector if data is empty
            zero_update_vector = {key: torch.zeros_like(param) for key, param in initial_global_model_state_dict.items()}
            print("Debug (Client.get_model_update): Returning zero update vector due to empty data.")
            return zero_update_vector

        for _ in range(num_local_rounds):
            if methode == 'GD':
                self.optimizer.gradient_descent(X, y, lr=learning_rate, max_iters=1, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)
            elif methode == 'SGD':
                self.optimizer.stochastic_gd(X, y, lr=learning_rate, max_iters=1, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)
            elif methode == 'MBGD':
                self.optimizer.mini_batch_gd(X, y, lr=learning_rate, batch_size=batchsize, max_iters=1, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)
            else:
                raise ValueError("Invalid gradient method specified.")
            
        return self.local_model.state_dict()

    def get_model_update_decay(self, idx, methode, batch_size, start, end, learning_rate, decay=False, decay_factor=1.0, decay_constant=1):
        if self.global_model is None:
            print("Debug (Client.get_model_update): Global model not set for the client.")
            raise ValueError("Global model not set for the client.")

        initial_global_model_state_dict = deepcopy(self.global_model.state_dict())

        self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.to(self.device)

        X = self.features
        y = self.targets

        if X.numel() == 0 or y.numel() == 0:
            print("Debug (Client.get_model_update): Warning: Client has empty data!")
            # Return a zero update vector if data is empty
            zero_update_vector = {key: torch.zeros_like(param) for key, param in initial_global_model_state_dict.items()}
            print("Debug (Client.get_model_update): Returning zero update vector due to empty data.")
            return zero_update_vector

        if methode == 'SGD':
            self.optimizer.online_stochastic_gd(idx, X, y, start, end, lr=learning_rate, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)
        elif methode == 'MBGD':
            self.optimizer.online_mini_batch_gd(idx, X, y, start, end, batchsize=batch_size, lr=learning_rate, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)
        else:
            raise ValueError("Invalid gradient method specified.")
            
        return self.local_model.state_dict()
    
    def online_get_model_update(self, idx, k_sched1, k_sched2, methode, batchsize, learning_rate, decay, decay_factor, decay_constant):
        if self.global_model is None:
            print("Debug (Client.get_model_update): Global model not set for the client.")
            raise ValueError("Global model not set for the client.")

        self.local_model.load_state_dict(self.global_model.state_dict())

        X = self.features
        y = self.targets

        if methode == 'SGD':
            self.optimizer.online_stochastic_gd(idx, X, y, k_sched1, k_sched2, lr=learning_rate, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)            
        elif methode == 'MBGD':
            self.optimizer.online_mini_batch_gd(idx, X, y, k_sched1, k_sched2, batchsize, lr=learning_rate, client=False, decay=decay, decay_factor=decay_factor, decay_constant=decay_constant)

        return self.local_model.state_dict()


class ByzantineClient(byzfl.ByzantineClient):
    def __init__(self, attack_params):
        super().__init__(attack_params)

    def apply_attack_to_model(self, list_of_model_state_dicts, template_model_state_dict):
        list_of_flattened_params = []
        for model_state_dict in list_of_model_state_dicts:
            model_parameters = [param for param in model_state_dict.values()]
            flattened_params = flat_updates_avg(model_parameters)
            list_of_flattened_params.append(flattened_params)

        attacked_flattened_params_list = self.apply_attack(list_of_flattened_params)

        attacked_state_dicts = []
        template_parameters = [param for param in template_model_state_dict.values()]

        for attacked_flattened_params in attacked_flattened_params_list:
            attacked_parameters = unflat_updates_avg(attacked_flattened_params, template_parameters)
            attacked_state_dict = {}
            for i, key in enumerate(template_model_state_dict.keys()):
                attacked_state_dict[key] = attacked_parameters[i]
            attacked_state_dicts.append(attacked_state_dict)

        return attacked_state_dicts    