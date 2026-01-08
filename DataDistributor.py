import random
import numpy as np
import torch
from torch.utils.data import DataLoader

##### DATA DISTRIBUTION #####

class DataDistributor:
    def __init__(self, dataset, num_clients, batch_size, num_classes, distribution='EqStd', dist_param=0.5):
        self.dataset = dataset
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.distribution = distribution
        self.distribution_parameter = dist_param
        self.num_classes = num_classes

    def iid_idx(self, idx):
        random.shuffle(idx)
        return np.array_split(idx, self.num_clients)

    def extreme_niid_idx(self, idx):
        if len(idx) == 0:
            return list([[]] * self.num_clients)
        sorted_idx = np.array(sorted(zip(self.dataset.targets[idx], idx)))[:, 1]
        return np.array_split(sorted_idx, self.num_clients)

    def distribute_data(self):
        data_size = len(self.dataset)
        indices = list(range(data_size))
        np.random.shuffle(indices)

        if self.distribution == 'EqStd':
            split_size = data_size // self.num_clients
            client_indices = [indices[i * split_size: (i + 1) * split_size] for i in range(self.num_clients)]

        elif self.distribution == 'GammaNiid':
            nb_similarity = int(len(indices) * self.distribution_parameter)
            iid = self.iid_idx(indices[:nb_similarity])
            niid = self.extreme_niid_idx(indices[nb_similarity:])
            split_idx = [np.concatenate((iid[i], niid[i])) for i in range(self.num_clients)]
            client_indices = [node_idx.astype(int) for node_idx in split_idx]

        elif self.distribution == 'Dirich':
            sample = np.random.dirichlet(np.repeat(self.distribution_parameter, self.num_clients), size=self.num_classes)
            class_indices_dict = {}
            indices_tensor = torch.tensor(indices)

            if isinstance(self.dataset.targets, list):
                targets_tensor = torch.tensor(self.dataset.targets)
            else:
                targets_tensor = self.dataset.targets

            for k in range(self.num_classes):
                class_k_indices = (targets_tensor[indices_tensor] == k)
                class_k = indices_tensor[class_k_indices]

                if class_k.numel() > 0:
                    class_indices_dict[k] = class_k
                else:
                    class_indices_dict[k] = torch.tensor([], dtype=torch.long)

            client_indices = [[] for _ in range(self.num_clients)]

            for k in range(self.num_classes):
                class_k_indices = class_indices_dict[k]
                num_class_k_samples = len(class_k_indices)

                if num_class_k_samples > 0:
                    client_class_k_counts = torch.tensor((sample[k] * num_class_k_samples)).long()

                    diff = num_class_k_samples - torch.sum(client_class_k_counts).item()
                    if diff != 0:
                        client_class_k_counts[:abs(diff)] += torch.sign(torch.tensor(diff)).long()

                    split_points = torch.cumsum(client_class_k_counts, dim=0)[:-1].tolist()
                    split_indices = torch.split(class_k_indices, client_class_k_counts.tolist())

                    for client_idx in range(self.num_clients):
                        client_indices[client_idx].extend(split_indices[client_idx].tolist())

            client_indices = [list(map(int, client_idx_list)) for client_idx_list in client_indices]


        client_dataloaders = []
        for idx in client_indices:
            subset = torch.utils.data.Subset(self.dataset, idx)
            dataloader = DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else 0)
            client_dataloaders.append(dataloader)

        return client_dataloaders