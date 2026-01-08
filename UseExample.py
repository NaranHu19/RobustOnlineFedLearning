# General Imports
import numpy as np
import sys
import pickle
import os
from copy import deepcopy
import math
import torch
from torchvision import transforms, datasets
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

# ByzFL Imports
import byzfl
from byzfl import ByzantineClient
from byzfl.utils.misc import set_random_seed

# Environemnt 
from HelperFunc import flat_updates_avg, unflat_updates_avg
from HelperFunc import evaluate, k_schedule
from Clients import Client, ByzantineClient
from Server import Server
from OnlineFedLearning_RandomAttacks import online_federated_averaging_randAttack
from Optimizer import CustomOptimizer
from DataDistributor import DataDistributor

# Data generation parameters
num_clients = 7
num_attackers = 3
batch_size = 32
num_classes = 10

# Training parameters
decay_factor = 0.1   # Decay factor for local training
decay_coeff = 0.1
learning_rate = 0.1   # Not applied given that decay_gd = True (Non constant learning rate) 
tot_num_loc_rounds = 8450


class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        """Defines a standard Convolutional Neural Network architecture with two convolutional layers and two fully connected layers tailored for MNIST digit classification."""
        self._c1 = nn.Conv2d(1, 20, 5, 1)
        self._c2 = nn.Conv2d(20, 50, 5, 1)
        self._f1 = nn.Linear(800, 500)
        self._f2 = nn.Linear(500, 10)

    def forward(self, x):
        """Implements the forward pass of the CNN, utilizing ReLU activations, max-pooling, and log-softmax for the output layer."""
        x = F.relu(self._c1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._c2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._f1(x.view(-1, 800)))
        x = F.log_softmax(self._f2(x), dim=1)
        return x

# Data
SEED = 42
set_random_seed(SEED)

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root="XXXX", train=True, download=True, transform=transform)

test_dataset = datasets.MNIST(root="XXXX", train=False, download=True, transform=transform)
test_dataset.targets = Tensor(test_dataset.targets).long() 

X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])  # Shape: [N, 1, 28, 28]
y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])  # Shape: [N]


def run_experiment(k_alpha, attack, defense, alpha):
    """
    The core experimental unit. It initializes the DataDistributor, sets up the Server and Clients, and executes the online_federated_averaging_randAttack loop. To ensure statistical significance, 
    it repeats the training 5 times for each configuration and returns the mean and standard deviation of the accuracy.
    """

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_distributor = DataDistributor(train_dataset, num_clients, batch_size, 
                                   num_classes, distribution='Dirich', dist_param=alpha)
    client_dataloaders = data_distributor.distribute_data()

    byz_worker = ByzantineClient({
        "name": attack,
        "f": num_attackers,
        "parameters": {"tau": 1.5},
    })

    results = []

    for _ in range(5):
        model = CNN_MNIST()
        server = Server(model=model)

        for i, dataloader in enumerate(client_dataloaders):
            client_model = CNN_MNIST()
            client_optimizer = CustomOptimizer(client_model, lambd=0.1, device=device)
            client = Client(dataloader=dataloader, model=client_model, optimizer=client_optimizer, idx=i, device=device)
            server.add_client(client)

        online_federated_averaging_randAttack(
                    server, tot_num_loc_rounds, k_alpha, 'MBGD', 32, learning_rate, X_test, y_test,
                    output=False, history=True, attackers=byz_worker,
                    aggeg_func=defense, pre_aggreg=False, pre_agg_method='NNM', 
                    decay_gd=True, decay_factor_gd=decay_factor, decay_constant_gd=decay_coeff)
        results.append(server.loss_history)

    results_array = np.array(results)
    final_means = np.mean(results_array, axis=0)
    final_stds = np.std(results_array, axis=0)
    return (k_alpha, attack, defense, alpha, final_means, final_stds, server.local_steps)

# Parameters
checkpoint_file = "XXXXX.pkl"
checkpoint_file2 = "XXXXX.pkl"
alphas = [1+1e-3, 1.5, 2, 2.5] # k-scheduling parameters
attacks = ['SignFlipping', 'InnerProductManipulation', 'ALittleIsEnough']
agg_func = ['Mean','TriMean','GeoMed','MultiKrum']
alpha_hetero = [0.2, 1, 100] # Heterogeneity levels

# Load or initialize results_dict
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        results_dict = pickle.load(f)
    print(f"Loaded checkpoint from {checkpoint_file}")
else:
    results_dict = {}
    for local_alpha in alphas:
        results_dict[local_alpha] = {}
        for attack in attacks:
            results_dict[local_alpha][attack] = {}
            for agg in agg_func:
                results_dict[local_alpha][attack][agg] = {}
                for al in alpha_hetero:
                    results_dict[local_alpha][attack][agg][al] = {}

if os.path.exists(checkpoint_file2):
    with open(checkpoint_file2, 'rb') as f:
        k_schedules = pickle.load(f)
    print(f"Loaded checkpoint from {checkpoint_file2}")
else:
    k_schedules = []

# Parallel execution
if __name__ == "__main__":
    for local_alpha in alphas:
        for attack in attacks:
            for defense in agg_func:
                for al in alpha_hetero:
                    # Skip if already computed
                    if results_dict[local_alpha][attack][defense][al]:
                        print(f"Skipping {local_alpha}, {attack}, {defense}, {al} (already done)")
                        continue
                    local_alpha, attack, defense, al, means, stds, k_sched = run_experiment(
                        local_alpha, attack, defense, al
                    )
                    results_dict[local_alpha][attack][defense][al] = {'mean': means, 'std': stds}
                    k_schedules.append(k_sched)
                    with open(checkpoint_file2, "wb") as f:
                        pickle.dump(k_schedules, f)
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(results_dict, f)
                    print(f"Saved checkpoint for {local_alpha}, {attack}, {defense}, {al}")