# RobustOnlineFedLearning

This repository contains a comprehensive framework for Robust Federated Learning (FL), designed to evaluate and mitigate the impact of Byzantine (malicious) actors in a decentralized training environment. The system implements various robust aggregation rules and dynamic scheduling to handle both IID and Non-IID data distributions.

## File Description 

## Environments Setting 

### HelperFunc.py

This module provides essential utility functions that support the internal operations of the Federated Learning system. It contains logic for model evaluation, a dynamic scheduling mechanism for local training steps, and critical transformation functions that ensure compatibility between PyTorch model states and the Byzantine-robust aggregation library (byzfl).

### Server.py

This file defines the Server class, which acts as the central orchestrator in the Federated Learning pipeline. It is responsible for maintaining the global model's state, broadcasting parameters to clients, and, most importantly, performing Byzantine-robust aggregation. By leveraging the byzfl library, the server can mitigate the impact of malicious "attackers".

### Clinets.py

This file defines the behavior of the participants in the Federated Learning network. It includes a standard Client class for honest local training and a ByzantineClient class that leverages the byzfl library to simulate adversarial behavior. The clients handle data management, local optimization (GD, SGD, or Mini-batch GD), and the application of poisoning attacks to model updates

### Optimizer.py

This file implements CustomOptimizer, a flexible optimization class designed to handle various gradient-based update strategies within the Federated Learning framework. It supports standard Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent (MBGD), along with their "online" variants for sequential data processing. The optimizer includes built-in L2 regularization and customizable learning rate decay to ensure stable training and convergence.

### DataDistributor.py

This file implements the DataDistributor class, which is responsible for partitioning a centralized dataset among multiple clients. In Federated Learning, data distribution is a critical factor; this class supports both IID (Independent and Identically Distributed) and Non-IID scenarios. By simulating realistic data heterogeneity, such as label skew, it allows for the evaluation of Robust Federated Learning algorithms under challenging, decentralized conditions.

## Training -- Robust Federative Learning 

While both files manage the training orchestration for the Federated Learning environment, they differ fundamentally in how they simulate and handle adversarial presence. OnlineFedLearning.py is designed for a consistent adversarial model where a fixed group of malicious clients is appended to the honest pool in every round. It is particularly optimized for stable execution and hardware efficiency, utilizing manual garbage collection and CUDA cache clearing to manage memory during the local training phase. In contrast, OnlineFedLearning_RandomAttacks.py introduces a dynamic and unpredictable threat model. Instead of appending new malicious updates, it randomly selects a subset of existing clients to "compromise" during each round, replacing their honest contributions with malignant ones. This script is intended for evaluating how robust aggregation rules perform when the identity of the attackers shifts unpredictably between global communication steps.

### OnlineFedLearning_RandomAttacks.py

This file implements the top-level execution logic for Online Federated Learning with a focus on adversarial resilience. Unlike standard Federated Learning, which often operates on fixed epochs, this script manages training rounds based on a dynamic data-access schedule. It orchestrates the communication between the server and clients, simulates randomized Byzantine attacks by injecting malicious updates into the aggregation pool, and monitors global model performance over time.

### OnlineFedLearning.py

This file provides the primary training orchestration for the Federated Learning system. It implements a standard Online Federated Averaging loop where clients communicate entire trained models back to the server as their updates. While similar to the "Random Attacks" variant, this script focuses on a steady-state training environment, including critical memory management (via garbage collection and CUDA cache clearing) to handle multiple client models on limited hardware.
