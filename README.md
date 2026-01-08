# RobustOnlineFedLearning

Complete Implementation of a Federated Online Learning Environment 

** File Description **

HelperFunc.py

This module provides essential utility functions that support the internal operations of the Federated Learning system. It contains logic for model evaluation, a dynamic scheduling mechanism for local training steps, and critical transformation functions that ensure compatibility between PyTorch model states and the Byzantine-robust aggregation library (byzfl).

Server.py

This file defines the Server class, which acts as the central orchestrator in the Federated Learning pipeline. It is responsible for maintaining the global model's state, broadcasting parameters to clients, and, most importantly, performing Byzantine-robust aggregation. By leveraging the byzfl library, the server can mitigate the impact of malicious "attackers".

Clinets.py

This file defines the behavior of the participants in the Federated Learning network. It includes a standard Client class for honest local training and a ByzantineClient class that leverages the byzfl library to simulate adversarial behavior. The clients handle data management, local optimization (GD, SGD, or Mini-batch GD), and the application of poisoning attacks to model updates

Optimizer.py

This file implements CustomOptimizer, a flexible optimization class designed to handle various gradient-based update strategies within the Federated Learning framework. It supports standard Gradient Descent (GD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent (MBGD), along with their "online" variants for sequential data processing. The optimizer includes built-in L2 regularization and customizable learning rate decay to ensure stable training and convergence.

DataDistributor.py

This file implements the DataDistributor class, which is responsible for partitioning a centralized dataset among multiple clients. In Federated Learning, data distribution is a critical factor; this class supports both IID (Independent and Identically Distributed) and Non-IID scenarios. By simulating realistic data heterogeneity, such as label skew, it allows for the evaluation of Robust Federated Learning algorithms under challenging, decentralized conditions.
