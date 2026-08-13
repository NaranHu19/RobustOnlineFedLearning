import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):  # type: ignore
    """Define a fully connected neural network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
    ) -> None:
        """
        Initialize the fully connected neural network.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dim : int
            Number of neurons in the hidden layer.
        n_classes : int
            Number of output classes.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Output logits produced by the network.
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_MNIST(nn.Module):  # type: ignore
    """
    Define a convolutional neural network for MNIST classification.

    The network consists of two convolutional layers followed by two
    fully connected layers.
    """

    def __init__(self) -> None:
        """
        Initialize the MNIST convolutional neural network.

        Create convolutional and fully connected layers for image
        classification.
        """
        super().__init__()
        self._c1 = nn.Conv2d(1, 20, 5, 1)
        self._c2 = nn.Conv2d(20, 50, 5, 1)
        self._f1 = nn.Linear(800, 500)
        self._f2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input MNIST image tensor.

        Returns
        -------
        torch.Tensor
            Log-probabilities for each MNIST class.
        """
        x = F.relu(self._c1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._c2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self._f1(x.view(-1, 800)))
        x = F.log_softmax(self._f2(x), dim=1)
        return x
