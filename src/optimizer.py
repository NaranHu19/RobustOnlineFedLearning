import torch
import torch.nn.functional as F


class CustomOptimizer:
    """Provide optimization methods for a PyTorch model."""

    def __init__(
        self,
        model: torch.nn.Module,
        lambd: float = 0.1,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the optimizer with a model, regularization, and device.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model whose parameters will be optimized.
        lambd : float, default=0.1
            L2 regularization coefficient.
        device : str, default="cpu"
            Device on which model computations are performed.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.lambd = lambd
        self.loss_history: list[float] = []

    def loss(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the total loss for the model.

        The loss combines cross-entropy classification loss with L2
        regularization of the model parameters.

        Parameters
        ----------
        X : torch.Tensor
            Input features used to calculate the loss.
        y : torch.Tensor
            Target labels corresponding to the input features.

        Returns
        -------
        torch.Tensor
            Total loss consisting of cross-entropy loss and L2
            regularization.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        logits = self.model(X)
        ce_loss = F.cross_entropy(logits, y)

        l2_reg = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param) ** 2

        return ce_loss + (self.lambd / 2) * l2_reg

    def compute_gradients(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Compute gradients for all model parameters.

        Perform a backward pass and return a copy of each parameter gradient.

        Parameters
        ----------
        X : torch.Tensor
            Input features used to compute the gradients.
        y : torch.Tensor
            Target labels corresponding to the input features.

        Returns
        -------
        list[torch.Tensor]
            Gradients corresponding to each model parameter.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        self.model.zero_grad()
        loss = self.loss(X, y)
        loss.backward()

        gradients: list[torch.Tensor] = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(torch.zeros_like(param))

        return gradients

    def apply_gradients(
        self,
        gradients: list[torch.Tensor],
        lr: float,
    ) -> None:
        """
        Update model parameters using calculated gradients.

        Parameters
        ----------
        gradients : list[torch.Tensor]
            Gradients corresponding to the model parameters.
        lr : float
            Learning rate used to scale the parameter updates.
        """
        with torch.no_grad():
            for i, param in enumerate(self.model.parameters()):
                if gradients[i] is not None:
                    param.data.add_(-lr * gradients[i])
                else:
                    print(
                        f"Debug Optimizer.apply_gradients for Client "
                        f"{id(self.model)}:"
                        f"Gradient for parameter {i} is None."
                    )

    # GRADIENT DESCENT - add scheduled/decaying learning rate

    def gradient_descent(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.1,
        max_iters: int = 500,
        client: bool = False,
        decay: bool = False,
        decay_factor: float = 1.0,
        decay_constant: float = 1,
    ) -> list[torch.Tensor] | None:
        """
        Perform full-batch gradient descent for a set number of iterations.

        Parameters
        ----------
        X : torch.Tensor
            Input features used for training.
        y : torch.Tensor
            Target labels corresponding to the input features.
        lr : float, default=0.1
            Learning rate used for parameter updates when decay is disabled.
        max_iters : int, default=500
            Maximum number of gradient descent iterations.
        client : bool, default=False
            Whether to return gradients for client-side processing instead
            of directly updating the model.
        decay : bool, default=False
            Whether to use a decaying learning rate.
        decay_factor : float, default=1.0
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1
            Constant used to scale the decaying learning rate.

        Returns
        -------
        list[torch.Tensor] or None
            Averaged gradients when ``client`` is True; otherwise, returns
            ``None`` after updating the model.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if client:
            gradients = self.compute_gradients(X, y)
            n_samples = X.shape[0]
            averaged_gradients = [grad / n_samples for grad in gradients]
            return averaged_gradients

        self.loss_history = []

        for i in range(max_iters):
            if decay:
                current_lr = decay_constant / (i + 1) ** decay_factor
            else:
                current_lr = lr

            gradients = self.compute_gradients(X, y)
            self.apply_gradients(gradients, current_lr)

            current_loss = self.loss(X, y).item()
            self.loss_history.append(current_loss)

        return None

    # STOCHASTIC GRADIENT DESCENT

    def stochastic_gd(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.1,
        max_iters: int = 500,
        client: bool = False,
        decay: bool = False,
        decay_factor: float = 1.0,
        decay_constant: float = 1,
    ) -> list[torch.Tensor] | None:
        """
        Perform stochastic gradient descent using random samples.

        Parameters
        ----------
        X : torch.Tensor
            Input features used for training.
        y : torch.Tensor
            Target labels corresponding to the input features.
        lr : float, default=0.1
            Learning rate used for parameter updates when decay is disabled.
        max_iters : int, default=500
            Maximum number of training iterations.
        client : bool, default=False
            Whether to return gradients instead of updating the model.
        decay : bool, default=False
            Whether to use a decaying learning rate.
        decay_factor : float, default=1.0
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1
            Constant used to scale the decaying learning rate.

        Returns
        -------
        list[torch.Tensor] or None
            Gradients when ``client`` is True; otherwise, returns ``None``
            after updating the model.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if client:
            n_samples = X.shape[0]
            idx = torch.randint(0, n_samples, (1,))
            X_batch = X[idx]
            y_batch = y[idx]

            gradients = self.compute_gradients(X_batch, y_batch)
            return gradients

        self.loss_history = []
        n_samples = X.shape[0]

        for i in range(max_iters):
            # Randomly select one sample.
            idx = torch.randint(0, n_samples, (1,))
            X_batch = X[idx]
            y_batch = y[idx]

            if decay:
                current_lr = decay_constant / (i + 1) ** decay_factor
            else:
                current_lr = lr

            gradients = self.compute_gradients(X_batch, y_batch)
            self.apply_gradients(gradients, current_lr)

            current_loss = self.loss(X, y).item()
            self.loss_history.append(current_loss)

        return None

    def online_stochastic_gd(
        self,
        idx: int,
        X: torch.Tensor,
        y: torch.Tensor,
        k_sched1: int,
        k_sched2: int,
        lr: float = 0.01,
        client: bool = False,
        decay: bool = False,
        decay_factor: float = 0.66,
        decay_constant: float = 1,
    ) -> list[torch.Tensor] | None:
        """
        Perform stochastic gradient descent over a scheduled data slice.

        Process the selected range of data sequentially.

        Parameters
        ----------
        idx : int
            Identifier of the client performing the update.
        X : torch.Tensor
            Input features used for training.
        y : torch.Tensor
            Target labels corresponding to the input features.
        k_sched1 : int
            Starting index of the scheduled data range.
        k_sched2 : int
            Ending index of the scheduled data range.
        lr : float, default=0.01
            Learning rate used when decay is disabled.
        client : bool, default=False
            Whether to return gradients instead of updating the model.
        decay : bool, default=False
            Whether to use a decaying learning rate.
        decay_factor : float, default=0.66
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1
            Constant used to scale the decaying learning rate.

        Returns
        -------
        list[torch.Tensor] or None
            Gradients when ``client`` is True; otherwise, returns ``None``.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if k_sched1 >= X.shape[0]:
            print(
                f"Client {idx} has used all its data, "
                "and does no longer contribute to the update of the global model"
            )
            return None

        if k_sched2 > X.shape[0]:
            k_sched2 = X.shape[0]

        X_batch = X[k_sched1:k_sched2]
        y_batch = y[k_sched1:k_sched2]

        diff = k_sched2 - k_sched1

        if diff == 0:
            print(
                f"Client {idx}: Empty batch "
                f"(k_sched1={k_sched1}, k_sched2={k_sched2}), "
                "skipping update."
            )
            return None

        if client:
            gradients: list[torch.Tensor] = []

            for i in range(diff):
                gradients = self.compute_gradients(
                    X_batch[i].unsqueeze(0),
                    y_batch[i].unsqueeze(0),
                )

            return gradients

        self.loss_history = []

        for i in range(diff):
            if decay:
                current_lr = decay_constant / (k_sched1 + i + 1) ** decay_factor
            else:
                current_lr = lr

            gradients = self.compute_gradients(
                X_batch[i].unsqueeze(0),
                y_batch[i].unsqueeze(0),
            )
            self.apply_gradients(gradients, current_lr)

            current_loss = self.loss(X, y).item()
            self.loss_history.append(current_loss)

        return None

    # MINI-BATCH GRADIENT DESCENT

    def mini_batch_gd(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.1,
        batch_size: int = 32,
        max_iters: int = 500,
        client: bool = False,
        decay: bool = False,
        decay_factor: float = 1.0,
        decay_constant: float = 1,
    ) -> list[torch.Tensor] | None:
        """
        Perform mini-batch gradient descent over the dataset.

        Iterate through batches of data for the specified number of iterations.

        Parameters
        ----------
        X : torch.Tensor
            Input features used for training.
        y : torch.Tensor
            Target labels corresponding to the input features.
        lr : float, default=0.1
            Learning rate used when decay is disabled.
        batch_size : int, default=32
            Number of samples processed in each mini-batch.
        max_iters : int, default=500
            Number of training iterations.
        client : bool, default=False
            Whether to return gradients instead of updating the model.
        decay : bool, default=False
            Whether to use a decaying learning rate.
        decay_factor : float, default=1.0
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1
            Constant used to scale the decaying learning rate.

        Returns
        -------
        list[torch.Tensor] or None
            Gradients when ``client`` is True; otherwise, returns ``None``.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        n_samples = X.shape[0]
        batch_size = min(batch_size, n_samples)

        if client:
            indices = torch.randperm(n_samples)[:batch_size]
            X_batch = X[indices]
            y_batch = y[indices]

            gradients = self.compute_gradients(X_batch, y_batch)
            return gradients

        self.loss_history = []

        for i in range(max_iters):
            permutation = torch.randperm(n_samples)

            current_lr = lr

            if decay:
                current_lr = decay_constant / (i + 1) ** decay_factor

            for start in range(0, n_samples, batch_size):
                indices = permutation[start : start + batch_size]
                X_batch = X[indices]
                y_batch = y[indices]

                gradients = self.compute_gradients(X_batch, y_batch)
                self.apply_gradients(gradients, current_lr)

            current_loss = self.loss(X, y).item()
            self.loss_history.append(current_loss)

        return None

    def online_mini_batch_gd(
        self,
        idx: int,
        X: torch.Tensor,
        y: torch.Tensor,
        k_sched1: int,
        k_sched2: int,
        batch_size: int,
        lr: float = 0.01,
        client: bool = False,
        decay: bool = False,
        decay_factor: float = 0.66,
        decay_constant: float = 1,
    ) -> list[torch.Tensor] | None:
        """
        Perform mini-batch gradient descent over a scheduled data window.

        Process the selected data range using the specified batch size.

        Parameters
        ----------
        idx : int
            Identifier of the client performing the update.
        X : torch.Tensor
            Input features used for training.
        y : torch.Tensor
            Target labels corresponding to the input features.
        k_sched1 : int
            Starting index of the scheduled data range.
        k_sched2 : int
            Ending index of the scheduled data range.
        batch_size : int
            Number of samples processed in each mini-batch.
        lr : float, default=0.01
            Learning rate used when decay is disabled.
        client : bool, default=False
            Whether to return gradients instead of updating the model.
        decay : bool, default=False
            Whether to use a decaying learning rate.
        decay_factor : float, default=0.66
            Exponent controlling the learning-rate decay.
        decay_constant : float, default=1
            Constant used to scale the decaying learning rate.

        Returns
        -------
        list[torch.Tensor] or None
            Gradients when ``client`` is True; otherwise, returns ``None``.
        """
        X = X.to(self.device)
        y = y.to(self.device)

        if k_sched1 >= X.shape[0]:
            print(
                f"Client {idx} has used all its data, "
                "and does no longer contribute to the update of the global model"
            )
            return None

        if k_sched2 > X.shape[0]:
            k_sched2 = X.shape[0]

        X_batch = X[k_sched1:k_sched2]
        y_batch = y[k_sched1:k_sched2]

        diff = k_sched2 - k_sched1
        full_batch = diff // batch_size
        remainder = diff % batch_size

        if client:
            gradients: list[torch.Tensor] = []

            for i in range(full_batch):
                gradients = self.compute_gradients(
                    X_batch[i * batch_size : (i + 1) * batch_size],
                    y_batch[i * batch_size : (i + 1) * batch_size],
                )

            if remainder > 0:
                gradients = self.compute_gradients(
                    X_batch[full_batch * batch_size :],
                    y_batch[full_batch * batch_size :],
                )

            return gradients

        self.loss_history = []

        for i in range(full_batch):
            if decay:
                current_lr = decay_constant / (k_sched1 + i + 1) ** decay_factor
            else:
                current_lr = lr

            gradients = self.compute_gradients(
                X_batch[i * batch_size : (i + 1) * batch_size],
                y_batch[i * batch_size : (i + 1) * batch_size],
            )
            self.apply_gradients(gradients, current_lr)

            current_loss = self.loss(X, y).item()
            self.loss_history.append(current_loss)

        if remainder > 0:
            gradients = self.compute_gradients(
                X_batch[full_batch * batch_size :],
                y_batch[full_batch * batch_size :],
            )

            if decay:
                current_lr = (
                    decay_constant / (k_sched1 + full_batch + 1) ** decay_factor
                )
            else:
                current_lr = lr

            self.apply_gradients(gradients, current_lr)

            current_loss = self.loss(X, y).item()
            self.loss_history.append(current_loss)

        return None
