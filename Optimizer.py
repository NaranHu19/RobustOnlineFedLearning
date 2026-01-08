# PyTorch Imports
import torch
import torch.nn.functional as F


##### Optimizer Class #####
    
class CustomOptimizer:
    def __init__(self, model, lambd=0.1, device=None):
        """Initializes the optimizer with a model, regularization strength (λ), and hardware device"""

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.lambd = lambd
        self.loss_history = []

    def loss(self, X, y):
        """Calculates the total loss, combining Cross-Entropy loss for classification and L2 regularization for weight penalization"""

        X = X.to(self.device)
        y = y.to(self.device)
        logits = self.model(X)
        ce_loss = F.cross_entropy(logits, y) 

        l2_reg = torch.tensor(0., dtype=torch.float32, device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param) ** 2

        return ce_loss + (self.lambd / 2) * l2_reg

    def compute_gradients(self, X, y):
        """Executes a backward pass to compute and extract gradients for all model parameters"""

        X = X.to(self.device)
        y = y.to(self.device)
        self.model.zero_grad()
        loss = self.loss(X, y)
        loss.backward()

        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
            else:
                gradients.append(torch.zeros_like(param))
        return gradients

    def apply_gradients(self, gradients, lr):
            """Updates model parameters using the calculated gradients and a specific learning rate."""

            with torch.no_grad():
                for i, param in enumerate(self.model.parameters()):
                    if gradients[i] is not None:
                        param.data.add_(-lr * gradients[i])
                    else:
                        print(f"Debug (Optimizer.apply_gradients for Client {id(self.model)}): Gradient for parameter {i} is None.")

### GRADIENT DESCENT - add scheduled/decaying learning rate

    def gradient_descent(self, X, y, lr=0.1, max_iters=500, client=False, decay=False, decay_factor=1.0, decay_constant=1):
        """Performs full-batch Gradient Descent over a set number of iterations or returns averaged gradients for client-side updates."""

        X = X.to(self.device)
        y = y.to(self.device)

        if client:
            gradients = self.compute_gradients(X, y)
            n_samples = X.shape[0]
            averaged_gradients = [grad / n_samples for grad in gradients]
            return averaged_gradients

        else:

            self.loss_history = []

            for i in range(max_iters):

                if decay:
                    current_lr = (decay_constant/(i+1)**decay_factor)

                else:
                    current_lr = lr

                gradients = self.compute_gradients(X, y)
                self.apply_gradients(gradients, current_lr)

                current_loss = self.loss(X, y).item()
                self.loss_history.append(current_loss)

### STOCHASTIC GRADEINT DESCENT 

    def stochastic_gd(self, X, y, lr=0.01, max_iters=500, client=False, decay=False, decay_factor=1.0, decay_constant=1):
        """Implements Stochastic Gradient Descent by selecting a single random sample per iteration to update the model."""

        X = X.to(self.device)
        y = y.to(self.device)
        
        if client:
            n_samples = X.shape[0]
            idx = torch.randint(0, n_samples, (1,))
            X_batch = X[idx]
            y_batch = y[idx]

            gradients = self.compute_gradients(X_batch, y_batch)
            return gradients

        else:
            self.loss_history = []
            n_samples = X.shape[0]

            for i in range(max_iters):
                # Randomly select one sample
                idx = torch.randint(0, n_samples, (1,))  # Change to avoid repeating data 
                X_batch = X[idx]
                y_batch = y[idx]

                if decay:
                    current_lr = (decay_constant/(i+1)**decay_factor)
                
                else:
                    current_lr = lr

                gradients = self.compute_gradients(X_batch, y_batch)
                self.apply_gradients(gradients, current_lr)

                current_loss = self.loss(X, y).item()
                self.loss_history.append(current_loss)


    def online_stochastic_gd(self, idx, X, y, k_sched1, k_sched2, lr=0.01, client=False, decay=False, decay_factor=0.66, decay_constant=1):
        """A specialized SGD variant that processes a specific slice of data (from k_sched1 to k_sched2) sequentially."""

        X = X.to(self.device)
        y = y.to(self.device)
        
        if k_sched1 >= X.shape[0]:
            print(f"Client {idx} has used all its data, and does no longer contribute to the update of the global model")
            return
        if k_sched2 > X.shape[0]:
            k_sched2 = X.shape[0]
        

        X_batch = X[k_sched1:k_sched2]
        y_batch = y[k_sched1:k_sched2]

        diff = k_sched2 - k_sched1

        if diff == 0:
            print(f"Client {idx}: Empty batch (k_sched1={k_sched1}, k_sched2={k_sched2}), skipping update.")
            return
        
        if client:
            for i in range(diff):
                gradients = self.compute_gradients(X_batch[i].unsqueeze(0), y_batch[i].unsqueeze(0))
            return gradients
        
        else:
            self.loss_history = []

            for i in range(diff):
                if decay:
                    current_lr = (decay_constant/(k_sched1 + i+1)**decay_factor)
                
                else:
                    current_lr = lr

                gradients = self.compute_gradients(X_batch[i].unsqueeze(0), y_batch[i].unsqueeze(0))                
                self.apply_gradients(gradients, current_lr)

                current_loss = self.loss(X, y).item()
                self.loss_history.append(current_loss)

### MINI-BATCH GRADIENT DESCENT 

    def mini_batch_gd(self, X, y, lr=0.01, batch_size=32, max_iters=500, client=False, decay=False, decay_factor=1.0, decay_constant=1):
        """Executes optimization using subsets (batches) of data, iterating through the entire dataset multiple times."""

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

        else:
            self.loss_history = []

            for i in range(max_iters):
                permutation = torch.randperm(n_samples)

                current_lr = lr

                if decay:
                    current_lr = (decay_constant/(i+1)**decay_factor)

                for start in range(0, n_samples, batch_size):
                    indices = permutation[start:start+batch_size]
                    X_batch = X[indices]
                    y_batch = y[indices]

                    gradients = self.compute_gradients(X_batch, y_batch)
                    self.apply_gradients(gradients, current_lr)

                current_loss = self.loss(X, y).item()
                self.loss_history.append(current_loss)
    
    def online_mini_batch_gd(self, idx, X, y, k_sched1, k_sched2, batchsize, lr=0.01, client=False, decay=False, decay_factor=0.66, decay_constant=1):
        """A mini-batch variant that operates on a targeted data window, commonly used in dynamic scheduling scenarios."""
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        if k_sched1 >= X.shape[0]:
            print(f"Client {idx} has used all its data, and does no longer contribute to the update of the global model")
            return
        if k_sched2 > X.shape[0]:
            k_sched2 = X.shape[0]
        

        X_batch = X[k_sched1:k_sched2]
        y_batch = y[k_sched1:k_sched2]

        diff = k_sched2 - k_sched1
        full_batch = diff// batchsize
        remainder = diff % batchsize

        if client:
            for i in range(full_batch):
                gradients = self.compute_gradients(X_batch[i*batchsize:(i+1)*batchsize], y_batch[i*batchsize:(i+1)*batchsize])
            if remainder > 0:
                gradients = self.compute_gradients(X_batch[full_batch*batchsize:], y_batch[full_batch*batchsize:])
            return gradients
        
        else:
            self.loss_history = []

            for i in range(full_batch):
                if decay:
                    current_lr = (decay_constant/(k_sched1 + i+1)**decay_factor)
                
                else:
                    current_lr = lr

                gradients = self.compute_gradients(X_batch[i*batchsize:(i+1)*batchsize], y_batch[i*batchsize:(i+1)*batchsize])
                self.apply_gradients(gradients, current_lr)

                current_loss = self.loss(X, y).item()
                self.loss_history.append(current_loss)

            if remainder > 0:
                gradients = self.compute_gradients(X_batch[full_batch*batchsize:], y_batch[full_batch*batchsize:])
                
                if decay:
                    current_lr = (decay_constant/(k_sched1 + full_batch+1)**decay_factor)
                
                else:
                    current_lr = lr
                
                self.apply_gradients(gradients, current_lr)

                current_loss = self.loss(X, y).item()
                self.loss_history.append(current_loss)