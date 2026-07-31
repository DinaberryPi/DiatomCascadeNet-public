"""
Training and Validation Utilities

Functions specific to the training and validation phase.
"""


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize (e.g., F1), 'min' for metrics to minimize (e.g., loss)
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, score, model):
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric score
            model: Model to save weights from if this is the best score
        
        Returns:
            bool: True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Best score: {self.best_score:.4f}")
        
        return self.early_stop
    
    def _is_better(self, current, best):
        """Check if current score is better than best score"""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
    
    def load_best_weights(self, model):
        """Load the best weights into the model"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            return True
        return False

