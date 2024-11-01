from typing import Optional
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch
from torch.optim import Adam, SGD, lr_scheduler
import wandb

from data.batch_generator import BatchGenerator
from nnetworks.learning.config import TrainerConfig
from nnetworks.learning.losses import FocalLoss


# Map optimizer and scheduler names to actual PyTorch classes
OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

SCHEDULERS = {
    "step_lr": lr_scheduler.StepLR,
    "cosine_annealing": lr_scheduler.CosineAnnealingLR
}

# Trainer class that performs training and evaluation
class Trainer:
    def __init__(self, model, 
                 train_dataset: BatchGenerator, 
                 test_dataset: Optional[BatchGenerator] = None, 
                 train_config: TrainerConfig = TrainerConfig()):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_config = train_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize components
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        self.loss_function = self._initialize_loss()

        # Initialize WandB
        wandb.init(project="MLTrainingProject", config=train_config)

    def _initialize_optimizer(self):
        optimizer_class = OPTIMIZERS[self.train_config.optimizer.name.lower()]
        return optimizer_class(self.model.parameters(), **self.train_config.optimizer.kwargs)

    def _initialize_scheduler(self):
        scheduler_class = SCHEDULERS[self.train_config.scheduler.name.lower()]
        return scheduler_class(self.optimizer, **self.train_config.scheduler.kwargs)

    def _initialize_loss(self):
        if self.train_config.loss.name == "FocalLoss":
            # Define FocalLoss if it's a custom implementation
            return FocalLoss(**self.train_config.loss.kwargs)
        else:
            return getattr(torch.nn, self.train_config.loss.name)(**self.train_config.loss.kwargs)

    def _compute_metrics(self, y_true, y_pred, threshold=0.5):
        y_pred_bin = (y_pred > threshold).float()
        accuracy = accuracy_score(y_true, y_pred_bin)
        precision = precision_score(y_true, y_pred_bin)
        recall = recall_score(y_true, y_pred_bin)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        return accuracy, precision, recall, tpr, fpr

    def _save_checkpoint(self, epoch: int):
        checkpoint_path = Path(f"checkpoints/epoch_{epoch+1}.pt")
        checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def train(self):
        wandb.watch(self.model, log="all")
        for epoch in range(self.train_config.num_of_epochs):
            print(f"Epoch {epoch+1}/{self.train_config.num_of_epochs}")
            self._train_one_epoch(epoch)
            if self.test_dataset:
                self._evaluate()

            # Save checkpoint at the end of each epoch
            self._save_checkpoint(epoch)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

        print("Training completed.")
        wandb.finish()

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        y_true_all, y_pred_all = [], []

        for step in range(self.train_config.steps_per_epoch):
            inputs, targets = next(self.train_dataset)  # Cycle through batches indefinitely
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            y_true_all.extend(targets.cpu().numpy())
            y_pred_all.extend(outputs.detach().cpu().numpy())

            if (step + 1) % self.train_config.log_interval == 0:
                avg_loss = running_loss / self.train_config.log_interval
                print(f"[Step {step+1}] Loss: {avg_loss:.4f}")
                wandb.log({"Train Batch Loss": avg_loss})
                running_loss = 0.0

        # Epoch-level metrics
        accuracy, precision, recall, tpr, fpr = self._compute_metrics(
            torch.tensor(y_true_all), torch.tensor(y_pred_all)
        )
        wandb.log({
            "Train Accuracy": accuracy,
            "Train Precision": precision,
            "Train Recall": recall,
            "Train TPR": tpr,
            "Train FPR": fpr
        })

    def _evaluate(self):
        self.model.eval()
        y_true_all, y_pred_all = [], []

        with torch.no_grad():
            for inputs, targets in self.test_dataset:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                y_true_all.extend(targets.cpu().numpy())
                y_pred_all.extend(outputs.detach().cpu().numpy())

        # Compute evaluation metrics
        accuracy, precision, recall, tpr, fpr = self._compute_metrics(
            torch.tensor(y_true_all), torch.tensor(y_pred_all)
        )
        wandb.log({
            "Test Accuracy": accuracy,
            "Test Precision": precision,
            "Test Recall": recall,
            "Test TPR": tpr,
            "Test FPR": fpr
        })
