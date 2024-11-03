from typing import Optional
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, SGD, lr_scheduler
from clearml import Task, Logger

from data.batch_generator import BatchGenerator
from nnetworks.learning.config import TrainerConfig
from nnetworks.learning.losses import FocalLoss

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score


# Map optimizer and scheduler names to PyTorch classes
OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

SCHEDULERS = {
    "step_lr": lr_scheduler.StepLR,
    "cosine_annealing": lr_scheduler.CosineAnnealingLR
}

class MuNuSepTrainer:
    def __init__(self, model: torch.nn.Module, 
                 train_gen: BatchGenerator, 
                 test_gen: Optional[BatchGenerator] = None, 
                 train_config: TrainerConfig = TrainerConfig(),
                 clearml_task: Optional[Task] = None):
        self.model = model
        self.train_gen = train_gen
        self.train_dataset = self.train_gen.get_batches()
        
        self.test_gen = test_gen
        if self.test_gen is not None:
            self.test_dataset = self.test_gen.get_batches()
        
        self.train_config = train_config
        
        self.steps_per_epoch = self.train_config.steps_per_epoch
        if self.steps_per_epoch is None:
            self.steps_per_epoch = np.inf
        
        self.checkpoint_path = Path(train_config.checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize ClearML task if provided
        self.use_clearml = clearml_task is not None
        if self.use_clearml:
            self.task = clearml_task
            self.task.connect(train_config)  # Log training hyperparameters
        
        # Initialize components
        self.optimizer = self._initialize_optimizer()
        if self.train_config.scheduler:
            self.scheduler = self._initialize_scheduler()
        else:
            self.scheduler = None
        self.loss_function = self._initialize_loss()
        
        self.total_steps: Optional[int] = None
        self.current_epoch: Optional[int] = None

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        optimizer_class = OPTIMIZERS[self.train_config.optimizer.name.lower()]
        return optimizer_class(self.model.parameters(), **self.train_config.optimizer.kwargs)

    def _initialize_scheduler(self) -> torch.optim.Optimizer:
        scheduler_class = SCHEDULERS[self.train_config.scheduler.name.lower()]
        return scheduler_class(self.optimizer, **self.train_config.scheduler.kwargs)

    def _initialize_loss(self) -> torch.nn.modules.loss._Loss:
        if self.train_config.loss.name == "FocalLoss":
            return FocalLoss(**self.train_config.loss.kwargs)
        else:
            return getattr(torch.nn, self.train_config.loss.name)(**self.train_config.loss.kwargs)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
        y_pred_bin = (y_pred > threshold).astype(float)
        accuracy = accuracy_score(y_true, y_pred_bin)
        precision = precision_score(y_true, y_pred_bin)
        recall = recall_score(y_true, y_pred_bin)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        auc = roc_auc_score(y_true, y_pred)
        return accuracy, precision, recall, tpr, fpr, auc

    def _log_metric(self, metric_name: str, value: float,  epoch: int, step: int, metric_type: str):
        log_message = f"{metric_type} {metric_name}: {value} (Epoch {epoch}, Total steps {step})"
        print(log_message)

        # Log locally to a file
        with open(self.checkpoint_path / f"{metric_type}_metrics.txt", "a") as log_file:
            log_file.write(log_message + "\n")
        
        # Log to ClearML if enabled
        if self.use_clearml:
            Logger.current_logger().report_scalar(metric_type, metric_name, value, epoch)
            Logger.current_logger().report_scalar(metric_type, metric_name+"ByStep", value, step)

    def _save_checkpoint(self, epoch: int):
        checkpoint_file = self.checkpoint_path / f"epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved: {checkpoint_file}")

        # Upload to ClearML if enabled
        if self.use_clearml:
            self.task.upload_artifact(f"checkpoint_epoch_{epoch+1}", checkpoint_file)

    def train(self):
        self.current_epoch = 0
        self.total_steps = 0
        for epoch in range(self.train_config.num_of_epochs):
            self.model.train()
            print(f"Epoch {epoch+1}/{self.train_config.num_of_epochs}")
            self._train_one_epoch(epoch)
            if self.test_gen is not None:
                self._evaluate(epoch)

            # Save checkpoint model
            if epoch % self.train_config.checkpoint_interval==0:
                self._save_checkpoint(epoch)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            self.current_epoch += 1

        print("Training completed.")
        if self.use_clearml:
            self.task.close()

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        sum_loss = 0.0
        y_true_all, y_pred_all = [], []

        current_step = 0
        while current_step<self.steps_per_epoch:
            # Cycle through batches
            try:
                inputs, mask, targets = next(self.train_dataset)  
            except StopIteration:
                self.train_gen.reinit()
                self.train_dataset = self.train_gen.get_batches()
                # if steps_per_epoch not congigured, epoch ends
                if self.steps_per_epoch == np.inf:
                    break
                inputs, mask, targets = next(self.train_dataset)
                
            inputs, mask, targets = inputs.to(self.device), mask.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            # assuming model accepts shape (bs, max_length, num_features) and mask
            outputs = self.model(inputs, mask) 
            loss = self.loss_function(outputs, targets)
            sum_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            y_true_all.extend(targets[:,1].cpu().numpy())
            y_pred_all.extend(outputs[:,1].detach().cpu().numpy())

            if current_step % self.train_config.log_interval == 0:
                avg_loss = sum_loss / ((self.train_config.log_interval+1) if current_step>0 else 1)
                self._log_metric(self.train_config.loss.name, avg_loss, self.current_epoch, self.total_steps, "Train Loss")
                sum_loss = 0.0
            current_step += 1
            self.total_steps += 1

        # Epoch-level metrics
        accuracy, precision, recall, tpr, fpr, auc = self._compute_metrics(
            np.array(y_true_all, dtype=np.float32), np.array(y_pred_all, dtype=np.float32)
        )
        self._log_metric("Accuracy", accuracy, self.current_epoch, self.total_steps, "Train Metrics")
        self._log_metric("Precision", precision, self.current_epoch, self.total_steps, "Train Metrics")
        self._log_metric("Recall", recall, self.current_epoch, self.total_steps, "Train Metrics")
        self._log_metric("TPR", tpr, self.current_epoch, self.total_steps, "Train Metrics")
        self._log_metric("FPR", fpr, self.current_epoch, self.total_steps, "Train Metrics")
        self._log_metric("AUC", auc, self.current_epoch, self.total_steps, "Train Metrics")

    def _evaluate(self, epoch: int):
        self.model.eval()
        y_true_all, y_pred_all = [], []

        with torch.no_grad():
            for inputs, mask, targets in self.test_dataset:
                inputs, mask, targets = inputs.to(self.device), mask.to(self.device), targets.to(self.device)
                # assuming model accepts shape (bs, max_length, num_features) and mask
                outputs = self.model(inputs, mask)
                y_true_all.extend(targets[:,1].cpu().numpy())
                y_pred_all.extend(outputs[:,1].detach().cpu().numpy())
        
        self.test_gen.reinit()
        self.test_dataset = self.test_gen.get_batches()

        # Compute evaluation metrics
        accuracy, precision, recall, tpr, fpr, auc = self._compute_metrics(
            np.array(y_true_all, dtype=np.float32), np.array(y_pred_all, dtype=np.float32)
        )
        self._log_metric("Accuracy", accuracy, self.current_epoch, self.total_steps, "Test Metrics")
        self._log_metric("Precision", precision, self.current_epoch, self.total_steps, "Test Metrics")
        self._log_metric("Recall", recall, self.current_epoch, self.total_steps, "Test Metrics")
        self._log_metric("TPR", tpr, self.current_epoch, self.total_steps, "Test Metrics")
        self._log_metric("FPR", fpr, self.current_epoch, self.total_steps, "Test Metrics")
        self._log_metric("AUC", auc, self.current_epoch, self.total_steps, "Test Metrics")
