from typing import Optional
from pathlib import Path
import logging
import csv

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, SGD, lr_scheduler
from clearml import Task, Logger

from data.batch_generators import MCMuNuSepBatchGenerator
from learning.config import TrainerConfig
from learning.losses import FocalLoss

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_auc_score

# Map optimizer and scheduler names to PyTorch classes
OPTIMIZERS = {
    "adam": Adam,
    "sgd": SGD
}

SCHEDULERS = {
    "step_lr": lr_scheduler.StepLR,
    "cosine_annealing": lr_scheduler.CosineAnnealingLR,
    "reduce_on_plateau": lr_scheduler.ReduceLROnPlateau
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MuNuSepTrainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_batches: MCMuNuSepBatchGenerator, 
                 experiment_folder: Path | str,
                 test_batches: Optional[MCMuNuSepBatchGenerator] = None, 
                 train_config: TrainerConfig = TrainerConfig(),
                 clearml_task: Optional[Task] = None
                 ):
        self.model = model
        self.Nparams = count_parameters(self.model)
        
        self.train_batches = train_batches
        self.test_batches = test_batches
        
        self.trainer_cfg = train_config
        
        # Set up experiment folder for logs
        if isinstance(experiment_folder, str):
            self.experiment_folder = Path(experiment_folder)
        else:
            self.experiment_folder = experiment_folder
        self.checkpoint_path = self.experiment_folder / "checkpoints"
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_logs_path = self.experiment_folder / "metrics_logs"
        self.metrics_logs_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ClearML task if provided
        self.task = clearml_task
        
        # Initialize components
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler() if self.trainer_cfg.scheduler else None
        self.loss_function = self._initialize_loss()
        
        # Stats
        self.total_steps: int = 0
        self.current_epoch: int = 0
        
        # Early stopping variables
        self.best_auc = 0
        self.epochs_after_best = 0
        
        # Dictionary to track if headers are written for each metric type file
        self.logged_headers = {}
        
        logging.info("Initialized MuNuSepTrainer")

    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        optimizer_class = OPTIMIZERS[self.trainer_cfg.optimizer.name.lower()]
        return optimizer_class(self.model.parameters(), **self.trainer_cfg.optimizer.kwargs)

    def _initialize_scheduler(self) -> torch.optim.Optimizer:
        scheduler_class = SCHEDULERS[self.trainer_cfg.scheduler.name.lower()]
        return scheduler_class(self.optimizer, **self.trainer_cfg.scheduler.kwargs)

    def _initialize_loss(self) -> torch.nn.modules.loss._Loss:
        if self.trainer_cfg.loss.name == "FocalLoss":
            return FocalLoss(**self.trainer_cfg.loss.kwargs)
        else:
            return getattr(torch.nn, self.trainer_cfg.loss.name)(**self.trainer_cfg.loss.kwargs)

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
        try:
            y_pred_bin = (y_pred > threshold).astype(float)
            accuracy = accuracy_score(y_true, y_pred_bin)
            precision = precision_score(y_true, y_pred_bin)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            auc = roc_auc_score(y_true, y_pred)
            logging.info("computed metrics")
            return accuracy, precision, tpr, fpr, auc
        except Exception as e:
            logging.info(e)
            return 0, 0, 0, 0, 0

    
    def _log_metrics(self, metrics: dict, epoch: int, step: int, metric_type: str):
        """
        Log multiple metrics to a CSV file for a given metric type.

        Parameters:
        - metrics: dict of metric names and their values, e.g., {"Accuracy": 0.85, "Precision": 0.90, ...}
        - epoch: Current epoch number
        - step: Current training step
        - metric_type: The type/category of metrics (e.g., 'Train Metrics', 'Test Metrics')
        """
        # Define the path for the CSV file for this metric type
        csv_file_path = self.metrics_logs_path / f"{metric_type}.csv"
        
        # Check if headers have been written for this metric type
        if metric_type not in self.logged_headers:
            # Write headers
            with open(csv_file_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                headers = ["epoch", "step"] + list(metrics.keys())
                writer.writerow(headers)
            self.logged_headers[metric_type] = True  # Mark headers as written for this metric type

        # Append the metric values
        with open(csv_file_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            row = [epoch, step] + list(metrics.values())
            writer.writerow(row)
        
        # Log to ClearML if enabled
        if self.task:
            for metric_name, value in metrics.items():
                Logger.current_logger().report_scalar(metric_type, f"{metric_name}", value, step)
        
        logging.info(f"Logged {metric_type} metrics to CSV")
        
        
    def _save_checkpoint(self, epoch: int):
        checkpoint_file = self.checkpoint_path / f"epoch_{epoch+1}.pt"
        torch.save(self.model.state_dict(), checkpoint_file)

        # Upload to ClearML if enabled
        if self.task:
            self.task.upload_artifact(f"checkpoint_epoch_{epoch+1}", checkpoint_file)
        
        logging.info(f"Saved checkpoint for Epoch {epoch+1}: {checkpoint_file}")

    def train(self):
        self.current_epoch = 0
        self.total_steps = 0
        for epoch in range(self.trainer_cfg.num_of_epochs):
            self.current_epoch=epoch
            self.model.train()
            self._train_one_epoch()

            # Save checkpoint model
            if epoch % self.trainer_cfg.checkpoint_interval==0:
                self._save_checkpoint(epoch)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.avg_training_loss)
                else:
                    self.scheduler.step()
                
            # Evaluating on test dataset
            if self.test_batches is not None:
                self._evaluate()
                if self.trainer_cfg.early_stopping_patience <= self.epochs_after_best:
                    message = f"Training stopped early after {epoch+1} epochs due to no improvement in AUC for {self.trainer_cfg.early_stopping_patience} epochs. Best AUC: {self.best_auc}"
                    logging.info(message)  # Log locally

                    # Log to ClearML
                    if self.task:
                        Logger.current_logger().report_text(message)
                        self.task.close()  # Ensure ClearML task is closed
                    break

        logging.info("Training completed.")
        if self.task:
            self.task.close()

    def _train_one_epoch(self):
        logging.info(f"\nStarted Epoch {self.current_epoch+1}/{self.trainer_cfg.num_of_epochs}")
        self.model.train()
        running_loss = 0.0
        y_true_all, y_pred_all = [], []

        # Cycle through batches
        current_step = 0
        for inputs, mask, targets in self.train_batches:
            self.optimizer.zero_grad()

            # Forward pass
            # assuming model accepts shape (bs, max_length, num_features) and mask
            outputs = self.model(inputs, mask) 
            loss = self.loss_function(outputs, targets)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            if targets.shape[-1] == 2:
                y_true_all.extend(targets[:,-1].cpu().numpy())
                y_pred_all.extend(outputs[:,-1].detach().cpu().numpy())
            elif targets.shape[-1] == 3:
                # get metrics on h5 s2 (good) events
                y_true_all.extend(targets[:,-1].cpu().numpy()[targets[:,0].cpu().numpy()==0])
                y_pred_all.extend(outputs[:,-1].detach().cpu().numpy()[targets[:,0].cpu().numpy()==0])
            else:
                raise ValueError(f"Wrong output shape. Expected (B,2) or (B,3), but got {targets.shape=}")

            if current_step % self.trainer_cfg.log_interval == 0:
                self.avg_training_loss = running_loss / ((self.trainer_cfg.log_interval+1) if current_step>0 else 1)
                logging.info(f"#Batch {current_step}")
                running_loss = 0.0
                
                accuracy, precision, tpr, fpr, auc = self._compute_metrics(
                    np.array(y_true_all, dtype=np.float32), np.array(y_pred_all, dtype=np.float32)
                )
                y_true_all, y_pred_all = [], []
                self._log_metrics(
                    metrics={
                        f"{self.trainer_cfg.loss.name}": self.avg_training_loss,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "TPR": tpr,
                        "FPR": fpr,
                        "AUC": auc
                    },
                    epoch=self.current_epoch,
                    step=self.total_steps,
                    metric_type="TrainMetrics"
                )
            current_step += 1
            self.total_steps += 1
        self.train_batches.reset()

    def _evaluate(self):
        
        logging.info("Starting evaluation on test dataset")
        
        self.model.eval()
        y_true_all, y_pred_all = [], []
        sum_test_loss = 0.0
        test_steps = 0
        with torch.no_grad():
            for inputs, mask, targets in self.test_batches:
                # assuming model accepts shape (bs, max_length, num_features) and mask
                outputs = self.model(inputs, mask)
                loss = self.loss_function(outputs, targets)
                sum_test_loss += loss.item()
                if targets.shape[-1] == 2:
                    y_true_all.extend(targets[:,-1].cpu().numpy())
                    y_pred_all.extend(outputs[:,-1].detach().cpu().numpy())
                elif targets.shape[-1] == 3:
                    # get metrics on h5 s2 (good) events
                    y_true_all.extend(targets[:,-1].cpu().numpy()[targets[:,0].cpu().numpy()==0])
                    y_pred_all.extend(outputs[:,-1].detach().cpu().numpy()[targets[:,0].cpu().numpy()==0])
                else:
                    raise ValueError(f"Wrong output shape. Expected (B,2) or (B,3), but got {targets.shape=}")
                test_steps+=1
        if test_steps==0:
            logging.warning(f"Strange behaviour with test_batches: {test_steps=} after evaluating.")
        
        # Compute evaluation metrics
        accuracy, precision, tpr, fpr, auc = self._compute_metrics(
            np.array(y_true_all, dtype=np.float32), np.array(y_pred_all, dtype=np.float32)
        )
        self._log_metrics(
            metrics={
                f"{self.trainer_cfg.loss.name}": sum_test_loss/(test_steps if test_steps>0 else 1),
                "Accuracy": accuracy,
                "Precision": precision,
                "TPR": tpr,
                "FPR": fpr,
                "AUC": auc
            },
            epoch=self.current_epoch,
            step=self.total_steps,
            metric_type="TestMetrics"  # or "Test Metrics" depending on context
        )
        
        # Re Init test data generator
        self.test_batches.reset()
        
        # Early stopper
        if self.best_auc<auc:
            self.best_auc = auc
            self.best_model = self.model
            checkpoint_file = self.checkpoint_path / f"epoch_{self.current_epoch+1}_best_by_test.pt"
            torch.save(self.model.state_dict(), checkpoint_file)
            # Upload to ClearML if enabled
            if self.task:
                self.task.upload_artifact(f"epoch_{self.current_epoch+1}_best_by_test", checkpoint_file)
            
            self.epochs_after_best = 0
            logging.info(f"Saved best model at Epoch {self.current_epoch+1}: {checkpoint_file}")
        else:
            self.epochs_after_best += 1
