import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_auc_score, roc_curve, auc
import plotly.graph_objects as go
import plotly.colors as pc

from learning.losses import FocalLoss
from data.batch_generator import BatchGenerator


# Predictions making

def load_loss(loss_name: str = "FocalLoss", kwargs: dict = dict(alpha=1, gamma=2)) -> torch.nn.modules.loss._Loss:
    if loss_name == "FocalLoss":
        return FocalLoss(**kwargs)
    else:
        return getattr(torch.nn, loss_name)(**kwargs)

def predict(model: torch.nn.Module, val_gen: BatchGenerator, device: torch.DeviceObjType, loss_function: torch.functional.F) -> tuple[float, pd.DataFrame]:
        val_gen.reinit()
        y_true, y_pred, num_signal_hits, num_signal_strings = [], [], [], []
        PrimePhi, PrimeTheta, PrimeEn, BundleEnReg = [], [], [], []
        ev_id, cluster_id = [], []
        sum_val_loss = 0.0
        val_steps = 0 
        val_dataset = val_gen.get_batches(device=device, yield_small_chunk=True)
        with torch.no_grad():
            for inputs, mask, targets, small_df in val_dataset:
                # assuming model accepts shape (bs, max_length, num_features) and mask
                outputs = model(inputs, mask)
                sum_val_loss += loss_function(outputs, targets).item()
                y_true.extend(targets.cpu().tolist()), y_pred.extend(outputs.detach().cpu().tolist())
                num_signal_hits.extend(small_df['num_signal_hits'].to_numpy()), num_signal_strings.extend(small_df['num_signal_strings'].to_numpy())
                PrimePhi.extend(small_df['PrimePhi'].to_numpy()), PrimeTheta.extend(small_df['PrimeTheta'].to_numpy())
                PrimeEn.extend(small_df['PrimeEn'].to_numpy()), BundleEnReg.extend(small_df['BundleEnReg'].to_numpy())
                ev_id.extend(small_df['ev_id'].to_numpy()), cluster_id.extend(small_df['cluster_id'].to_numpy())
                val_steps+=1
                
        result_df = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred,
                "num_signal_hits": num_signal_hits,
                "num_signal_strings": num_signal_strings,
                "PrimePhi": PrimePhi,
                "PrimeTheta": PrimeTheta,
                "PrimeEn": PrimeEn,
                "BundleEnReg": BundleEnReg,
                "ev_id": ev_id,
                "cluster_id": cluster_id
            })
        return sum_val_loss/val_steps, result_df


# Plotting fot binary classification

def plot_roc_auc(fpr, tpr, fig, name=f'ROC curve'):
    roc_auc = auc(fpr, tpr)
    
    # Add trace for ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name=f'{name}(AUC = {roc_auc:.2f})',
        line=dict(width=2)
    ))
    
    # Display the plot
    return fig

def generate_plotly_colors(num_colors):
    # Use Plotly's built-in color scales to get distinct colors
    color_scale = pc.qualitative.Plotly  # This is a predefined qualitative color scale
    colors = color_scale * (num_colors // len(color_scale) + 1)  # Repeat the color scale if more colors are needed
    return colors[:num_colors]

def plot_metrics_vs_thresholds(fpr, tpr, thresholds, fig, name=f'', color='green'):
    
    # Plot TPR vs Threshold
    fig.add_trace(go.Scatter(
        x=thresholds, y=tpr, mode='lines', name=f"TPR / Recall / Sensitivity,<br>{name}",
        line=dict(color=color), showlegend=True
    ))
    
    # Plot FPR vs Threshold
    fig.add_trace(go.Scatter(
        x=thresholds, y=1-fpr, mode='lines', name=f"(1-FPR / Precision / Purity),<br>{name}",
        line=dict(dash='dot', color=color), showlegend=True
    ))
    return fig

def plot_metrics_vs_energy(energies, fpr, tpr, fig, name=f'', color='green'):
    
    # Plot TPR vs Threshold
    fig.add_trace(go.Scatter(
        x=energies, y=tpr, mode='lines', name=f"TPR / Recall / Sensitivity,<br>{name}",
        line=dict(color=color), showlegend=True
    ))
    
    # Plot FPR vs Threshold
    fig.add_trace(go.Scatter(
        x=energies, y=1-fpr, mode='lines', name=f"(1-FPR / Precision / Purity),<br>{name}",
        line=dict(dash='dot', color=color), showlegend=True
    ))
    return fig