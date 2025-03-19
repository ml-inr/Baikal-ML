import numpy as np
import pandas as pd
import polars as pl
import torch

from learning.losses import FocalLoss
from data.batch_generators import MCMuNuSepBatchGenerator, ExpBatchGenerator


#=======================Predictions making=========================#

def load_loss(loss_name: str = "FocalLoss", kwargs: dict = dict(alpha=1, gamma=2)) -> torch.nn.modules.loss._Loss:
    if loss_name == "FocalLoss":
        return FocalLoss(**kwargs)
    else:
        return getattr(torch.nn, loss_name)(**kwargs)

def predict_MC(model: torch.nn.Module, val_dataset: MCMuNuSepBatchGenerator, loss_function: torch.functional.F) -> tuple[float, pd.DataFrame]:
        Qs = []
        y_true, y_pred, num_signal_hits, num_signal_strings = [], [], [], []
        PrimePhi, PrimeTheta, PrimeEn, BundleEnReg = [], [], [], []
        ev_id, cluster_id = [], []
        
        sum_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for inputs, mask, targets, batch_df in val_dataset:
                Qs.extend(batch_df["PulsesAmpl"].to_list())
                # assuming model accepts shape (bs, max_length, num_features) and mask
                outputs = model(inputs, mask)
                sum_val_loss += loss_function(outputs, targets).item()
                y_true.extend(targets.cpu().tolist()), y_pred.extend(outputs.cpu().tolist())
                num_signal_hits.extend(batch_df['num_signal_hits'].to_numpy()), num_signal_strings.extend(batch_df['num_signal_strings'].to_numpy())
                PrimePhi.extend(batch_df['PrimePhi'].to_numpy()), PrimeTheta.extend(batch_df['PrimeTheta'].to_numpy())
                PrimeEn.extend(batch_df['PrimeEn'].to_numpy()), BundleEnReg.extend(batch_df['BundleEnReg'].to_numpy())
                ev_id.extend(batch_df['ev_id'].to_numpy()), cluster_id.extend(batch_df['cluster_id'].to_numpy())
                val_steps+=1
                
        result_df = pd.DataFrame({
                "y_true": np.array(y_true)[:,1],
                "y_pred": np.array(y_pred)[:,1],
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
    

def predict(model: torch.nn.Module, val_dataset: ExpBatchGenerator, with_data: bool = False, num_steps=float('inf')) -> pl.DataFrame:
        dfs = []
        val_steps = 0
        with torch.no_grad():
            for inputs, mask, batch_df in val_dataset:
                # assuming model accepts shape (bs, max_length, num_features) and mask
                outputs = model(inputs, mask)[:,1]
                if not with_data:
                    batch_df = batch_df[[c for c in batch_df.columns if c not in ["PulsesAmpl", "PulsesTime", "Xrel", "Yrel", "Zrel", "X", "Y", "Z"]]]
                batch_df = batch_df.with_columns(y_pred=pl.Series(outputs.cpu().tolist()))
                dfs.append(batch_df)
                val_steps+=1
                if val_steps>num_steps:
                    break
        result_df = pl.concat(dfs)
        return result_df