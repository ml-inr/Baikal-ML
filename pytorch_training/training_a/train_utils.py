from pathlib import Path
from tqdm import tqdm
import wandb
import torch
import torch_geometric


def _run_model(
    model,
    data,
    is_graph=False,
    is_classification=False,
    is_track_cascade_tres_train=False,
    is_angle_reconstruction=False,
    is_angle_and_track_cascade=False,
):
    if is_graph:
        data = data.to(model.device)
        data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
        y_true = data.y
        output = model(data.x, data.edge_index, data.batch).squeeze()
    else:
        x, y_true, angles_true, mask = data
        # print(x.shape, y_true.shape, angles.shape, mask.shape)
        # a = input()
        x, y_true, angles_true, mask = (
            x.to(model.device),
            y_true.to(model.device),
            angles_true.to(model.device),
            mask.to(model.device),
        )
        output = model(x, mask)

        angles_pred = output[:, 0]
        angles_pred[:, 0] *= torch.pi
        angles_pred[:, 1] *= 2 * torch.pi # tanh(angles_pred[:, 1])
        vec = (
            torch.cos(angles_pred[:, 0]) * torch.cos(angles_pred[:, 1]),
            torch.sin(angles_pred[:, 0]) * torch.cos(angles_pred[:, 1]),
            torch.sin(angles_pred[:, 1])
        )
        
        # 3
        output = output[:, 1:]
        y_true = y_true[mask != 0]
        output = output[mask != 0] 

        y_pred = torch.sigmoid(output[:, 1])
    # print(output.shape, y_pred.shape, y_true.shape, angles_pred.shape, angles.shape)
    # a = input()
    return output, y_pred, y_true, angles_pred, angles_true


def train_iters(
    model,
    optimizer,
    train_loader,
    criterion,
    metrics_calc_fun,
    is_graph=False,
    is_classification=False,
    is_track_cascade_tres_train=False,
    is_angle_reconstruction=False,
    is_angle_and_track_cascade=False,
    num_iters=1,
    grad_clip_value=None,
):
    model.train()
    y_pred_hist = None
    y_true_hist = None
    angles_pred_hist = None
    angles_true_hist = None
    loss_hist = []

    for iter in range(num_iters):
        data = next(train_loader)
        optimizer.zero_grad()
        output, y_pred, y_true, angles_pred, angles_true = _run_model(
            model,
            data,
            is_graph,
            is_classification,
            is_track_cascade_tres_train,
            is_angle_reconstruction,
            is_angle_and_track_cascade
        )
        loss = criterion(output, y_true, angles_pred, angles_true)
        loss.backward()
        if grad_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        loss_hist.append(loss.item())
        y_pred_hist = (
            torch.cat((y_pred_hist, y_pred), dim=0)
            if y_pred_hist is not None
            else y_pred
        )

        y_true_hist = (
            torch.cat((y_true_hist, y_true), dim=0)
            if y_true_hist is not None
            else y_true
        )

        angles_pred_hist = (
            torch.cat((angles_pred_hist, angles_pred), dim=0)
            if angles_pred_hist is not None
            else angles_pred
        )

        angles_true_hist = (
            torch.cat((angles_true_hist, angles_true), dim=0)
            if angles_true_hist is not None
            else angles_true
        )
    train_metrics = metrics_calc_fun(
        y_pred_hist.detach().cpu(), y_true_hist.detach().cpu(),
        angles_pred_hist.detach().cpu(), angles_true_hist.detach().cpu()
    )
    train_metrics["loss"] = sum(loss_hist) / len(loss_hist) if loss_hist else None
    train_metrics["lr"] = optimizer.param_groups[0]["lr"]
    return train_metrics


def validate(
    model,
    val_loader,
    criterion,
    scheduler,
    metrics_calc_fun,
    is_graph=False,
    is_classification=False,
    is_track_cascade_tres_train=False,
    is_angle_reconstruction=False,
    is_angle_and_track_cascade=False,
    return_preds=False,
) -> dict[str, float]:
    y_pred_hist = None
    y_true_hist = None
    angles_pred_hist = None
    angles_true_hist = None
    loss_hist = []

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            output, y_pred, y_true, angles_pred, angles_true = _run_model(
                model,
                data,
                is_graph,
                is_classification,
                is_track_cascade_tres_train,
                is_angle_reconstruction,
                is_angle_and_track_cascade
            )
            loss = criterion(output, y_true, angles_pred, angles_true)
            loss_hist.append(loss.item())

            y_pred_hist = (
                torch.cat((y_pred_hist, y_pred.detach().cpu()), dim=0)
                if y_pred_hist is not None
                else y_pred.detach().cpu()
            )
            y_true_hist = (
                torch.cat((y_true_hist, y_true.detach().cpu()), dim=0)
                if y_true_hist is not None
                else y_true.detach().cpu()
            )

            angles_pred_hist = (
                torch.cat((angles_pred_hist, angles_pred.detach().cpu()), dim=0)
                if angles_pred_hist is not None
                else angles_pred.detach().cpu()
            )
            angles_true_hist = (
                torch.cat((angles_true_hist, angles_true.detach().cpu()), dim=0)
                if angles_true_hist is not None
                else angles_true.detach().cpu()
            )
    if return_preds:
        return y_pred_hist, y_true_hist
    
    val_mertics = metrics_calc_fun(y_pred_hist, y_true_hist, angles_pred_hist, angles_true_hist)
    val_mertics["loss"] = sum(loss_hist) / len(loss_hist) if loss_hist else None
    if val_mertics["loss"]:
        scheduler.step(val_mertics["loss"])
    
    return val_mertics


def train(
    model,
    train_fun,
    validate_fun,
    train_fun_kwargs={},
    validate_fun_kwargs={},
    validate_every=1,
    log_every=1,
    epochs=1000,
    use_wandb=False,
    save_best_model=True,
    valid_main_metric="loss",  # should be lower -> better (mb fix in feautere)
    model_save_dir="models",
    validate_befor_train=True,
):
    best_val_metric = None
    with tqdm(range(epochs), unit="batch", dynamic_ncols=True) as epoch_iter:
        for epoch in epoch_iter:
            if not validate_befor_train:
                train_logs = train_fun(model, **train_fun_kwargs)
                train_logs_ = {"train/" + k: v for k, v in train_logs.items()}
                epoch_iter.set_description(str(train_logs))
                if use_wandb and epoch % log_every == 0:
                    wandb.log(train_logs_)

            if epochs % validate_every == 0 or validate_befor_train:
                validate_befor_train = False
                val_logs = validate_fun(model, **validate_fun_kwargs)
                val_logs_ = {"val/" + k: v for k, v in val_logs.items()}
                if use_wandb:
                    wandb.log(val_logs_)

                if save_best_model and (
                    best_val_metric is None
                    or val_logs[valid_main_metric] < best_val_metric
                ):
                    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
                    best_val_metric = val_logs[valid_main_metric]
                    save_path = model_save_dir + "/model.pth"
                    torch.save(model.state_dict(), save_path)
