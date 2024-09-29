from pathlib import Path
from tqdm import tqdm
import wandb
import torch
import torch_geometric


def train_iters(
    model,
    optimizer,
    scheduler,
    train_loader,
    criterion,
    metrics_calc_fun,
    is_graph=False,
    is_classification=False,
    is_track_cascade_tres_train=False,
    num_iters=1,
):
    model.train()
    y_pred_hist = None
    y_true_hist = None
    loss_hist = []

    for iter in range(num_iters):
        data = next(train_loader)

        optimizer.zero_grad()
        if is_graph:
            data = data.to(model.device)
            data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
            y_true = data.y
            output = model(data.x, data.edge_index, data.batch)
        else:
            x, y_true, mask = data
            x, y_true, mask = (
                x.to(model.device),
                y_true.to(model.device),
                mask.to(model.device),
            )
            output = model(x, mask)
            if is_track_cascade_tres_train:
                y_true = y_true.reshape(-1, 2)
            else:
                y_true = y_true.reshape(-1)
            mask = mask.reshape(-1)
            y_true = y_true[mask != 0]
            output = output.reshape(-1, output.shape[-1]).squeeze()
            output = output[mask != 0]

        if is_classification:
            y_pred = output.argmax(dim=1)
        elif is_track_cascade_tres_train:
            y_pred = output[:, :-1].argmax(dim=1)
        else:
            y_pred = output

        loss = criterion(output, y_true)

        loss.backward()
        # grads = [
        #     param.grad.detach().flatten()
        #     for param in model.parameters()
        #     if param.grad is not None
        # ]
        # norm = torch.cat(grads).norm()
        optimizer.step()
        scheduler.step(loss)

        loss_hist.append(loss.item())
        y_pred_hist = (
            torch.cat((y_pred_hist, y_pred), dim=0)
            if y_pred_hist is not None
            else y_pred
        )
        if is_track_cascade_tres_train:
            y_true = y_true[:, 0]
        y_true_hist = (
            torch.cat((y_true_hist, y_true), dim=0)
            if y_true_hist is not None
            else y_true
        )

    train_metrics = metrics_calc_fun(
        y_pred_hist.detach().cpu(), y_true_hist.detach().cpu()
    )
    train_metrics["loss"] = sum(loss_hist) / len(loss_hist) if loss_hist else None
    train_metrics["lr"] = optimizer.param_groups[0]["lr"]
    return train_metrics


def validate(
    model,
    val_loader,
    criterion,
    metrics_calc_fun,
    is_graph=False,
    is_classification=False,
    is_track_cascade_tres_train=False,
) -> dict[str, float]:
    y_pred_hist = None
    y_true_hist = None
    loss_hist = []

    model.eval()
    with torch.no_grad():
        for data in val_loader:
            if is_graph:
                data = data.to(model.device)
                data.edge_index = torch_geometric.utils.sort_edge_index(data.edge_index)
                y_true = data.y
                output = model(data.x, data.edge_index, data.batch)
            else:
                x, y_true, mask = data
                x, y_true, mask = (
                    x.to(model.device),
                    y_true.to(model.device),
                    mask.to(model.device),
                )
                output = model(x, mask)
                if is_track_cascade_tres_train:
                    y_true = y_true.reshape(-1, 2)
                else:
                    y_true = y_true.reshape(-1)
                mask = mask.reshape(-1)
                y_true = y_true[mask != 0]
                output = output.reshape(-1, output.shape[-1]).squeeze()
                output = output[mask != 0]

            if is_classification:
                y_pred = output.argmax(dim=1)
            elif is_track_cascade_tres_train:
                y_pred = output[:, :2].argmax(dim=1)

            loss = criterion(output, y_true)

            loss_hist.append(loss.item())
            if is_track_cascade_tres_train:
                y_true = y_true[:, 0]
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

    val_mertics = metrics_calc_fun(y_pred_hist, y_true_hist)
    val_mertics["loss"] = sum(loss_hist) / len(loss_hist) if loss_hist else None

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
):
    best_val_metric = None
    with tqdm(range(epochs), unit="batch", dynamic_ncols=True) as epoch_iter:
        for epoch in epoch_iter:
            train_logs = train_fun(model, **train_fun_kwargs)
            train_logs_ = {"train/" + k: v for k, v in train_logs.items()}
            epoch_iter.set_description(str(train_logs))
            if use_wandb and epoch % log_every == 0:
                wandb.log(train_logs_)

            if epochs % validate_every == 0:
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
