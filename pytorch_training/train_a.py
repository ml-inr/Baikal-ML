from data_utils import *
from models import load_model
from training_a import train, train_iters, validate
from metrics import *
import typing as tp
import torch
import numpy as np
import argparse
import yaml
import wandb

DEVICE = "cuda"


def fix_seed():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-dw", "--disable-wandb", action="store_true")
    parser.add_argument("-c", "--config", help="path to config file", required=True)
    return parser.parse_args()


def validate_config(parsed_config: dict[str, tp.Any]):
    for key in [
        "path_to_data",
        "model_type",
        "batch_size",
        "lr",
        "model_params",
        "is_graph",
    ]:
        assert key in parsed_config, f"required key={key} wasn't provided in config"


def main():
    fix_seed()
    args = parse_args()
    with open(args.config, "r") as f:
        train_params = yaml.safe_load(f)

    train_type = train_params.get("train_type")
    is_graph = train_params.get("is_graph")
    is_classification = False
    model = load_model(train_params["model_type"], train_params["model_params"]).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_params["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.2, min_lr=1e-5, patience=1000
    )

    set_tres_stats = False
    if train_type == "noise_sig":
        DatasetType = BaikalDatasetGraph if is_graph else BaikalDataset
        preprocessor = (
            NoiseSigGraphPreprocessor(train_params["knn_neighbours"])
            if is_graph
            else NoiseSigPreprocessor()
        )
        is_classification = True
        criterion = torch.nn.CrossEntropyLoss()
        metrics_calc_fun = binary_clf_metrics
    elif train_type == "track_cascade":
        DatasetType = (
            BaikalDatasetTrackCascadeGraph if is_graph else BaikalDatasetTrackCascade
        )
        preprocessor = (
            TrackCascadeGraphPreprocessor(
                train_params["knn_neighbours"], train_params["tres_cut"]
            )
            if is_graph
            else TrackCascadePreprocessor(train_params["tres_cut"])
        )
        metrics_calc_fun = track_cascade_clf_metrics
        is_classification = True
        criterion = torch.nn.CrossEntropyLoss()
    elif train_type == "tres":
        set_tres_stats = True
        DatasetType = BaikalDatasetTresGraph if is_graph else BaikalDatasetTres
        preprocessor = (
            TresGraphPreprocessor(train_params["knn_neighbours"])
            if is_graph
            else TresPreprocessor()
        )
        metrics_calc_fun = regression_metrics
        criterion = torch.nn.MSELoss()
    elif train_type == "tres_and_track_cascade":
        set_tres_stats = True
        DatasetType = (
            BaikalDatasetTrackCascadeGraph if is_graph else BaikalDatasetTrackCascade
        )
        preprocessor = (
            TresAndTrackCascadeGraphPreprocessor(
                train_params["knn_neighbours"], train_params["tres_cut"]
            )
            if is_graph
            else TresAndTrackCascadePreprocessor(train_params["tres_cut"])
        )
        metrics_calc_fun = tres_and_track_cascade_metrics

        ce_ = torch.nn.CrossEntropyLoss()
        mse_ = torch.nn.MSELoss()

        def criterion(y_pred, y_true):
            ce_loss = ce_(y_pred[:, :-1], y_true[:, 0].long())
            mse_loss = mse_(y_pred[:, -1], y_true[:, -1])
            return ce_loss + train_params["tres_mse_coef"] * mse_loss
    elif train_type == "angle_reconstruction":
        DatasetType = BaikalDatasetAngles
        preprocessor = AnglePreprocessor()
        metrics_calc_fun = angle_reconstruction_metrics
        criterion = torch.nn.MSELoss()
    elif train_type == "angle_and_track_cascade":
        DatasetType = BaikalDatasetAnglesAndTrackCascade
        preprocessor = AngleAndTrackCascadePreprocessor(train_params["tres_cut"])
        ce_ = torch.nn.CrossEntropyLoss()
        mse_ = torch.nn.MSELoss()
        metrics_calc_fun = angle_and_track_cascade_metrics
        def criterion(output, y_true, angles_pred, angles_true):
            ce_loss = ce_(output, y_true.long())
            mse_loss = mse_(angles_pred, angles_true)
            return ce_loss + train_params["mse_coef"] * mse_loss
    else:
        raise ValueError("unknown train_type")

    dataloaders = create_dataloaders(
        path_to_data=train_params["path_to_data"],
        is_graph=train_params["is_graph"],
        batch_size=train_params["batch_size"],
        DatasetType=DatasetType,
        is_classification=is_classification,
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
        preprocessor=preprocessor,
        set_tres_stats=set_tres_stats,
    )

    train_fun_kwargs = dict(
        optimizer=optimizer,
        train_loader=dataloaders["train"],
        criterion=criterion,
        is_graph=is_graph,
        num_iters=train_params.get("num_train_steps_per_validation", 256),
        metrics_calc_fun=metrics_calc_fun,
        is_classification=is_classification,
        is_track_cascade_tres_train=(train_type == "tres_and_track_cascade"),
        is_angle_reconstruction=(train_type == "angle_reconstruction"),
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
        grad_clip_value=train_params.get("grad_clip_value", None),
    )

    validate_fun_kwargs = dict(
        val_loader=dataloaders["val"],
        criterion=criterion,
        is_graph=is_graph,
        scheduler=scheduler,
        metrics_calc_fun=metrics_calc_fun,
        is_classification=is_classification,
        is_track_cascade_tres_train=(train_type == "tres_and_track_cascade"),
        is_angle_reconstruction=(train_type == "angle_reconstruction"),
        is_angle_and_track_cascade=(train_type == "angle_and_track_cascade"),
    )

    if not args.disable_wandb:
        wandb.init(
            project=train_params["exp_project"],
            name=train_params["exp_name"],
            config=train_params,
        )
    print(model)
    train(
        model,
        train_fun=train_iters,
        train_fun_kwargs=train_fun_kwargs,
        validate_fun=validate,
        validate_fun_kwargs=validate_fun_kwargs,
        use_wandb=not args.disable_wandb,
    )


if __name__ == "__main__":
    main()
