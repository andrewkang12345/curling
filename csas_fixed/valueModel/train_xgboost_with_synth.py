import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import xgboost as xgb

from dataset import ValueDataset
from split_utils import COMPETITION_KEY, make_train_val_test_indices, write_split_competition_ids, write_test_shot_keys


def _log(msg: str, log_file: Path | None):
    print(msg)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(msg + "\n")


def _dataset_to_numpy(ds, batch_size: int, num_workers: int):
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    X_chunks, y_chunks = [], []

    for x, c, y in loader:
        feats = torch.cat([x, c], dim=1).cpu().numpy().astype(np.float32)
        target = y.squeeze(-1).cpu().numpy().astype(np.float32)
        X_chunks.append(feats)
        y_chunks.append(target)

    X = np.concatenate(X_chunks, axis=0) if X_chunks else np.zeros((0, 0), dtype=np.float32)
    y = np.concatenate(y_chunks, axis=0) if y_chunks else np.zeros((0,), dtype=np.float32)
    return X, y


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float32, copy=False)
    y_pred = y_pred.astype(np.float32, copy=False)
    diff = y_pred - y_true
    return float(np.mean(diff * diff))


def _predict_with_best_iteration(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """
    Predict using best_iteration when early stopping is enabled.
    """
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        return model.predict(X)

    try:
        return model.predict(X, iteration_range=(0, int(best_iter) + 1))
    except TypeError:
        booster = model.get_booster()
        dmat = xgb.DMatrix(X)
        return booster.predict(dmat, iteration_range=(0, int(best_iter) + 1))


def _load_booster_if_exists(path: str | None, log_path: Path | None):
    """
    Resume training by loading a Booster and passing it to fit(xgb_model=...).
    """
    if not path:
        return None
    if not os.path.exists(path):
        _log(f"WARNING: --resume path {path} does not exist. Starting from scratch.", log_path)
        return None

    _log(f"Resuming from booster model: {path}", log_path)
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def train(args):
    log_path = Path(args.log_file) if args.log_file else None
    _log(f"xgboost version: {xgb.__version__}", log_path)

    real_ds = ValueDataset(args.stones_csv, args.ends_csv)
    synth_ds = ValueDataset(args.synth_stones_csv, args.synth_ends_csv)

    _log(f"Real dataset size={len(real_ds)}, Synth dataset size={len(synth_ds)}", log_path)
    _log(
        f"input_dim={real_ds.input_dim}, cond_dim={real_ds.cond_dim}, value_dim=1, num_tasks={real_ds.num_tasks}",
        log_path,
    )

    # Split only real data, holding out whole competitions.
    group_keys = real_ds.df[COMPETITION_KEY].to_numpy()
    real_train_idx, real_val_idx, real_test_idx = make_train_val_test_indices(
        n=len(real_ds),
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.split_seed,
        group_keys=group_keys,
    )
    real_train_ds = Subset(real_ds, real_train_idx)

    if args.val_split > 0.0 and len(real_val_idx) == 0:
        raise ValueError(
            "Validation split is empty under competition-level grouping. "
            "Increase --val_split so at least one competition is held out."
        )

    if real_val_idx is not None and len(real_val_idx) > 0:
        real_val_ds = Subset(real_ds, real_val_idx)
        _log(f"Real split: train={len(real_train_ds)}, val={len(real_val_ds)}, test={len(real_test_idx)}", log_path)
    else:
        real_val_ds = None
        _log(f"Real split: train={len(real_train_ds)}, val=NONE, test={len(real_test_idx)}", log_path)

    if args.val_competitions_out:
        val_comp_path, n_val_comp = write_split_competition_ids(real_ds.df, real_val_idx, args.val_competitions_out)
        _log(
            f"Held-out val competition ids written to {val_comp_path} (rows={n_val_comp}).",
            log_path,
        )

    if args.test_keys_out:
        test_keys_path, n_test_keys = write_test_shot_keys(real_ds.df, real_test_idx, args.test_keys_out)
        _log(
            f"Held-out test shot keys written to {test_keys_path} (rows={n_test_keys}).",
            log_path,
        )

    # Train set is real-train + ALL synth
    train_ds = ConcatDataset([real_train_ds, synth_ds])
    _log(f"Train dataset (real_train + synth) size={len(train_ds)}", log_path)

    # Materialize
    X_train, y_train = _dataset_to_numpy(train_ds, args.materialize_bs, args.num_workers)
    _log(f"Materialized train: X={X_train.shape}, y={y_train.shape}", log_path)

    if real_val_ds is not None:
        X_val, y_val = _dataset_to_numpy(real_val_ds, args.materialize_bs, args.num_workers)
        _log(f"Materialized val (REAL ONLY): X={X_val.shape}, y={y_val.shape}", log_path)
    else:
        X_val, y_val = None, None

    # XGBoost 3.x sklearn API: early stopping must be set on estimator, not passed to fit().
    params = dict(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method=args.tree_method,
        random_state=args.split_seed,
        n_jobs=args.n_jobs,
    )
    if X_val is not None and args.early_stopping_rounds > 0:
        params["early_stopping_rounds"] = args.early_stopping_rounds

    model = xgb.XGBRegressor(**params)

    resume_booster = _load_booster_if_exists(args.resume, log_path)

    if X_val is not None:
        _log(
            f"Training with validation (REAL only). early_stopping_rounds="
            f"{params.get('early_stopping_rounds', 0)} (configured on estimator).",
            log_path,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=args.verbose_eval,
            xgb_model=resume_booster,
        )
        if getattr(model, "best_iteration", None) is not None:
            _log(f"Best iteration: {model.best_iteration}", log_path)
        if getattr(model, "best_score", None) is not None:
            _log(f"Best score (val rmse from xgb): {model.best_score}", log_path)
    else:
        _log("Training without validation (no early stopping).", log_path)
        model.fit(
            X_train,
            y_train,
            verbose=args.verbose_eval,
            xgb_model=resume_booster,
        )

    # ---- Comparable losses (MSE like nn.MSELoss) ----
    yhat_train = _predict_with_best_iteration(model, X_train)
    train_mse = _mse(y_train, yhat_train)
    train_rmse = float(np.sqrt(train_mse))
    _log(f"Final TRAIN mse={train_mse:.6f} | rmse={train_rmse:.6f}", log_path)

    if X_val is not None:
        yhat_val = _predict_with_best_iteration(model, X_val)
        val_mse = _mse(y_val, yhat_val)
        val_rmse = float(np.sqrt(val_mse))
        _log(f"Final VAL   mse={val_mse:.6f} | rmse={val_rmse:.6f}", log_path)

    # ---- Save via Booster to avoid sklearn wrapper `_estimator_type` error ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    booster = model.get_booster()
    booster.save_model(str(out_path))
    _log(f"Saved booster model to: {out_path}", log_path)

    if args.save_config:
        cfg_path = out_path.with_suffix(out_path.suffix + ".config.json")
        cfg_path.write_text(booster.save_config())
        _log(f"Saved booster config to: {cfg_path}", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train XGBoost value model on real_train + synth, validate on held-out competitions only."
    )

    parser.add_argument("--stones_csv", type=str, default="/mnt/data/curling2/testBrax/brax/2026/Stones.csv")
    parser.add_argument("--ends_csv", type=str, default="/mnt/data/curling2/testBrax/brax/2026/Ends.csv")

    parser.add_argument("--synth_stones_csv", type=str, default="/mnt/data/curling2/simple/valueModel/synth_stones.csv")
    parser.add_argument("--synth_ends_csv", type=str, default="/mnt/data/curling2/simple/valueModel/synth_ends.csv")

    parser.add_argument("--out", type=str, default="xgb_value_model_synth.json")
    parser.add_argument("--log_file", type=str, default="train_xgb_with_synth.log")
    parser.add_argument("--resume", type=str, default="", help="Path to an existing booster model to continue training from.")
    parser.add_argument("--save_config", action="store_true", help="Also save booster.save_config() alongside the model.")

    parser.add_argument("--val_split", type=float, default=0.25, help="Fraction of REAL competitions used for validation.")
    parser.add_argument("--test_split", type=float, default=0.0, help="Held-out test fraction from REAL competitions.")
    parser.add_argument("--test_keys_out", type=str, default="", help="CSV path for held-out test shot keys.")
    parser.add_argument("--val_competitions_out", type=str, default="value_val_competitions.csv", help="CSV path for held-out validation competition ids.")
    parser.add_argument("--split_seed", type=int, default=123)

    parser.add_argument("--materialize_bs", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample_bytree", type=float, default=0.9)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)

    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--n_jobs", type=int, default=0)

    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--verbose_eval", type=int, default=50)

    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = None
    if args.resume == "":
        args.resume = None

    train(args)
