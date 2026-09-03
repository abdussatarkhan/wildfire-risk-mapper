"""
Machine Learning and Spatial Validation Engine for Wildfire Spread Risk Mapper.
Implements:
1. Custom SpatialBlockKFold cross-validation to prevent spatial autocorrelation data leakage.
2. Comparative modeling: Regularized Logistic Regression vs. Random Forest Classifier.
3. Strict out-of-fold spatial evaluation (ROC-AUC, PR-AUC, Brier Calibration Loss, F1).
4. Feature importance extraction and model artifact serialization.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Generator

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, classification_report
from sklearn.model_selection import KFold

from utils import setup_logger, load_config, get_project_root

LOGGER = setup_logger("modeling")


class SpatialBlockKFold:
    """
    Spatial Block Cross-Validation splitter.
    Groups adjacent spatial grid cells into geographic blocks (e.g. 25km x 25km)
    and assigns entire blocks to either train or test folds.
    Guarantees strict geographic isolation, preventing spatial autocorrelation leakage (Tobler's Law).
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields train/test index splits based on spatial group block IDs.
        """
        if groups is None:
            raise ValueError("SpatialBlockKFold requires a 'groups' parameter containing spatial block identifiers.")

        unique_blocks = np.unique(groups)
        n_blocks = len(unique_blocks)

        if n_blocks < self.n_splits:
            raise ValueError(f"Number of unique spatial blocks ({n_blocks}) is less than n_splits ({self.n_splits}).")

        # Shuffle spatial blocks
        rng = np.random.RandomState(self.random_state)
        if self.shuffle:
            shuffled_blocks = rng.permutation(unique_blocks)
        else:
            shuffled_blocks = unique_blocks

        block_folds = np.array_split(shuffled_blocks, self.n_splits)

        indices = np.arange(len(X))
        groups_arr = np.asarray(groups)

        for fold_idx in range(self.n_splits):
            test_blocks = set(block_folds[fold_idx])
            test_mask = np.isin(groups_arr, list(test_blocks))
            train_mask = ~test_mask

            train_indices = indices[train_mask]
            test_indices = indices[test_mask]

            yield train_indices, test_indices


class WildfireRiskModeler:
    """Trains and evaluates predictive wildfire probability models with spatial block validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.feature_cols = [
            "elevation_m",
            "slope_degrees",
            "aspect_degrees",
            "temp_max_c",
            "relative_humidity_min",
            "wind_speed_ms",
            "wind_gust_ms",
            "fwi_score",
            "fuel_model_group",
            "canopy_cover_pct",
            "dist_to_road_m",
            "pop_density_km2",
            "south_facing_exposure",
            "fuel_weather_hazard_index",
            "slope_spread_multiplier"
        ]
        self.target_col = "fire_occurrence"
        self.block_col = "block_id"
        self.root = get_project_root()

    def load_data(self, parquet_path: Optional[Path] = None) -> pd.DataFrame:
        """Loads processed tabular dataset containing spatial block IDs and covariates."""
        path = parquet_path or (self.root / "data" / "processed" / "ca_wildfire_grid_250m.parquet")
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found at: {path}. Run feature_extraction.py first.")
        df = pd.read_parquet(path)
        LOGGER.info(f"Loaded training dataset: {len(df)} samples, {df[self.target_col].sum()} positive fire events.")
        return df

    def evaluate_spatial_cv(
        self,
        df: pd.DataFrame,
        model_type: str = "random_forest"
    ) -> Dict[str, Any]:
        """
        Executes 5-fold Spatial Block Cross-Validation and computes out-of-fold performance.
        """
        # Ensure all feature columns exist, filter available
        avail_features = [col for col in self.feature_cols if col in df.columns]
        X = df[avail_features]
        y = df[self.target_col].values
        groups = df[self.block_col].values

        n_splits = self.config["spatial"]["spatial_cv"].get("n_splits", 5)
        sp_cv = SpatialBlockKFold(n_splits=n_splits, random_state=42)

        oof_preds = np.zeros(len(df))
        fold_metrics = []

        LOGGER.info(f"Initiating {n_splits}-fold Spatial Block CV for model: {model_type}...")

        for fold, (train_idx, val_idx) in enumerate(sp_cv.split(X, y, groups=groups), 1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            if model_type == "logistic_regression":
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=1000, random_state=42))
                ])
            elif model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=250,
                    max_depth=16,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            model.fit(X_train, y_train)

            # Predict probabilities for positive class (fire spread/ignition)
            val_probs = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_probs

            fold_roc = roc_auc_score(y_val, val_probs)
            fold_pr = average_precision_score(y_val, val_probs)
            fold_brier = brier_score_loss(y_val, val_probs)
            fold_metrics.append({"fold": fold, "roc_auc": fold_roc, "pr_auc": fold_pr, "brier_score": fold_brier})

            LOGGER.info(f"Fold {fold}/{n_splits} - ROC-AUC: {fold_roc:.4f} | PR-AUC: {fold_pr:.4f} | Brier: {fold_brier:.4f}")

        overall_roc = roc_auc_score(y, oof_preds)
        overall_pr = average_precision_score(y, oof_preds)
        overall_brier = brier_score_loss(y, oof_preds)
        binary_preds = (oof_preds >= 0.5).astype(int)
        overall_f1 = f1_score(y, binary_preds)

        LOGGER.info(f"=== {model_type.upper()} SPATIAL BLOCK CV RESULTS ===")
        LOGGER.info(f"Mean ROC-AUC: {overall_roc:.4f}")
        LOGGER.info(f"Mean PR-AUC:  {overall_pr:.4f}")
        LOGGER.info(f"Brier Score:  {overall_brier:.4f}")
        LOGGER.info(f"F1 Score:     {overall_f1:.4f}")

        return {
            "model_type": model_type,
            "overall_roc_auc": round(overall_roc, 4),
            "overall_pr_auc": round(overall_pr, 4),
            "overall_brier_loss": round(overall_brier, 4),
            "overall_f1": round(overall_f1, 4),
            "fold_metrics": fold_metrics,
            "oof_predictions": oof_preds
        }

    def train_final_model(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> Tuple[Any, Dict[str, float]]:
        """Fits final Random Forest model on complete dataset and exports serialized artifact."""
        avail_features = [col for col in self.feature_cols if col in df.columns]
        X = df[avail_features]
        y = df[self.target_col].values

        LOGGER.info(f"Training production Random Forest model on all {len(X)} cells across {len(avail_features)} features...")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Feature importances
        importances = dict(zip(avail_features, [round(float(v), 4) for v in rf.feature_importances_]))
        sorted_imp = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        LOGGER.info("Top 5 Predictive Features:")
        for feat, imp in list(sorted_imp.items())[:5]:
            LOGGER.info(f"  - {feat}: {imp:.4f}")

        out_path = output_path or (self.root / "models" / "wildfire_rf_risk_model.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": rf, "feature_names": avail_features, "feature_importances": sorted_imp}, out_path)
        LOGGER.info(f"Persisted trained model bundle to: {out_path}")

        return rf, sorted_imp


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate wildfire models with spatial block CV.")
    parser.add_argument("--data", type=str, default="data/processed/ca_wildfire_grid_250m.parquet")
    parser.add_argument("--save-metrics", type=str, default="reports/spatial_block_cv_metrics.json")
    args = parser.parse_args()

    root = get_project_root()
    modeler = WildfireRiskModeler()

    data_path = root / args.data
    df = modeler.load_data(data_path)

    # 1. Evaluate Logistic Regression under Spatial Block CV
    lr_results = modeler.evaluate_spatial_cv(df, model_type="logistic_regression")

    # 2. Evaluate Random Forest under Spatial Block CV
    rf_results = modeler.evaluate_spatial_cv(df, model_type="random_forest")

    # 3. Train final model & save
    final_model, importances = modeler.train_final_model(df)

    # Save validation report
    metrics_path = root / args.save_metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "spatial_block_kfold_summary": {
            "logistic_regression": {k: v for k, v in lr_results.items() if k != "oof_predictions"},
            "random_forest": {k: v for k, v in rf_results.items() if k != "oof_predictions"}
        },
        "feature_importances": importances
    }
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    LOGGER.info(f"Saved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
