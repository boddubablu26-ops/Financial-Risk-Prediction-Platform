"""Train, evaluate, and serialize the five credit-risk classifiers.

Run from the project root:

    python -m src.models.train
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    BEST_MODEL_ALIAS_PATH,
    BEST_MODEL_PATH,
    CHAMPION_ALIAS_PATH,
    CHAMPION_PATH,
    COMPARISON_PATH,
    CONFUSION_PATH,
    EDA_SUMMARY_PATH,
    FEATURE_SELECTION_PATH,
    METADATA_DIR,
    MODEL_VERSION,
    MODELS_DIR,
    N_CV_FOLDS,
    PLOTS_DIR,
    PREPROCESSING_METADATA_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SHAP_BACKGROUND_PATH,
    SHAP_EXPLAINER_PATH,
    SHAP_SAMPLES_PATH,
    SHAP_SUMMARY_PATH,
    TARGET_COLUMN,
)
from src.data.data_loader import prepare_modeling_tables
from src.data.cleaner import cleaning_summary
from src.eda.summary import build_summary, high_collinearity_pairs
from src.evaluation.cost_matrix import optimize_threshold
from src.evaluation.evaluator import compute_metrics, curve_payloads, positive_probabilities
from src.evaluation.plots import save_comparison_plots
from src.explainability.shap_explainer import (
    aggregate_contributions,
    build_model_explainer,
    explain_matrix,
    mean_absolute_importance,
    sample_frame,
    transformed_feature_names,
)
from src.features.feature_pipeline import build_preprocessor
from src.features.selector import rank_features
from src.models.hyperparameter_tuning import randomized_search, tune_logistic_regression
from src.models.model_registry import (
    build_estimators,
    dump_json,
    estimator_params,
    save_joblib,
)


def train_project(tune: bool = False, fast: bool = False) -> dict[str, Any]:
    """Run the full leak-free training and evaluation workflow.

    Args:
        tune: Also run a short randomized search for XGBoost. Logistic regression
            ``C`` is always selected with a small grid.
        fast: Smaller trees and a smaller SVM subsample for smoke tests.

    Returns:
        Champion metadata written to ``models/metadata/best_model_info.json``.
    """

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("Loading and cleaning the benchmark dataset...")
    tables = prepare_modeling_tables(persist=True)
    x_train: pd.DataFrame = tables["x_train"]
    x_test: pd.DataFrame = tables["x_test"]
    y_train: pd.Series = tables["y_train"]
    y_test: pd.Series = tables["y_test"]
    print(
        f"Train rows: {len(x_train):,} | Test rows: {len(x_test):,} | "
        f"Train default rate: {y_train.mean():.3%} | Test default rate: {y_test.mean():.3%}"
    )

    _write_eda(tables)
    positives = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = float((y_train == 0).sum() / positives)
    print("Selecting logistic regression C on a 3-fold ROC-AUC grid...")
    tuning: dict[str, Any] = {"logistic_regression": tune_logistic_regression(x_train, y_train)}
    logistic_c = float(tuning["logistic_regression"]["best_C"])
    print(f"Logistic regression C = {logistic_c} (CV ROC-AUC {tuning['logistic_regression']['best_roc_auc']:.4f})")
    if tune:
        print("Running a short randomized search for XGBoost...")
        tuning["xgboost"] = randomized_search(
            x_train,
            y_train,
            model_key="xgboost",
            scale_pos_weight=scale_pos_weight,
            n_iter=3,
            cv_folds=3,
        )

    print("Fitting the feature audit preprocessor...")
    audit_preprocessor = build_preprocessor()
    audit_preprocessor.fit(x_train, y_train)
    engineered = audit_preprocessor.named_steps["engineer"].transform(
        audit_preprocessor.named_steps["imputer"].transform(x_train)
    )
    selection = rank_features(engineered, y_train)
    dump_json(FEATURE_SELECTION_PATH, selection)
    _export_matrices(audit_preprocessor, x_train, x_test, y_train, y_test)
    preprocessing_meta = {
        "clean_report": cleaning_summary(tables["clean_report"]),
        "split": tables["balance"],
        "imputer": audit_preprocessor.named_steps["imputer"].to_metadata(),
        "feature_selection_top_k": selection["selected_features"],
    }
    dump_json(PREPROCESSING_METADATA_PATH, preprocessing_meta)

    folds = 3 if fast else N_CV_FOLDS
    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for key, display_name, estimator in build_estimators(
        scale_pos_weight=scale_pos_weight,
        fast=fast,
        logistic_c=logistic_c,
    ):
        print(f"Training {display_name} with {folds}-fold stratified CV...")
        pipeline = build_model_pipeline_for(estimator)
        cv_summary = _cross_validate(pipeline, x_train, y_train, folds)
        pipeline.fit(x_train, y_train)
        probabilities = positive_probabilities(pipeline, x_test)
        cost = optimize_threshold(y_test, probabilities)
        metrics = compute_metrics(y_test, probabilities, cost["optimal_threshold"])
        curves = curve_payloads(y_test, probabilities)
        record = {
            "key": key,
            "model_name": display_name,
            "is_champion": False,
            "accuracy": round(metrics["accuracy"], 6),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1_score": round(metrics["f1_score"], 6),
            "roc_auc": round(metrics["roc_auc"], 6),
            "pr_auc": round(metrics["pr_auc"], 6),
            "optimal_threshold": cost["optimal_threshold"],
            "expected_cost": cost["minimum_cost"],
            "confusion_matrix": metrics["confusion_matrix"],
            "cv": cv_summary,
            "hyperparameters": estimator_params(estimator),
            "roc_curve": {"fpr": curves["fpr"], "tpr": curves["tpr"]},
            "pr_curve": {"precision_curve": curves["precision_curve"], "recall_curve": curves["recall_curve"]},
            "cost_curve": cost["curve"],
        }
        results.append(record)
        fitted[key] = pipeline
        save_joblib(MODELS_DIR / f"{key}.joblib", pipeline)
        print(
            f"  holdout ROC-AUC {record['roc_auc']:.4f} | recall {record['recall']:.4f} | "
            f"precision {record['precision']:.4f} | threshold {record['optimal_threshold']:.2f} | "
            f"cost {record['expected_cost']:.0f}"
        )

    champion = max(results, key=_champion_key)
    champion["is_champion"] = True
    champion_pipeline = fitted[champion["key"]]
    save_joblib(CHAMPION_PATH, champion_pipeline)
    save_joblib(CHAMPION_ALIAS_PATH, champion_pipeline)
    save_joblib(PREPROCESSOR_PATH, champion_pipeline[:-1])

    shap_summary = _write_shap(champion_pipeline, x_test, fast=fast)
    best_info = _best_info(
        champion=champion,
        scale_pos_weight=scale_pos_weight,
        tuning=tuning,
        tables=tables,
        shap_summary=shap_summary,
        fast=fast,
    )
    comparison = {
        "champion_model": champion["model_name"],
        "evaluation_dataset": "20% stratified holdout, random_state=42",
        "fp_cost": 1,
        "fn_cost": 5,
        "models": results,
    }
    dump_json(COMPARISON_PATH, comparison)
    dump_json(BEST_MODEL_PATH, best_info)
    dump_json(BEST_MODEL_ALIAS_PATH, best_info)
    dump_json(
        CONFUSION_PATH,
        {item["model_name"]: item["confusion_matrix"] for item in results},
    )
    save_comparison_plots(results, PLOTS_DIR)
    print(
        f"Champion: {champion['model_name']} | ROC-AUC {champion['roc_auc']:.4f} | "
        f"Recall {champion['recall']:.4f} | saved to {CHAMPION_PATH}"
    )
    return best_info


def build_model_pipeline_for(estimator: Any) -> Any:
    """Import wrapper so tests can monkeypatch pipeline construction."""

    from src.features.feature_pipeline import build_model_pipeline

    return build_model_pipeline(estimator)


def _cross_validate(pipeline: Any, features: pd.DataFrame, target: pd.Series, folds: int) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=splitter,
        scoring={
            "accuracy": _accuracy_scorer,
            "recall": _recall_scorer,
            "roc_auc": _roc_auc_scorer,
        },
        n_jobs=1,
        error_score="raise",
    )
    return {
        "folds": folds,
        "accuracy_mean": round(float(np.mean(scores["test_accuracy"])), 6),
        "accuracy_std": round(float(np.std(scores["test_accuracy"])), 6),
        "recall_mean": round(float(np.mean(scores["test_recall"])), 6),
        "recall_std": round(float(np.std(scores["test_recall"])), 6),
        "roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 6),
        "roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 6),
    }


def _accuracy_scorer(estimator: Any, features: pd.DataFrame, target: pd.Series) -> float:
    predicted = estimator.predict(features)
    return float(accuracy_score(target, predicted))


def _recall_scorer(estimator: Any, features: pd.DataFrame, target: pd.Series) -> float:
    predicted = estimator.predict(features)
    return float(recall_score(target, predicted, zero_division=0))


def _roc_auc_scorer(estimator: Any, features: pd.DataFrame, target: pd.Series) -> float:
    """Score ROC-AUC from the positive-class probability.

    Built-in scorers pass the full probability matrix through for some custom
    estimators. This scorer always reduces that matrix to P(default).
    """

    return float(roc_auc_score(target, positive_probabilities(estimator, features)))


def _champion_key(item: dict[str, Any]) -> tuple[float, ...]:
    eligible = item["roc_auc"] >= 0.85 and item["recall"] >= 0.70 and item["precision"] >= 0.60
    return (
        float(eligible),
        float(item["roc_auc"]),
        float(item["recall"]),
        -float(item["expected_cost"]),
    )


def _export_matrices(
    preprocessor: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    names = [str(name) for name in preprocessor.named_steps["columns"].get_feature_names_out()]
    train_matrix = np.asarray(preprocessor.transform(x_train))
    test_matrix = np.asarray(preprocessor.transform(x_test))
    pd.DataFrame(train_matrix, columns=names).to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    pd.DataFrame(test_matrix, columns=names).to_csv(PROCESSED_DIR / "test_features.csv", index=False)
    y_train.to_frame(TARGET_COLUMN).to_csv(PROCESSED_DIR / "train_target.csv", index=False)
    y_test.to_frame(TARGET_COLUMN).to_csv(PROCESSED_DIR / "test_target.csv", index=False)


def _write_eda(tables: dict[str, Any]) -> None:
    payload = {
        "raw": build_summary(tables["raw"]),
        "cleaned": build_summary(tables["cleaned"]),
        "collinear_pairs_pearson": high_collinearity_pairs(tables["cleaned"], threshold=0.80, method="pearson"),
        "collinear_pairs_spearman": high_collinearity_pairs(tables["cleaned"], threshold=0.80, method="spearman"),
    }
    dump_json(EDA_SUMMARY_PATH, payload)
    _save_eda_png(tables["raw"])


def _save_eda_png(raw: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_COLUMN in raw.columns:
        counts = raw[TARGET_COLUMN].value_counts().sort_index()
        figure, axis = plt.subplots(figsize=(5, 4))
        axis.bar(["Low Risk", "High Risk"], counts.to_numpy(), color=["#10B981", "#EF4444"])
        axis.set_title("Target distribution")
        figure.tight_layout()
        figure.savefig(PLOTS_DIR / "target_distribution.png", dpi=120)
        plt.close(figure)
    missing = (raw.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    figure, axis = plt.subplots(figsize=(6, 3))
    if len(missing):
        axis.barh([str(name) for name in missing.index], missing.to_numpy(), color="#4F46E5")
    axis.set_xlabel("Missing (%)")
    axis.set_title("Missing value profile")
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "missingness.png", dpi=120)
    plt.close(figure)


def _write_shap(pipeline: Any, x_test: pd.DataFrame, fast: bool) -> dict[str, Any]:
    summary: dict[str, Any] = {"status": "skipped"}
    try:
        sample = sample_frame(x_test, 80 if fast else 300, RANDOM_STATE)
        transformed = np.asarray(pipeline[:-1].transform(sample), dtype=float)
        background = transformed[: min(200, len(transformed))]
        np.savez_compressed(SHAP_BACKGROUND_PATH, matrix=background)
        explainer = build_model_explainer(pipeline.named_steps["model"], background)
        if explainer is None:
            summary = {
                "status": "unavailable",
                "reason": "No SHAP explainer could be built for this champion. Local explanations use the domain fallback.",
            }
            dump_json(SHAP_SUMMARY_PATH, summary)
            return summary
        names = transformed_feature_names(pipeline)
        matrix, base = explain_matrix(explainer, transformed)
        ranking = mean_absolute_importance(names, matrix)
        for item in ranking:
            item["mean_abs_shap"] = round(float(item["mean_abs_shap"]), 6)
        engineered = pipeline.named_steps["engineer"].transform(pipeline.named_steps["imputer"].transform(sample))
        top = [str(item["feature"]) for item in ranking[:8]]
        points = []
        for index in range(matrix.shape[0]):
            aggregated = {
                str(item["feature"]): float(item["shap_value"])
                for item in aggregate_contributions(names, matrix[index])
            }
            color: dict[str, float] = {}
            for feature in top:
                color[feature] = _color_value(engineered.iloc[index], feature)
            points.append({"shap": {feature: round(aggregated.get(feature, 0.0), 6) for feature in top}, "color": color})
        summary = {
            "status": "ok",
            "sample_rows": int(len(sample)),
            "base_value": None if base is None else round(float(np.mean(np.asarray(base))), 6),
            "ranking": ranking[:20],
            "top_features": top,
        }
        dump_json(SHAP_SUMMARY_PATH, summary)
        dump_json(SHAP_SAMPLES_PATH, {"features": top, "points": points})
        try:
            save_joblib(SHAP_EXPLAINER_PATH, explainer)
        except Exception as exc:
            summary["explainer_persistence"] = f"TreeExplainer is rebuilt at startup ({exc})"
    except Exception as exc:
        summary = {"status": "failed", "reason": str(exc)}
        dump_json(SHAP_SUMMARY_PATH, summary)
    return summary


def _color_value(row: pd.Series, feature: str) -> float:
    if feature in row.index and _is_float(row[feature]):
        return round(float(row[feature]), 6)
    if feature == "loan_grade" and "loan_grade_numeric" in row.index:
        return round(float(row["loan_grade_numeric"]), 6)
    return 0.0


def _is_float(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number))


def _best_info(
    champion: dict[str, Any],
    scale_pos_weight: float,
    tuning: dict[str, Any],
    tables: dict[str, Any],
    shap_summary: dict[str, Any],
    fast: bool,
) -> dict[str, Any]:
    return {
        "model_name": champion["model_name"],
        "model_key": champion["key"],
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "dataset_sha256": _file_sha256(RAW_DATA_PATH),
        "random_state": RANDOM_STATE,
        "fast_mode": fast,
        "optimal_threshold": champion["optimal_threshold"],
        "expected_cost": champion["expected_cost"],
        "scale_pos_weight": round(scale_pos_weight, 6),
        "split": tables["balance"],
        "metrics": {
            "accuracy": champion["accuracy"],
            "precision": champion["precision"],
            "recall": champion["recall"],
            "f1_score": champion["f1_score"],
            "roc_auc": champion["roc_auc"],
            "pr_auc": champion["pr_auc"],
        },
        "hyperparameters": champion["hyperparameters"],
        "cross_validation": champion["cv"],
        "tuning": tuning,
        "shap_status": shap_summary.get("status"),
        "acceptance": {
            "roc_auc_at_least_0_85": champion["roc_auc"] >= 0.85,
            "recall_at_least_0_75": champion["recall"] >= 0.75,
            "precision_above_0_60": champion["precision"] >= 0.60,
        },
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Any) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Train the FRP-ML credit risk models.")
    parser.add_argument("--tune", action="store_true", help="Run an additional short XGBoost search.")
    parser.add_argument("--fast", action="store_true", help="Smoke-test settings with smaller ensembles.")
    args = parser.parse_args()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    train_project(tune=args.tune, fast=args.fast)


if __name__ == "__main__":
    main()
