from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import xgboost as xgb
import shap


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
FONT_PATH = PROJECT_ROOT / "assets" / "fonts" / "TimesNewRoman.ttf"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TimesNewRoman"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    }
)


TARGET_COLUMNS = {
    "cooling_kwh_per_m2": "Cooling load",
    "heating_kwh_per_m2": "Heating load",
    "other_electricity_kwh_per_m2": "Other electricity",
}

FORM_FEATURES = [
    "ci_norm",
    "vci_norm",
    "lum_norm",
    "lum_adjacency_norm",
    "lum_intensity_norm",
    "lum_proximity_norm",
    "gi_norm",
    "li_norm",
]

BUILT_FEATURES = [
    "single_family_ha",
    "multi_family_ha",
    "facility_neighborhood_ha",
    "facility_sales_ha",
    "facility_office_ha",
    "facility_education_ha",
    "facility_industrial_ha",
    "parks_green_ha",
    "water_area_ha",
    "road_area_ha",
]

TRANSPORT_FEATURES = [
    "subway_influence_ha",
    "bus_routes_cnt",
]


def _get_paths() -> Dict[str, Path]:
    mapping_root = SCRIPT_PATH.parents[2]
    model_dir = mapping_root / "20x_3y" / "xgboost"
    out_dir = SCRIPT_PATH.parent
    fig_dir = out_dir / "figures"
    dependence_dir = fig_dir / "dependence"
    summary_dir = fig_dir / "summary"
    interaction_dir = fig_dir / "interaction"
    for sub in [fig_dir, dependence_dir, summary_dir, interaction_dir]:
        sub.mkdir(parents=True, exist_ok=True)
    return {
        "mapping": mapping_root,
        "model_dir": model_dir,
        "dataset": model_dir / "dataset.csv",
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "dependence_dir": dependence_dir,
        "summary_dir": summary_dir,
        "interaction_dir": interaction_dir,
        "compare_summary": mapping_root / "compare" / "xgboost_form_vs_all_summary.csv",
    }


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = set(FORM_FEATURES + BUILT_FEATURES + TRANSPORT_FEATURES)
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def compute_shap(booster: xgb.Booster, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    dmatrix = xgb.DMatrix(data, feature_names=list(data.columns))
    shap_matrix = booster.predict(dmatrix, pred_contribs=True)
    if shap_matrix.ndim != 2:
        raise RuntimeError("Unexpected SHAP matrix shape.")
    return shap_matrix[:, :-1], shap_matrix[:, -1]


def plot_dependence(
    out_dir: Path,
    target_name: str,
    feature_name: str,
    feature_values: np.ndarray,
    shap_values: np.ndarray,
) -> Path:
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    scatter = ax.scatter(
        feature_values,
        shap_values,
        c=feature_values,
        cmap="viridis",
        s=18,
        alpha=0.55,
        linewidths=0,
    )
    bins = np.linspace(feature_values.min(), feature_values.max(), 15)
    if np.unique(bins).size > 1:
        digitized = np.digitize(feature_values, bins) - 1
        centers = []
        means = []
        for b in range(len(bins) - 1):
            mask = digitized == b
            if not np.any(mask):
                continue
            centers.append((bins[b] + bins[b + 1]) / 2)
            means.append(shap_values[mask].mean())
        if centers:
            ax.plot(
                centers,
                means,
                color="#C00000",
                linewidth=1.5,
                label="Binned mean",
                marker="o",
                markersize=3,
            )
            ax.legend()
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value (kWh/m²)")
    ax.set_title(f"{target_name} vs {feature_name}", fontweight="bold", pad=6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)
    cbar = fig.colorbar(scatter, ax=ax, label=feature_name, pad=0.02)
    cbar.ax.tick_params(labelsize=8, width=0.8, length=3)
    fig.tight_layout(pad=0.6)
    filename = f"shap_dependence_{target_name.replace(' ', '_').lower()}_{feature_name}.png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_summary(
    shap_values: np.ndarray,
    feature_frame: pd.DataFrame,
    feature_names: List[str],
    out_dir: Path,
    target_name: str,
) -> None:
    max_display = min(12, len(feature_names))
    shap.summary_plot(
        shap_values,
        feature_frame,
        feature_names=feature_names,
        max_display=max_display,
        plot_type="violin",
        color=mpl.cm.coolwarm,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(4.6, 3.6)
    ax = plt.gca()
    ax.set_title(f"{target_name} summary", fontweight="bold", pad=6)
    ax.axvline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["bottom", "left"]:
        ax.spines[spine_name].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)
    fig.tight_layout(pad=0.6)
    out_path = out_dir / f"summary_{target_name.replace(' ', '_').lower()}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def compute_interaction_values(booster: xgb.Booster, feature_matrix: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(booster)
    interaction_values = explainer.shap_interaction_values(feature_matrix)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]
    return np.array(interaction_values)


def plot_interaction_summary(
    interaction_values: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    target_name: str,
) -> None:
    feature_names_array = np.array(feature_names)
    mean_abs = np.mean(np.abs(interaction_values), axis=0)
    sym_matrix = 0.5 * (mean_abs + mean_abs.T)
    order = np.argsort(sym_matrix.sum(axis=1))[::-1]
    max_display = min(10, len(order))
    top_idx = order[:max_display]
    matrix = sym_matrix[np.ix_(top_idx, top_idx)]
    labels = feature_names_array[top_idx]

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(matrix, cmap="magma")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(f"{target_name} interaction summary", fontweight="bold", pad=6)
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)
    for spine_name in ["bottom", "left"]:
        ax.spines[spine_name].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.set_ylabel("Mean |interaction SHAP|", rotation=270, labelpad=12)
    fig.tight_layout(pad=0.6)
    out_path = out_dir / f"interaction_summary_{target_name.replace(' ', '_').lower()}.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def compute_group_importance(feature_importance: pd.Series) -> Dict[str, float]:
    return {
        "Form": float(feature_importance[FORM_FEATURES].sum()),
        "Built": float(feature_importance[BUILT_FEATURES].sum()),
        "Transport": float(feature_importance[TRANSPORT_FEATURES].sum()),
    }


def main() -> None:
    paths = _get_paths()
    df = load_dataset(paths["dataset"])
    feature_cols = [
        c for c in df.columns if c not in ["grid_id_main"] + list(TARGET_COLUMNS.keys())
    ]
    feature_idx = {feat: idx for idx, feat in enumerate(feature_cols)}
    feature_data = df[feature_cols]

    global_records = []
    group_records = []
    top_form_records = []
    dependence_records = []

    for target_col, target_label in TARGET_COLUMNS.items():
        model_path = paths["model_dir"] / f"xgb_{target_col}.json"
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        shap_values, base_values = compute_shap(booster, feature_data)
        preds = booster.predict(xgb.DMatrix(feature_data))
        residual = np.abs(shap_values.sum(axis=1) + base_values - preds).max()
        if residual > 5e-4:
            raise RuntimeError(f"SHAP sum mismatch for {target_col} (max diff {residual:.4e})")
        mean_abs = np.abs(shap_values).mean(axis=0)
        total = mean_abs.sum()
        for feature, value in zip(feature_cols, mean_abs):
            global_records.append(
                {
                    "target": target_label,
                    "feature": feature,
                    "mean_abs_shap": float(value),
                    "share": float(value / total) if total else 0.0,
                    "group": (
                        "Form"
                        if feature in FORM_FEATURES
                        else "Built"
                        if feature in BUILT_FEATURES
                        else "Transport"
                    ),
                }
            )
        group_totals = compute_group_importance(pd.Series(mean_abs, index=feature_cols))
        for group, value in group_totals.items():
            group_records.append(
                {
                    "target": target_label,
                    "group": group,
                    "mean_abs_shap": value,
                    "share": float(value / total) if total else 0.0,
                }
            )
        form_importance = pd.Series(mean_abs, index=feature_cols)[FORM_FEATURES]
        top_features = form_importance.sort_values(ascending=False).head(4)
        for rank, (feat, value) in enumerate(top_features.items(), start=1):
            top_form_records.append(
                {
                    "target": target_label,
                    "rank": rank,
                    "feature": feat,
                    "mean_abs_shap": float(value),
                    "share_within_form": float(
                        value / form_importance.sum() if form_importance.sum() else 0.0
                    ),
                }
            )
            fig_path = plot_dependence(
                paths["fig_dir"],
                target_label,
                feat,
                df[feat].to_numpy(),
                shap_values[:, feature_idx[feat]],
            )
            dependence_records.append(
                {
                    "target": target_label,
                    "feature": feat,
                    "figure": str(fig_path.relative_to(paths["mapping"])),
                }
            )
        np.save(paths["out_dir"] / f"shap_values_{target_col}.npy", shap_values)
        plot_summary(
            shap_values,
            feature_data,
            feature_cols,
            paths["summary_dir"],
            target_label,
        )
        interaction_values = compute_interaction_values(booster, feature_data)
        plot_interaction_summary(
            interaction_values,
            feature_cols,
            paths["interaction_dir"],
            target_label,
        )

    global_df = pd.DataFrame(global_records).sort_values(
        ["target", "mean_abs_shap"], ascending=[True, False]
    )
    global_df.to_csv(paths["out_dir"] / "shap_global_importance.csv", index=False)

    group_df = pd.DataFrame(group_records)
    group_df.to_csv(paths["out_dir"] / "shap_group_importance.csv", index=False)

    top_form_df = pd.DataFrame(top_form_records)
    top_form_df.to_csv(paths["out_dir"] / "shap_top_form_features.csv", index=False)

    dep_df = pd.DataFrame(dependence_records)
    dep_df.to_csv(paths["out_dir"] / "shap_dependence_plots.csv", index=False)

    compare_df = pd.read_csv(paths["compare_summary"])
    group_summary = group_df.pivot_table(index="target", columns="group", values="share").reset_index()
    merged = compare_df.merge(group_summary, on="target", how="left").rename(
        columns={
            "Form": "Form_SHAP_share",
            "Built": "Built_SHAP_share",
            "Transport": "Transport_SHAP_share",
        }
    )
    merged.to_csv(paths["out_dir"] / "r2_vs_shap_group_share.csv", index=False)


if __name__ == "__main__":
    main()
