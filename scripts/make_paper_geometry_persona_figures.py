#!/usr/bin/env python3
"""Generate paper figures for mechanism geometry and persona projection.

The script uses already-computed summary CSVs. It does not run model inference.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/persona_empathy_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
EX15_DIR = ROOT / "outputs" / "ex15_mutual_orthogonal_complement"
EX17_DIR = ROOT / "outputs" / "ex17_persona_effects_v2"
FIG_DIR = ROOT / "latex" / "figures"
SOURCE_DIR = ROOT / "outputs" / "figures" / "paper_geometry_persona"
STEERING_OVERLAY_SOURCE = SOURCE_DIR / "figure_persona_projection_steering_overlay_source.csv"

MECHANISMS = ["ER", "IP", "EX"]
MODEL_FILES = {
    "Llama": EX17_DIR / "ex17_persona_effects_v2_llama31_n200_summary_vs_person.csv",
    "Qwen": EX17_DIR / "ex17_persona_effects_v2_qwen25_n200_summary_vs_person.csv",
    "Mistral": EX17_DIR / "ex17_persona_effects_v2_mistral7b_n200_summary_vs_person.csv",
}
MODEL_COLORS = {
    "Llama": "#2563eb",
    "Qwen": "#d97706",
    "Mistral": "#059669",
}
MODEL_MARKERS = {
    "Llama": "o",
    "Qwen": "s",
    "Mistral": "^",
}
STEERING_MARKERS = {
    -1.0: "v",
    1.0: "^",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_cosine_matrix(path: Path, layer: int) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    row = df.loc[df["layer"] == layer]
    if row.empty:
        raise ValueError(f"Layer {layer} not found in {path}")
    values = row.iloc[0]

    mat = pd.DataFrame(np.eye(3), index=MECHANISMS, columns=MECHANISMS, dtype=float)
    mat.loc["ER", "EX"] = mat.loc["EX", "ER"] = values["cos_er_ex"]
    mat.loc["ER", "IP"] = mat.loc["IP", "ER"] = values["cos_er_ip"]
    mat.loc["EX", "IP"] = mat.loc["IP", "EX"] = values["cos_ex_ip"]
    return mat


def write_cosine_source(original: pd.DataFrame, residualized: pd.DataFrame, layer: int) -> None:
    rows = []
    for condition, mat in [("original", original), ("residualized", residualized)]:
        for m1, m2 in [("ER", "IP"), ("ER", "EX"), ("IP", "EX")]:
            rows.append(
                {
                    "layer": layer,
                    "condition": condition,
                    "mechanism_1": m1,
                    "mechanism_2": m2,
                    "cosine": mat.loc[m1, m2],
                }
            )
    pd.DataFrame(rows).to_csv(SOURCE_DIR / "figure_mechanism_cosine_heatmaps_source.csv", index=False)


def plot_cosine_heatmaps(layer: int = 15) -> None:
    original = load_cosine_matrix(EX15_DIR / "ex15_pairwise_cosine_wide.csv", layer)
    residualized = load_cosine_matrix(EX15_DIR / "ex15_residualized_pairwise_cosine_wide.csv", layer)
    write_cosine_source(original, residualized, layer)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f3f4f6")
    norm = TwoSlopeNorm(vmin=-0.50, vcenter=0.0, vmax=0.50)

    for ax, title, mat in [
        (axes[0], "Original vectors", original),
        (axes[1], "Residualized vectors", residualized),
    ]:
        display = mat.to_numpy(dtype=float).copy()
        np.fill_diagonal(display, np.nan)
        image = ax.imshow(display, cmap=cmap, norm=norm)

        ax.set_xticks(range(len(MECHANISMS)), labels=MECHANISMS)
        ax.set_yticks(range(len(MECHANISMS)), labels=MECHANISMS)
        ax.tick_params(axis="both", length=0, labelsize=9)
        ax.set_title(title, fontsize=10, pad=8)

        for i in range(len(MECHANISMS)):
            for j in range(len(MECHANISMS)):
                if i == j:
                    ax.text(j, i, "1.00", ha="center", va="center", color="#9ca3af", fontsize=8)
                else:
                    value = mat.iloc[i, j]
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color="#111827", fontsize=8)

        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("Cosine similarity", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Mechanism-vector geometry at layer {layer}", fontsize=11, y=1.05)

    out_base = FIG_DIR / "figure_mechanism_cosine_heatmaps"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_persona_projection_data() -> pd.DataFrame:
    frames = []
    for model, path in MODEL_FILES.items():
        require_file(path)
        df = pd.read_csv(path)
        df = df.loc[df["persona"] != "person"].copy()
        df["model"] = model
        frames.append(
            df[
                [
                    "model",
                    "persona",
                    "n",
                    "delta_ER_vs_baseline",
                    "delta_proj_er_mean_vs_baseline",
                    "delta_IP_vs_baseline",
                    "delta_EX_vs_baseline",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(
        columns={
            "delta_ER_vs_baseline": "delta_er",
            "delta_proj_er_mean_vs_baseline": "delta_proj_er",
            "delta_IP_vs_baseline": "delta_ip",
            "delta_EX_vs_baseline": "delta_ex",
        }
    )
    out.to_csv(SOURCE_DIR / "figure_persona_projection_scatter_source.csv", index=False)
    return out


def load_steering_overlay_data() -> pd.DataFrame:
    if not STEERING_OVERLAY_SOURCE.exists():
        return pd.DataFrame()
    df = pd.read_csv(STEERING_OVERLAY_SOURCE)
    required = {"model", "alpha", "delta_er", "delta_proj_er"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{STEERING_OVERLAY_SOURCE} missing columns: {missing}")
    df = df.copy()
    df["point_type"] = "steering"
    return df


def short_persona_label(persona: str) -> str:
    if persona.endswith(" person"):
        persona = persona[: -len(" person")]
    return persona


def plot_persona_projection_scatter() -> None:
    df = load_persona_projection_data()
    steering_df = load_steering_overlay_data()

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#6b7280", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhspan(-0.02, 0.02, color="#e5e7eb", alpha=0.45, zorder=0)

    for model, group in df.groupby("model", sort=False):
        ax.scatter(
            group["delta_er"],
            group["delta_proj_er"],
            label=model,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            s=42,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            zorder=3,
        )

    if not steering_df.empty:
        for alpha, group in steering_df.groupby("alpha", sort=True):
            marker = STEERING_MARKERS.get(float(alpha), "D")
            label = f"ER steering α={float(alpha):+g}"
            ax.scatter(
                group["delta_er"],
                group["delta_proj_er"],
                label=label,
                c=[MODEL_COLORS[m] for m in group["model"]],
                marker=marker,
                s=88,
                edgecolor="#111827",
                linewidth=0.9,
                alpha=0.98,
                zorder=4,
            )

    label_rows = []
    for model in ["Llama", "Qwen", "Mistral"]:
        label_rows.extend(df.loc[(df["model"] == model) & (df["persona"] == "cynical person")].to_dict("records"))
    label_rows.extend(
        df.loc[
            (df["model"] == "Llama")
            & (df["persona"].isin(["empathetic person", "black person", "engineer"]))
        ].to_dict("records")
    )

    offsets = {
        ("Llama", "cynical person"): (0.025, -0.008),
        ("Qwen", "cynical person"): (0.015, 0.004),
        ("Mistral", "cynical person"): (0.015, -0.010),
        ("Llama", "empathetic person"): (0.016, 0.006),
        ("Llama", "black person"): (0.016, 0.006),
        ("Llama", "engineer"): (0.015, -0.012),
    }
    for row in label_rows:
        dx, dy = offsets.get((row["model"], row["persona"]), (0.01, 0.004))
        label = short_persona_label(row["persona"])
        if row["persona"] == "cynical person":
            label = f"{label} ({row['model']})"
        ax.annotate(
            label,
            xy=(row["delta_er"], row["delta_proj_er"]),
            xytext=(row["delta_er"] + dx, row["delta_proj_er"] + dy),
            fontsize=9.8,
            color="#111827",
            arrowprops={"arrowstyle": "-", "color": "#9ca3af", "lw": 0.5},
        )

    ax.set_xlabel("Persona-induced ER score shift vs. neutral persona", fontsize=11.0)
    ax.set_ylabel("Shift along recovered ER direction", fontsize=11.0)

    x_values = [df["delta_er"].to_numpy(dtype=float)]
    y_values = [df["delta_proj_er"].to_numpy(dtype=float)]
    if not steering_df.empty:
        x_values.append(steering_df["delta_er"].to_numpy(dtype=float))
        y_values.append(steering_df["delta_proj_er"].to_numpy(dtype=float))
    all_x = np.concatenate(x_values)
    all_y = np.concatenate(y_values)
    ax.set_xlim(min(-0.80, float(np.nanmin(all_x)) - 0.06), max(0.25, float(np.nanmax(all_x)) + 0.06))
    ax.set_ylim(min(-0.065, float(np.nanmin(all_y)) - 0.015), max(0.045, float(np.nanmax(all_y)) + 0.015))
    ax.tick_params(axis="both", labelsize=10.0)

    model_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[model],
            color="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.55,
            markersize=6,
            label=model,
        )
        for model in ["Llama", "Qwen", "Mistral"]
    ]
    legend_1 = ax.legend(
        handles=model_handles,
        frameon=False,
        fontsize=11.0,
        loc="upper left",
        ncols=3,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    ax.add_artist(legend_1)
    if not steering_df.empty:
        steering_handles = [
            Line2D(
                [0],
                [0],
                marker="^",
                color="none",
                markerfacecolor="#ffffff",
                markeredgecolor="#111827",
                markeredgewidth=1.0,
                markersize=7,
                label=r"ER steer $\alpha=+1$",
            ),
            Line2D(
                [0],
                [0],
                marker="v",
                color="none",
                markerfacecolor="#ffffff",
                markeredgecolor="#111827",
                markeredgewidth=1.0,
                markersize=7,
                label=r"ER steer $\alpha=-1$",
            ),
        ]
        ax.legend(
            handles=steering_handles,
            frameon=False,
            fontsize=9.5,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.84),
            handletextpad=0.4,
            labelspacing=0.35,
        )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")

    out_base = FIG_DIR / "figure_persona_projection_scatter"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not steering_df.empty:
        combined = pd.concat(
            [
                df.assign(point_type="persona", alpha=np.nan, source="Ex17 persona prompt"),
                steering_df.assign(persona=np.nan, delta_ip=np.nan, delta_ex=np.nan),
            ],
            ignore_index=True,
            sort=False,
        )
        combined.to_csv(SOURCE_DIR / "figure_persona_projection_scatter_with_steering_source.csv", index=False)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    plot_cosine_heatmaps(layer=15)
    plot_persona_projection_scatter()
    print(f"[done] {FIG_DIR / 'figure_mechanism_cosine_heatmaps.pdf'}")
    print(f"[done] {FIG_DIR / 'figure_mechanism_cosine_heatmaps.png'}")
    print(f"[done] {FIG_DIR / 'figure_persona_projection_scatter.pdf'}")
    print(f"[done] {FIG_DIR / 'figure_persona_projection_scatter.png'}")
    print(f"[done] {SOURCE_DIR / 'figure_mechanism_cosine_heatmaps_source.csv'}")
    print(f"[done] {SOURCE_DIR / 'figure_persona_projection_scatter_source.csv'}")
    if STEERING_OVERLAY_SOURCE.exists():
        print(f"[done] {SOURCE_DIR / 'figure_persona_projection_scatter_with_steering_source.csv'}")


if __name__ == "__main__":
    main()
