#!/usr/bin/env python3
"""Ex15: pairwise cosine and mutual orthogonal complement steering.

This script uses the Llama-3.1-8B vector files produced by:
  - Ex11 (ER)
  - Ex12 (EX)
  - Ex13 (IP)

It performs two related analyses on the shared layer set:
  1) pairwise cosine similarity among ER/EX/IP vectors at each layer
  2) residualized steering, where the target vector is projected onto the
     orthogonal complement of the other two vectors before steering

The steering part reuses the evaluation splits stored by Ex11/12/13 and
classifies generated responses with the EPITOME ER/IP/EX classifiers.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_llama31_multi_response_activation_task1 import (
    generate_n_responses_for_batch,
    get_input_device,
    resolve_torch_dtype,
    set_seed,
)
from run_llama31_persona_epitome_experiment import classify_er_ip_ex


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPITOME_PROJECT_DEFAULT = PROJECT_ROOT.parent / "Empathy-Mental-Health"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class TaskSpec:
    task: str
    label: str
    exp_dir: str
    vector_prefix: str
    label_col: str
    summary_col: str


TASKS: Sequence[TaskSpec] = (
    TaskSpec("ER", "ER", "outputs/ex11_filtered_er", "ex11_filtered_er_llama31", "ER_label", "ER_mean"),
    TaskSpec("EX", "EX", "outputs/ex12_filtered_ex", "ex12_filtered_ex_llama31", "EX_label", "EX_mean"),
    TaskSpec("IP", "IP", "outputs/ex13_filtered_ip", "ex13_filtered_ip_llama31", "IP_label", "IP_mean"),
)

DEFAULT_LAYERS = "3,7,11,15,19,23,27,31"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex15 mutual orthogonal complement steering")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--epitome-project", type=Path, default=EPITOME_PROJECT_DEFAULT)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--layers", type=str, default=DEFAULT_LAYERS)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 0.0, 1.0, 2.0])
    p.add_argument(
        "--residual-scale",
        type=str,
        default="matched",
        choices=["matched", "unit", "raw"],
        help="How to scale residualized vectors: matched norm, unit norm, or raw residual.",
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--skip-steering", action="store_true")
    p.add_argument("--skip-classification", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex15_mutual_orthogonal_complement"))
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def parse_layers(layer_text: str) -> List[int]:
    layers = [int(tok.strip()) for tok in (layer_text or "").split(",") if tok.strip()]
    if not layers:
        raise ValueError("--layers must include at least one layer index")
    return sorted(set(layers))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def find_latest(pattern_dir: Path, pattern: str) -> Path:
    matches = sorted(pattern_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files found for pattern: {pattern_dir / pattern}")
    return matches[0]


def load_layer_vectors(npz_path: Path) -> Dict[int, np.ndarray]:
    data = np.load(npz_path)
    if "layers" not in data or "vectors" not in data:
        raise ValueError(f"Invalid npz format: {npz_path}")
    layers = [int(x) for x in np.asarray(data["layers"]).tolist()]
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(layers):
        raise ValueError(f"Layer/vector shape mismatch in {npz_path}: {vectors.shape} vs {len(layers)}")
    return {int(layer): np.asarray(vec, dtype=np.float32) for layer, vec in zip(layers, vectors)}


def residualize_vector(target: np.ndarray, basis: List[np.ndarray], scale_mode: str) -> Tuple[np.ndarray, dict]:
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    if not basis:
        resid = t.copy()
    else:
        B = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in basis], axis=1)
        coeffs, *_ = np.linalg.lstsq(B, t, rcond=None)
        resid = t - B @ coeffs

    orig_norm = float(np.linalg.norm(t))
    resid_norm = float(np.linalg.norm(resid))
    if scale_mode == "matched" and resid_norm > 0.0 and orig_norm > 0.0:
        resid = resid * (orig_norm / resid_norm)
    elif scale_mode == "unit" and resid_norm > 0.0:
        resid = resid / resid_norm

    info = {
        "original_norm": orig_norm,
        "residual_norm_raw": resid_norm,
        "residual_norm_final": float(np.linalg.norm(resid)),
        "residual_cosine_with_original": cosine(t, resid),
        "scale_mode": scale_mode,
    }
    return resid.astype(np.float32), info


def get_transformer_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture")


class LayerVectorSteerer:
    def __init__(self, model: AutoModelForCausalLM, layer_idx: int, vector: torch.Tensor, coeff: float):
        layers = get_transformer_layers(model)
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"layer index out of range: {layer_idx} (n_layers={len(layers)})")
        self.target_layer = layers[layer_idx]
        self.vector = vector.detach().float().cpu()
        self.coeff = float(coeff)
        self.handle = None

    def _hook_fn(self, _module, _inputs, outputs):
        if self.coeff == 0.0:
            return outputs
        if isinstance(outputs, tuple):
            hidden, rest = outputs[0], outputs[1:]
        else:
            hidden, rest = outputs, None
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            return outputs
        vec = self.vector.to(hidden.device, dtype=hidden.dtype).view(1, 1, -1)
        steered = hidden.clone()
        steered[:, -1, :] = steered[:, -1, :] + (self.coeff * vec.squeeze(0).squeeze(0))
        return (steered,) + rest if rest is not None else steered

    def __enter__(self):
        self.handle = self.target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


def build_prompt(seeker_post: str) -> str:
    return f"Seeker post:\n{seeker_post}\n\nResponse:"


def load_eval_seekers(exp_dir: Path, prefix: str) -> pd.DataFrame:
    path = find_latest(exp_dir, f"{prefix}_*_eval_seekers.csv")
    df = pd.read_csv(path)
    if "seeker_post" not in df.columns:
        raise ValueError(f"Missing seeker_post column: {path}")
    return df


def load_summary(exp_dir: Path, prefix: str) -> pd.DataFrame:
    path = find_latest(exp_dir, f"{prefix}_*_summary.csv")
    return pd.read_csv(path)


def build_pairwise_cosine_tables(
    layer_vectors: Dict[str, Dict[int, np.ndarray]],
    layers: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[dict] = []
    wide_rows: List[dict] = []
    pairs = [("ER", "EX"), ("ER", "IP"), ("EX", "IP")]

    common_layers = [layer for layer in layers if all(layer in layer_vectors[k] for k in layer_vectors)]
    if not common_layers:
        raise ValueError("No overlapping layers among ER/EX/IP vectors")

    for layer in common_layers:
        layer_vecs = {k: layer_vectors[k][layer] for k in ["ER", "EX", "IP"]}
        wide_row = {"layer": int(layer)}
        for a, b in pairs:
            cos = cosine(layer_vecs[a], layer_vecs[b])
            rows.append(
                {
                    "layer": int(layer),
                    "pair": f"{a}_vs_{b}",
                    "vector_a": a,
                    "vector_b": b,
                    "cosine": cos,
                    "norm_a": float(np.linalg.norm(layer_vecs[a])),
                    "norm_b": float(np.linalg.norm(layer_vecs[b])),
                }
            )
            wide_row[f"cos_{a.lower()}_{b.lower()}"] = cos
        wide_rows.append(wide_row)

    long_df = pd.DataFrame(rows).sort_values(["pair", "layer"]).reset_index(drop=True)
    wide_df = pd.DataFrame(wide_rows).sort_values(["layer"]).reset_index(drop=True)
    summary_df = (
        long_df.groupby("pair", as_index=False)
        .agg(
            n_layers=("layer", "count"),
            cosine_mean=("cosine", "mean"),
            cosine_std=("cosine", "std"),
            cosine_min=("cosine", "min"),
            cosine_max=("cosine", "max"),
        )
        .reset_index(drop=True)
    )
    summary_df["cosine_std"] = summary_df["cosine_std"].fillna(0.0)
    return long_df, wide_df, summary_df


def summarize_by_layer_alpha(
    df: pd.DataFrame,
    label_col: str,
    summary_col: str,
    baseline: float,
) -> pd.DataFrame:
    rows: List[dict] = []
    dim_prefix = summary_col.replace("_mean", "")
    for (layer, alpha), g in df.groupby(["layer", "alpha"], sort=True):
        mean_metric = float(g[label_col].mean())
        rows.append(
            {
                "layer": int(layer),
                "alpha": float(alpha),
                "n": int(len(g)),
                summary_col: mean_metric,
                f"{dim_prefix}_pct_0": float((g[label_col] == 0).mean()),
                f"{dim_prefix}_pct_1": float((g[label_col] == 1).mean()),
                f"{dim_prefix}_pct_2": float((g[label_col] == 2).mean()),
                "delta_vs_unsteered": mean_metric - baseline,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    alpha0 = out[np.isclose(out["alpha"], 0.0)].set_index("layer")[summary_col].to_dict()
    out["delta_vs_alpha0"] = out.apply(
        lambda r: r[summary_col] - alpha0.get(int(r["layer"]), float("nan")), axis=1
    )
    return out.sort_values(["layer", "alpha"]).reset_index(drop=True)


def classify_generated_rows(
    generated_df: pd.DataFrame,
    epitome_project: Path,
    er_model_path: Path,
    ip_model_path: Path,
    ex_model_path: Path,
    batch_size: int,
) -> pd.DataFrame:
    return classify_er_ip_ex(
        df=generated_df,
        epitome_project=epitome_project,
        er_model_path=er_model_path,
        ip_model_path=ip_model_path,
        ex_model_path=ex_model_path,
        batch_size=batch_size,
        max_length=64,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    persona_project = args.persona_project.resolve()
    epitome_project = args.epitome_project.resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = parse_layers(args.layers)
    if not layers:
        raise ValueError("No layers selected")

    task_vectors: Dict[str, Dict[int, np.ndarray]] = {}
    task_eval_seekers: Dict[str, pd.DataFrame] = {}
    task_original_summaries: Dict[str, pd.DataFrame] = {}
    task_paths: Dict[str, dict] = {}

    for task in TASKS:
        exp_dir = persona_project / task.exp_dir
        vec_path = find_latest(exp_dir, f"{task.vector_prefix}_*_layer_vectors.npz")
        eval_path = find_latest(exp_dir, f"{task.vector_prefix}_*_eval_seekers.csv")
        summary_path = find_latest(exp_dir, f"{task.vector_prefix}_*_summary.csv")

        task_vectors[task.task] = load_layer_vectors(vec_path)
        task_eval_seekers[task.task] = pd.read_csv(eval_path)
        task_original_summaries[task.task] = pd.read_csv(summary_path)
        task_paths[task.task] = {"vectors": vec_path, "eval_seekers": eval_path, "summary": summary_path}

    pair_long, pair_wide, pair_summary = build_pairwise_cosine_tables(task_vectors, layers)

    out_pair_long = output_dir / "ex15_pairwise_cosine_per_layer.csv"
    out_pair_wide = output_dir / "ex15_pairwise_cosine_wide.csv"
    out_pair_summary = output_dir / "ex15_pairwise_cosine_summary.csv"

    if not args.overwrite:
        for p in [out_pair_long, out_pair_wide, out_pair_summary]:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")

    pair_long.to_csv(out_pair_long, index=False)
    pair_wide.to_csv(out_pair_wide, index=False)
    pair_summary.to_csv(out_pair_summary, index=False)

    resid_info_rows: List[dict] = []
    resid_vectors: Dict[str, Dict[int, np.ndarray]] = {"ER": {}, "EX": {}, "IP": {}}
    for layer in layers:
        v_er = task_vectors["ER"].get(layer)
        v_ex = task_vectors["EX"].get(layer)
        v_ip = task_vectors["IP"].get(layer)
        if v_er is None or v_ex is None or v_ip is None:
            continue

        resid_er, info_er = residualize_vector(v_er, [v_ip, v_ex], args.residual_scale)
        resid_ex, info_ex = residualize_vector(v_ex, [v_er, v_ip], args.residual_scale)
        resid_ip, info_ip = residualize_vector(v_ip, [v_er, v_ex], args.residual_scale)

        resid_vectors["ER"][layer] = resid_er
        resid_vectors["EX"][layer] = resid_ex
        resid_vectors["IP"][layer] = resid_ip

        resid_info_rows.extend(
            [
                {"task": "ER", "layer": int(layer), **info_er},
                {"task": "EX", "layer": int(layer), **info_ex},
                {"task": "IP", "layer": int(layer), **info_ip},
            ]
        )

    resid_info_df = pd.DataFrame(resid_info_rows).sort_values(["task", "layer"]).reset_index(drop=True)
    out_resid_info = output_dir / "ex15_residualized_vector_info.csv"
    resid_info_df.to_csv(out_resid_info, index=False)

    resid_pair_long, resid_pair_wide, resid_pair_summary = build_pairwise_cosine_tables(resid_vectors, layers)
    out_resid_pair_long = output_dir / "ex15_residualized_pairwise_cosine_per_layer.csv"
    out_resid_pair_wide = output_dir / "ex15_residualized_pairwise_cosine_wide.csv"
    out_resid_pair_summary = output_dir / "ex15_residualized_pairwise_cosine_summary.csv"
    resid_pair_long.to_csv(out_resid_pair_long, index=False)
    resid_pair_wide.to_csv(out_resid_pair_wide, index=False)
    resid_pair_summary.to_csv(out_resid_pair_summary, index=False)

    if not args.skip_steering:
        er_model_path = resolve_path(epitome_project, Path("output/reddit_ER.pth")).resolve()
        ip_model_path = resolve_path(epitome_project, Path("output/reddit_IP.pth")).resolve()
        ex_model_path = resolve_path(epitome_project, Path("output/reddit_EX.pth")).resolve()

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=resolve_torch_dtype(args.torch_dtype),
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        input_device = get_input_device(model)
        n_model_layers = len(get_transformer_layers(model))
        bad_layers = [l for l in layers if l < 0 or l >= n_model_layers]
        if bad_layers:
            raise ValueError(f"Invalid layers {bad_layers}; model has {n_model_layers} layers")

        steering_summaries: List[pd.DataFrame] = []
        steering_comparisons: List[pd.DataFrame] = []
        generated_manifest: List[dict] = []

        for task in TASKS:
            eval_df = task_eval_seekers[task.task].copy()
            eval_df = eval_df.reset_index(drop=True)
            if "seeker_post" not in eval_df.columns:
                raise ValueError(f"Missing seeker_post column in eval split for {task.task}")

            generated_rows: List[dict] = []
            run_name = f"ex15_{task.task.lower()}_residualized_{args.model_name.split('/')[-1].lower()}"
            out_generated = output_dir / f"{run_name}_generated.csv"
            out_classified = output_dir / f"{run_name}_classified.csv"
            out_summary = output_dir / f"{run_name}_summary.csv"

            label_col = task.label_col
            summary_col = task.summary_col
            baseline_alpha0 = float(
                task_original_summaries[task.task][
                    np.isclose(task_original_summaries[task.task]["alpha"], 0.0)
                ][summary_col].mean()
            )

            for layer in layers:
                if layer not in resid_vectors[task.task]:
                    continue
                vec_t = torch.tensor(resid_vectors[task.task][layer], dtype=torch.float32)
                for alpha in args.alphas:
                    set_seed(args.seed)
                    with LayerVectorSteerer(model, layer, vec_t, float(alpha)):
                        prompts = [build_prompt(sp) for sp in eval_df["seeker_post"].astype(str).tolist()]
                        for start in range(0, len(prompts), args.batch_size):
                            batch_prompts = prompts[start:start + args.batch_size]
                            grouped = generate_n_responses_for_batch(
                                model=model,
                                tokenizer=tokenizer,
                                input_device=input_device,
                                prompt_texts=batch_prompts,
                                n_responses=1,
                                max_input_tokens=args.max_length,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                            )
                            for i, responses in enumerate(grouped):
                                idx = start + i
                                text = str(responses[0])
                                generated_rows.append(
                                    {
                                        "task": task.task,
                                        "layer": int(layer),
                                        "alpha": float(alpha),
                                        "base_sample_index": int(idx),
                                        "seeker_post": eval_df.iloc[idx]["seeker_post"],
                                        "generated_response": text,
                                        "response_post": text,
                                        "response_token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                                        "model_name": args.model_name,
                                        "residual_scale": args.residual_scale,
                                    }
                                )

            generated_df = pd.DataFrame(generated_rows).sort_values(["layer", "alpha", "base_sample_index"]).reset_index(drop=True)
            generated_df.to_csv(out_generated, index=False)

            if args.skip_classification:
                continue

            classified_df = classify_generated_rows(
                generated_df=generated_df,
                epitome_project=epitome_project,
                er_model_path=er_model_path,
                ip_model_path=ip_model_path,
                ex_model_path=ex_model_path,
                batch_size=args.classifier_batch_size,
            )
            classified_df.to_csv(out_classified, index=False)

            summary_df = summarize_by_layer_alpha(
                classified_df,
                label_col=label_col,
                summary_col=summary_col,
                baseline=baseline_alpha0,
            )
            summary_df.to_csv(out_summary, index=False)

            original_summary = task_original_summaries[task.task].copy()
            orig_metric = original_summary[["layer", "alpha", summary_col, "delta_vs_alpha0"]].rename(
                columns={summary_col: f"original_{summary_col}", "delta_vs_alpha0": "original_delta_vs_alpha0"}
            )
            resid_metric = summary_df[["layer", "alpha", summary_col, "delta_vs_alpha0"]].rename(
                columns={summary_col: f"residual_{summary_col}", "delta_vs_alpha0": "residual_delta_vs_alpha0"}
            )
            comparison = orig_metric.merge(resid_metric, on=["layer", "alpha"], how="inner")
            comparison["task"] = task.task
            steering_comparisons.append(comparison)
            steering_summaries.append(summary_df.assign(task=task.task))
            generated_manifest.append(
                {
                    "task": task.task,
                    "generated_csv": str(out_generated),
                    "classified_csv": str(out_classified),
                    "summary_csv": str(out_summary),
                }
            )

        if steering_summaries:
            all_summary = pd.concat(steering_summaries, ignore_index=True)
            all_summary.to_csv(output_dir / "ex15_residualized_all_tasks_summary.csv", index=False)
        if steering_comparisons:
            all_comp = pd.concat(steering_comparisons, ignore_index=True)
            all_comp.to_csv(output_dir / "ex15_original_vs_residualized_comparison.csv", index=False)
        (output_dir / "ex15_generated_manifest.json").write_text(
            json.dumps(generated_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    report_lines: List[str] = []
    report_lines.append("Ex15 mutual orthogonal complement steering")
    report_lines.append("")
    report_lines.append("Pairwise cosine summary (original vectors):")
    for row in pair_summary.itertuples(index=False):
        report_lines.append(
            f"  {row.pair}: mean={row.cosine_mean:+.4f}, std={row.cosine_std:.4f}, "
            f"min={row.cosine_min:+.4f}, max={row.cosine_max:+.4f}, n={int(row.n_layers)}"
        )
    report_lines.append("")
    report_lines.append("Pairwise cosine summary (residualized vectors):")
    for row in resid_pair_summary.itertuples(index=False):
        report_lines.append(
            f"  {row.pair}: mean={row.cosine_mean:+.4f}, std={row.cosine_std:.4f}, "
            f"min={row.cosine_min:+.4f}, max={row.cosine_max:+.4f}, n={int(row.n_layers)}"
        )

    out_report = output_dir / "ex15_report.txt"
    out_meta = output_dir / "ex15_metadata.json"
    out_meta.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "layers": layers,
                "alphas": [float(a) for a in args.alphas],
                "residual_scale": args.residual_scale,
                "vector_sources": {k: {kk: str(vv) for kk, vv in task_paths[k].items()} for k in task_paths},
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out_report.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    print("\n".join(report_lines).rstrip())
    print(f"\n[done] pairwise cosine: {out_pair_long}")
    print(f"[done] residualized vector info: {out_resid_info}")
    print(f"[done] residualized pairwise cosine: {out_resid_pair_long}")
    print(f"[done] report: {out_report}")
    print(f"[done] metadata: {out_meta}")


if __name__ == "__main__":
    main()
