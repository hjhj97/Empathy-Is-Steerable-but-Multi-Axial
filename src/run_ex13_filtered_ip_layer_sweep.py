#!/usr/bin/env python3
"""Ex13: filtered IP dataset layer-sweep steering experiment.

Design summary:
1) Build IP steering vectors with CAA contrast on filtered dataset:
   - positive pool: IP level >= 1
   - negative pool: IP level == 0
   - sample N from each pool (default N=100)
   - vector per layer = mean(hidden_last | pos) - mean(hidden_last | neg)
2) Pick disjoint evaluation seekers (default 200) to avoid data leakage.
3) For each (layer, alpha), generate responses and classify with EPITOME ER/IP/EX.
4) Save generated responses for later LLM-judge use.
"""

from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_llama31_multi_response_activation_task1 import (
    ensure_id_column,
    generate_n_responses_for_batch,
    get_input_device,
    resolve_torch_dtype,
    set_seed,
)
from run_llama31_persona_epitome_experiment import classify_er_ip_ex


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPITOME_PROJECT_DEFAULT = PROJECT_ROOT.parent / "Empathy-Mental-Health"
DEFAULT_DATASET = Path("dataset/filtered/interpretations-reddit-filtered.csv")

TF_PROMPT_TEMPLATE = "Seeker post:\n{seeker_post}\n\nResponse:{response_text}"
GEN_PROMPT_TEMPLATE = "Seeker post:\n{seeker_post}\n\nResponse:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex13 filtered IP layer-sweep steering")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--epitome-project", type=Path, default=EPITOME_PROJECT_DEFAULT)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--eval-seekers-path",
        type=Path,
        default=None,
        help="Optional CSV with source_id,seeker_post; bypasses internal eval sampling.",
    )

    p.add_argument("--seeker-column", type=str, default="seeker_post")
    p.add_argument("--response-column", type=str, default="response_post")
    p.add_argument("--level-column", type=str, default="level")
    p.add_argument("--id-column", type=str, default="sp_id")

    p.add_argument("--vector-size-per-class", type=int, default=100)
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--level-threshold", type=int, default=1, help="positive class: level >= threshold")

    p.add_argument("--layers", type=str, default="3,7,11,15,19,23,27,31")
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 0.0, 1.0, 2.0])
    p.add_argument("--normalize-vector", action="store_true")

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--n-responses", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--torch-dtype", type=str, default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=12)

    p.add_argument("--er-model-path", type=Path, default=Path("output/reddit_ER.pth"))
    p.add_argument("--ip-model-path", type=Path, default=Path("output/reddit_IP.pth"))
    p.add_argument("--ex-model-path", type=Path, default=Path("output/reddit_EX.pth"))
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--skip-classification", action="store_true")
    p.add_argument("--reuse-generated", action="store_true")

    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex13_filtered_ip"))
    p.add_argument("--run-name", type=str, default="ex13_llama31_filtered_ip_n200")
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def resolve_dataset_path(persona_project: Path, epitome_project: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    cand1 = (persona_project / p).resolve()
    if cand1.exists():
        return cand1
    cand2 = (epitome_project / p).resolve()
    return cand2


def parse_layers(layer_text: str) -> List[int]:
    layers = [int(tok.strip()) for tok in (layer_text or "").split(",") if tok.strip()]
    if not layers:
        raise ValueError("--layers must include at least one layer index")
    return sorted(set(layers))


def load_external_eval_seekers(path: Path, eval_size: int) -> pd.DataFrame:
    eval_seekers = pd.read_csv(path, dtype={"source_id": str})
    required = {"source_id", "seeker_post"}
    if not required.issubset(eval_seekers.columns):
        raise ValueError(f"External eval CSV must contain {sorted(required)}: {path}")
    eval_seekers = eval_seekers[["source_id", "seeker_post"]].copy()
    eval_seekers["source_id"] = eval_seekers["source_id"].fillna("").astype(str)
    eval_seekers["seeker_post"] = eval_seekers["seeker_post"].fillna("").astype(str)
    if (eval_seekers["source_id"].str.strip() == "").any():
        raise ValueError(f"External eval CSV contains empty source_id: {path}")
    if (eval_seekers["seeker_post"].str.strip() == "").any():
        raise ValueError(f"External eval CSV contains empty seeker_post: {path}")
    if eval_seekers["source_id"].duplicated().any():
        raise ValueError(f"External eval CSV contains duplicate source_id: {path}")
    if len(eval_seekers) != eval_size:
        raise ValueError(f"External eval CSV must contain {eval_size} rows, got {len(eval_seekers)}: {path}")
    return eval_seekers.reset_index(drop=True)


def get_transformer_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture")


class LayerVectorSteerer(AbstractContextManager):
    """Add coeff * vector to last-token hidden state at one transformer layer."""

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


def extract_last_token_hidden(
    model: AutoModelForCausalLM,
    tokenizer,
    input_device: torch.device,
    seeker_post: str,
    response_text: str,
    max_length: int,
    target_layers: List[int],
) -> Optional[Dict[int, np.ndarray]]:
    prompt_only = TF_PROMPT_TEMPLATE.format(seeker_post=seeker_post, response_text="")
    full_text = TF_PROMPT_TEMPLATE.format(seeker_post=seeker_post, response_text=response_text)

    prompt_ids = tokenizer(prompt_only, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    full_enc = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=max_length, return_tensors="pt")
    seq_len = int(full_enc["input_ids"].shape[1])
    response_start = min(len(prompt_ids), seq_len - 1)
    if seq_len <= 1 or response_start >= seq_len:
        return None

    full_enc = {k: v.to(input_device) for k, v in full_enc.items()}
    with torch.inference_mode():
        outputs = model(**full_enc, output_hidden_states=True, use_cache=False, return_dict=True)

    result: Dict[int, np.ndarray] = {}
    for layer in target_layers:
        hs = outputs.hidden_states[layer + 1]
        resp_h = hs[0, response_start:, :]
        result[layer] = resp_h[-1].float().cpu().numpy().astype(np.float32)
    return result


def prepare_splits(
    raw_df: pd.DataFrame,
    seeker_column: str,
    response_column: str,
    level_column: str,
    id_column: str,
    vector_size_per_class: int,
    eval_size: int,
    sample_seed: int,
    level_threshold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    work = raw_df.copy()
    for col in [seeker_column, response_column, level_column]:
        if col not in work.columns:
            raise ValueError(f"Missing column: {col}")

    work[seeker_column] = work[seeker_column].fillna("").astype(str)
    work[response_column] = work[response_column].fillna("").astype(str)
    work = work[
        (work[seeker_column].str.strip() != "") &
        (work[response_column].str.strip() != "")
    ].copy()
    work = ensure_id_column(work, id_column)

    work["level_int"] = pd.to_numeric(work[level_column], errors="coerce")
    work = work[~work["level_int"].isna()].copy()
    work["level_int"] = work["level_int"].astype(int)

    pos_pool = work[work["level_int"] >= int(level_threshold)].copy()
    neg_pool = work[work["level_int"] == 0].copy()
    if len(pos_pool) < vector_size_per_class:
        raise RuntimeError(
            f"Positive pool too small: need {vector_size_per_class}, got {len(pos_pool)} (level>={level_threshold})"
        )
    if len(neg_pool) < vector_size_per_class:
        raise RuntimeError(f"Negative pool too small: need {vector_size_per_class}, got {len(neg_pool)} (level==0)")

    pos_sample = pos_pool.sample(n=vector_size_per_class, random_state=sample_seed).copy()
    neg_sample = neg_pool.sample(n=vector_size_per_class, random_state=sample_seed + 1).copy()

    def to_records(df: pd.DataFrame, condition: str) -> List[dict]:
        rows: List[dict] = []
        for row in df.itertuples(index=False):
            rows.append({
                "condition": condition,
                "source_id": str(getattr(row, id_column)),
                "seeker_post": str(getattr(row, seeker_column)),
                "response_text": str(getattr(row, response_column)),
                "source_level": int(getattr(row, "level_int")),
            })
        return rows

    vector_df = pd.DataFrame(to_records(pos_sample, "pos") + to_records(neg_sample, "neg"))
    vector_df = vector_df.sample(frac=1.0, random_state=sample_seed + 2).reset_index(drop=True)
    vector_df["vector_index"] = np.arange(len(vector_df), dtype=int)

    used_ids = set(vector_df["source_id"].astype(str).tolist())
    seekers = work[[id_column, seeker_column]].drop_duplicates(id_column).copy()
    seekers[id_column] = seekers[id_column].astype(str)
    seekers = seekers[~seekers[id_column].isin(used_ids)].copy()
    if len(seekers) < eval_size:
        raise RuntimeError(f"Not enough disjoint seekers for eval: need {eval_size}, got {len(seekers)}")

    eval_seekers = seekers.sample(n=eval_size, random_state=sample_seed + 3).reset_index(drop=True)
    eval_seekers = eval_seekers.rename(columns={id_column: "source_id", seeker_column: "seeker_post"})

    return vector_df, eval_seekers


def build_layer_vectors_caa(
    vector_df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer,
    input_device: torch.device,
    layers: List[int],
    max_length: int,
    normalize: bool,
) -> Tuple[Dict[int, np.ndarray], List[dict]]:
    pos_by_layer: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    neg_by_layer: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}

    total = len(vector_df)
    skipped = 0
    for i, row in enumerate(vector_df.itertuples(index=False)):
        hidden = extract_last_token_hidden(
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            seeker_post=str(row.seeker_post),
            response_text=str(row.response_text),
            max_length=max_length,
            target_layers=layers,
        )
        if hidden is None:
            skipped += 1
            continue
        target = pos_by_layer if str(row.condition) == "pos" else neg_by_layer
        for layer in layers:
            target[layer].append(hidden[layer])
        if (i + 1) % 20 == 0:
            print(f"  [vector] extracted {i + 1}/{total}", flush=True)

    layer_vectors: Dict[int, np.ndarray] = {}
    vector_info: List[dict] = []
    for layer in layers:
        pos_list = pos_by_layer[layer]
        neg_list = neg_by_layer[layer]
        if not pos_list or not neg_list:
            raise RuntimeError(f"Empty class activation at layer={layer}: pos={len(pos_list)} neg={len(neg_list)}")

        pos_mean = np.stack(pos_list, axis=0).mean(axis=0).astype(np.float32)
        neg_mean = np.stack(neg_list, axis=0).mean(axis=0).astype(np.float32)
        vec = pos_mean - neg_mean
        raw_norm = float(np.linalg.norm(vec))
        if normalize and raw_norm > 0.0:
            vec = vec / raw_norm
        layer_vectors[layer] = vec.astype(np.float32)
        vector_info.append({
            "layer": int(layer),
            "n_pos": int(len(pos_list)),
            "n_neg": int(len(neg_list)),
            "n_skipped_total": int(skipped),
            "raw_vector_norm": raw_norm,
            "used_vector_norm": float(np.linalg.norm(vec)),
            "normalized": bool(normalize),
        })
    return layer_vectors, vector_info


def summarize_by_layer_alpha(classified_df: pd.DataFrame, baseline_ip: float) -> pd.DataFrame:
    rows: List[dict] = []
    for (layer, alpha), g in classified_df.groupby(["layer", "alpha"], sort=True):
        mean_ip = float(g["IP_label"].mean())
        rows.append({
            "layer": int(layer),
            "alpha": float(alpha),
            "n": int(len(g)),
            "IP_mean": mean_ip,
            "IP_pct_0": float((g["IP_label"] == 0).mean()),
            "IP_pct_1": float((g["IP_label"] == 1).mean()),
            "IP_pct_2": float((g["IP_label"] == 2).mean()),
            "delta_vs_unsteered": mean_ip - baseline_ip,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    alpha0 = out[np.isclose(out["alpha"], 0.0)].set_index("layer")["IP_mean"].to_dict()
    out["delta_vs_alpha0"] = out.apply(
        lambda r: r["IP_mean"] - alpha0.get(int(r["layer"]), float("nan")), axis=1
    )
    return out.sort_values(["layer", "alpha"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    persona_project = args.persona_project.resolve()
    epitome_project = args.epitome_project.resolve()
    dataset_path = resolve_dataset_path(persona_project, epitome_project, args.dataset_path)
    external_eval_path = (
        resolve_path(persona_project, args.eval_seekers_path).resolve()
        if args.eval_seekers_path is not None
        else None
    )
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if external_eval_path is not None and not external_eval_path.exists():
        raise FileNotFoundError(f"eval seekers not found: {external_eval_path}")

    er_model_path = resolve_path(epitome_project, args.er_model_path).resolve()
    ip_model_path = resolve_path(epitome_project, args.ip_model_path).resolve()
    ex_model_path = resolve_path(epitome_project, args.ex_model_path).resolve()

    out_vector_csv = output_dir / f"{args.run_name}_vector_rows.csv"
    out_eval_csv = output_dir / f"{args.run_name}_eval_seekers.csv"
    out_generated_csv = output_dir / f"{args.run_name}_generated.csv"
    out_classified_csv = output_dir / f"{args.run_name}_classified.csv"
    out_summary_csv = output_dir / f"{args.run_name}_summary.csv"
    out_best_csv = output_dir / f"{args.run_name}_best_layer.csv"
    out_vec_npz = output_dir / f"{args.run_name}_layer_vectors.npz"
    out_vec_info_csv = output_dir / f"{args.run_name}_layer_vector_info.csv"
    out_meta = output_dir / f"{args.run_name}_metadata.json"

    raw_df = pd.read_csv(dataset_path)
    vector_df, sampled_eval_seekers = prepare_splits(
        raw_df=raw_df,
        seeker_column=args.seeker_column,
        response_column=args.response_column,
        level_column=args.level_column,
        id_column=args.id_column,
        vector_size_per_class=args.vector_size_per_class,
        eval_size=0 if external_eval_path is not None else args.eval_size,
        sample_seed=args.sample_seed,
        level_threshold=args.level_threshold,
    )
    eval_seekers = (
        load_external_eval_seekers(external_eval_path, args.eval_size)
        if external_eval_path is not None
        else sampled_eval_seekers
    )
    overlap_ids = set(vector_df["source_id"].astype(str)) & set(eval_seekers["source_id"].astype(str))
    if overlap_ids:
        raise RuntimeError(
            f"Vector/eval seeker overlap detected ({len(overlap_ids)} IDs): {sorted(overlap_ids)[:5]}"
        )
    vector_df.to_csv(out_vector_csv, index=False)
    eval_seekers.to_csv(out_eval_csv, index=False)
    print(f"[done] vector rows={len(vector_df)} ({out_vector_csv})", flush=True)
    print(f"[done] eval seekers={len(eval_seekers)} ({out_eval_csv})", flush=True)

    layers = parse_layers(args.layers)

    print(f"[load model] {args.model_name}", flush=True)
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

    print("[phase 1] vector extraction (CAA: mean(pos)-mean(neg))", flush=True)
    layer_vectors, vector_info = build_layer_vectors_caa(
        vector_df=vector_df,
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
        layers=layers,
        max_length=args.max_length,
        normalize=args.normalize_vector,
    )
    vector_stack = np.stack([layer_vectors[l] for l in layers], axis=0).astype(np.float32)
    np.savez_compressed(out_vec_npz, layers=np.array(layers, dtype=np.int32), vectors=vector_stack)
    pd.DataFrame(vector_info).to_csv(out_vec_info_csv, index=False)
    print(f"[done] layer vectors saved: {out_vec_npz}", flush=True)

    if args.reuse_generated and out_generated_csv.exists():
        print(f"[reuse] generated csv: {out_generated_csv}", flush=True)
        generated_df = pd.read_csv(out_generated_csv)
    else:
        prompts = [GEN_PROMPT_TEMPLATE.format(seeker_post=sp) for sp in eval_seekers["seeker_post"].tolist()]
        source_ids = eval_seekers["source_id"].astype(str).tolist()

        total_runs = len(layers) * len(args.alphas)
        print(
            f"[phase 2] generation sweep runs={total_runs} "
            f"(layers={len(layers)} x alphas={len(args.alphas)}) "
            f"seekers={len(prompts)}",
            flush=True,
        )

        records: List[dict] = []
        run_idx = 0
        for layer in layers:
            vec_t = torch.tensor(layer_vectors[layer], dtype=torch.float32)
            for alpha in args.alphas:
                run_idx += 1
                print(f"[progress] run {run_idx}/{total_runs} layer={layer} alpha={alpha:+g}", flush=True)
                set_seed(args.seed)
                with LayerVectorSteerer(model, layer, vec_t, float(alpha)):
                    for start in range(0, len(prompts), args.batch_size):
                        batch_prompts = prompts[start:start + args.batch_size]
                        grouped = generate_n_responses_for_batch(
                            model=model,
                            tokenizer=tokenizer,
                            input_device=input_device,
                            prompt_texts=batch_prompts,
                            n_responses=args.n_responses,
                            max_input_tokens=args.max_length,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                        )
                        for i, responses in enumerate(grouped):
                            idx = start + i
                            for r_rank, text in enumerate(responses):
                                text = str(text)
                                records.append({
                                    "layer": int(layer),
                                    "alpha": float(alpha),
                                    "base_sample_index": int(idx),
                                    "source_id": source_ids[idx],
                                    "response_rank": int(r_rank),
                                    "seeker_post": eval_seekers.iloc[idx]["seeker_post"],
                                    "generated_response": text,
                                    "response_post": text,
                                    "response_token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                                    "model_name": args.model_name,
                                })
        generated_df = pd.DataFrame(records).sort_values(
            ["layer", "alpha", "base_sample_index", "response_rank"]
        ).reset_index(drop=True)
        generated_df.to_csv(out_generated_csv, index=False)
        print(f"[done] generated rows={len(generated_df)} ({out_generated_csv})", flush=True)

    if args.skip_classification:
        print("[skip] classification", flush=True)
        return

    print("[phase 3] EPITOME classification", flush=True)
    classified_df = classify_er_ip_ex(
        df=generated_df,
        epitome_project=epitome_project,
        er_model_path=er_model_path,
        ip_model_path=ip_model_path,
        ex_model_path=ex_model_path,
        batch_size=args.classifier_batch_size,
        max_length=64,
    )
    classified_df.to_csv(out_classified_csv, index=False)
    print(f"[done] classified: {out_classified_csv}", flush=True)

    baseline_ip = float(classified_df[np.isclose(classified_df["alpha"], 0.0)]["IP_label"].mean())
    summary_df = summarize_by_layer_alpha(classified_df, baseline_ip=baseline_ip)
    summary_df.to_csv(out_summary_csv, index=False)

    best_df = (
        summary_df[~np.isclose(summary_df["alpha"], 0.0)]
        .sort_values("delta_vs_alpha0", ascending=False)
        .groupby("alpha", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_df.to_csv(out_best_csv, index=False)

    expected_generated = len(layers) * len(args.alphas) * len(eval_seekers) * int(args.n_responses)
    metadata = {
        "task": "ex13_filtered_ip_layer_sweep",
        "dataset_path": str(dataset_path),
        "model_name": args.model_name,
        "layers": layers,
        "alphas": [float(a) for a in args.alphas],
        "vector_size_per_class": int(args.vector_size_per_class),
        "eval_size": int(args.eval_size),
        "eval_seekers_path": str(external_eval_path) if external_eval_path is not None else None,
        "vector_unique_seekers": int(vector_df["source_id"].astype(str).nunique()),
        "eval_unique_seekers": int(eval_seekers["source_id"].astype(str).nunique()),
        "vector_eval_overlap": int(len(overlap_ids)),
        "baseline_ip_alpha0": baseline_ip,
        "expected_generated": int(expected_generated),
        "actual_generated": int(len(generated_df)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY (IP_mean by layer x alpha)", flush=True)
    print("=" * 60, flush=True)
    print(summary_df.pivot_table(index="layer", columns="alpha", values="IP_mean").to_string(), flush=True)
    print(f"\n[done] summary: {out_summary_csv}", flush=True)
    print(f"[done] best layer: {out_best_csv}", flush=True)
    print(f"[done] metadata: {out_meta}", flush=True)


if __name__ == "__main__":
    main()
