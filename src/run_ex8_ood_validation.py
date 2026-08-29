#!/usr/bin/env python3
"""Ex8: OOD transfer validation on EmpatheticDialogues situations.

Design:
1) Reuse ER steering vector from Ex11 (fixed model-specific vector, target layer=15).
2) Build eval set from EmpatheticDialogues prompt column (situation text).
3) For each alpha, generate responses with activation steering on one target layer.
4) Classify generated responses with EPITOME ER/IP/EX classifiers.
"""

from __future__ import annotations

import argparse
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
DEFAULT_ED_PATH = Path("dataset/empatheticdialogues_test.csv")
DEFAULT_NEGATIVE_EMOTIONS = "afraid,angry,anxious,sad,devastated,lonely,terrified"
GEN_PROMPT_TEMPLATE = "Situation:\n{situation}\n\nResponse:"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex8 OOD validation on EmpatheticDialogues")

    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--epitome-project", type=Path, default=EPITOME_PROJECT_DEFAULT)

    p.add_argument("--ed-source", type=str, default="auto", choices=["auto", "csv", "hf"])
    p.add_argument("--ed-path", type=Path, default=DEFAULT_ED_PATH)
    p.add_argument("--hf-dataset-name", type=str, default="empathetic_dialogues")
    p.add_argument("--hf-split", type=str, default="test")
    p.add_argument("--context-column", type=str, default="context")
    p.add_argument("--prompt-column", type=str, default="prompt")
    p.add_argument("--conv-id-column", type=str, default="conv_id")
    p.add_argument("--negative-emotions", type=str, default=DEFAULT_NEGATIVE_EMOTIONS)
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--sample-seed", type=int, default=42)

    p.add_argument("--vector-path", type=Path, required=True)
    p.add_argument("--target-layer", type=int, default=15)
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
    p.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=12)

    p.add_argument("--er-model-path", type=Path, default=Path("output/reddit_ER.pth"))
    p.add_argument("--ip-model-path", type=Path, default=Path("output/reddit_IP.pth"))
    p.add_argument("--ex-model-path", type=Path, default=Path("output/reddit_EX.pth"))
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--skip-classification", action="store_true")
    p.add_argument("--reuse-generated", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex8_ood_validation"))
    p.add_argument("--run-name", type=str, default="ex8_llama31")
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def parse_csv_list(text: str) -> List[str]:
    return [tok.strip() for tok in str(text).split(",") if tok.strip()]


def get_transformer_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture")


class LayerVectorSteerer(AbstractContextManager):
    """Add coeff * vector to the last-token hidden state at one layer."""

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


def load_ed_dataframe(args: argparse.Namespace, persona_project: Path) -> pd.DataFrame:
    ed_path = resolve_path(persona_project, args.ed_path).resolve()

    if args.ed_source in ("auto", "csv") and ed_path.exists():
        suffix = ed_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(ed_path)
        if suffix == ".jsonl":
            return pd.read_json(ed_path, lines=True)
        if suffix == ".json":
            return pd.read_json(ed_path)
        raise ValueError(f"Unsupported file extension for --ed-path: {ed_path}")

    if args.ed_source == "csv":
        raise FileNotFoundError(f"--ed-source=csv but file not found: {ed_path}")

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Failed to import datasets. Provide a local --ed-path CSV/JSONL or install datasets."
        ) from e

    ds = load_dataset(args.hf_dataset_name, split=args.hf_split)
    return ds.to_pandas()


def prepare_eval_situations(
    raw_df: pd.DataFrame,
    context_column: str,
    prompt_column: str,
    conv_id_column: str,
    negative_emotions: Sequence[str],
    eval_size: int,
    sample_seed: int,
) -> pd.DataFrame:
    if context_column not in raw_df.columns:
        raise ValueError(f"Missing context column: {context_column}")
    if prompt_column not in raw_df.columns:
        raise ValueError(f"Missing prompt column: {prompt_column}")

    work = raw_df.copy()
    work[context_column] = work[context_column].fillna("").astype(str).str.strip()
    work[prompt_column] = work[prompt_column].fillna("").astype(str).str.strip()
    work = work[(work[context_column] != "") & (work[prompt_column] != "")].copy()

    keep_emotions = {e.lower() for e in negative_emotions}
    work = work[work[context_column].str.lower().isin(keep_emotions)].copy()
    if work.empty:
        raise RuntimeError("No rows remain after emotion filtering.")

    if conv_id_column in work.columns:
        work[conv_id_column] = work[conv_id_column].astype(str)
        work = work.drop_duplicates(subset=[conv_id_column], keep="first").copy()
        work = work.rename(columns={conv_id_column: "source_conv_id"})
    else:
        work = work.drop_duplicates(subset=[prompt_column], keep="first").copy()
        work["source_conv_id"] = [f"NO_CONV_{i}" for i in range(len(work))]

    if len(work) < eval_size:
        raise RuntimeError(f"Not enough filtered conversations: need {eval_size}, got {len(work)}")

    sampled = work.sample(n=eval_size, random_state=sample_seed).reset_index(drop=True)
    sampled = sampled.rename(columns={context_column: "emotion_context", prompt_column: "situation"})
    sampled["eval_index"] = np.arange(len(sampled), dtype=int)
    return sampled[["eval_index", "source_conv_id", "emotion_context", "situation"]]


def load_layer_vector(vector_path: Path, target_layer: int, normalize: bool) -> tuple[np.ndarray, dict]:
    if not vector_path.exists():
        raise FileNotFoundError(f"vector file not found: {vector_path}")

    obj = np.load(vector_path)
    if "layers" not in obj or "vectors" not in obj:
        raise ValueError(f"Unsupported vector npz format: {vector_path} (need keys: layers, vectors)")

    layers = obj["layers"].astype(int).tolist()
    vectors = obj["vectors"]
    if vectors.ndim != 2:
        raise ValueError(f"Unsupported vectors shape: {vectors.shape}")

    if int(target_layer) not in layers:
        raise ValueError(f"target_layer={target_layer} not in vector file layers={layers}")
    idx = layers.index(int(target_layer))
    vec = vectors[idx].astype(np.float32)

    raw_norm = float(np.linalg.norm(vec))
    if normalize and raw_norm > 0.0:
        vec = vec / raw_norm
    used_norm = float(np.linalg.norm(vec))
    info = {
        "target_layer": int(target_layer),
        "raw_norm": raw_norm,
        "used_norm": used_norm,
        "normalized": bool(normalize),
        "available_layers": layers,
        "vector_path": str(vector_path),
    }
    return vec, info


def summarize_by_alpha(classified_df: pd.DataFrame, baseline_er: float) -> pd.DataFrame:
    rows: List[dict] = []
    for alpha, g in classified_df.groupby(["alpha"], sort=True):
        mean_er = float(g["ER_label"].mean())
        mean_ip = float(g["IP_label"].mean())
        mean_ex = float(g["EX_label"].mean())
        rows.append({
            "alpha": float(alpha),
            "n": int(len(g)),
            "ER_mean": mean_er,
            "IP_mean": mean_ip,
            "EX_mean": mean_ex,
            "ER_pct_0": float((g["ER_label"] == 0).mean()),
            "ER_pct_1": float((g["ER_label"] == 1).mean()),
            "ER_pct_2": float((g["ER_label"] == 2).mean()),
            "delta_vs_alpha0": mean_er - baseline_er,
        })
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def refuse_overwrite(paths: Dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    persona_project = args.persona_project.resolve()
    epitome_project = args.epitome_project.resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_path = resolve_path(persona_project, args.vector_path).resolve()
    er_model_path = resolve_path(epitome_project, args.er_model_path).resolve()
    ip_model_path = resolve_path(epitome_project, args.ip_model_path).resolve()
    ex_model_path = resolve_path(epitome_project, args.ex_model_path).resolve()

    out_eval_csv = output_dir / f"{args.run_name}_ed_eval_situations.csv"
    out_generated_csv = output_dir / f"{args.run_name}_generated.csv"
    out_classified_csv = output_dir / f"{args.run_name}_classified.csv"
    out_summary_csv = output_dir / f"{args.run_name}_summary.csv"
    out_meta_json = output_dir / f"{args.run_name}_metadata.json"

    refuse_overwrite(
        {
            "eval_csv": out_eval_csv,
            "generated_csv": out_generated_csv,
            "classified_csv": out_classified_csv,
            "summary_csv": out_summary_csv,
            "meta_json": out_meta_json,
        },
        args.overwrite,
    )

    raw_ed = load_ed_dataframe(args=args, persona_project=persona_project)
    negative_emotions = parse_csv_list(args.negative_emotions)
    eval_df = prepare_eval_situations(
        raw_df=raw_ed,
        context_column=args.context_column,
        prompt_column=args.prompt_column,
        conv_id_column=args.conv_id_column,
        negative_emotions=negative_emotions,
        eval_size=args.eval_size,
        sample_seed=args.sample_seed,
    )
    eval_df.to_csv(out_eval_csv, index=False)
    print(f"[done] eval situations={len(eval_df)} ({out_eval_csv})", flush=True)

    vector_np, vector_info = load_layer_vector(
        vector_path=vector_path,
        target_layer=int(args.target_layer),
        normalize=bool(args.normalize_vector),
    )
    vector_t = torch.tensor(vector_np, dtype=torch.float32)
    print(
        "[info] vector loaded "
        f"layer={args.target_layer} raw_norm={vector_info['raw_norm']:.6f} "
        f"used_norm={vector_info['used_norm']:.6f}",
        flush=True,
    )

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

    n_layers = len(get_transformer_layers(model))
    if args.target_layer < 0 or args.target_layer >= n_layers:
        raise ValueError(f"Invalid --target-layer={args.target_layer}; model has {n_layers} layers")

    if args.reuse_generated and out_generated_csv.exists():
        print(f"[reuse] generated csv: {out_generated_csv}", flush=True)
        generated_df = pd.read_csv(out_generated_csv)
    else:
        prompts = [GEN_PROMPT_TEMPLATE.format(situation=s) for s in eval_df["situation"].tolist()]
        conv_ids = eval_df["source_conv_id"].astype(str).tolist()

        print(
            f"[phase] generation runs={len(args.alphas)} "
            f"(layer={args.target_layer}, alphas={args.alphas}, situations={len(prompts)})",
            flush=True,
        )
        records: List[dict] = []
        for run_i, alpha in enumerate(args.alphas, start=1):
            print(f"[progress] run {run_i}/{len(args.alphas)} alpha={alpha:+g}", flush=True)
            set_seed(args.seed)
            with LayerVectorSteerer(model=model, layer_idx=int(args.target_layer), vector=vector_t, coeff=float(alpha)):
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
                        for rank, text in enumerate(responses):
                            text = str(text)
                            records.append(
                                {
                                    "layer": int(args.target_layer),
                                    "alpha": float(alpha),
                                    "eval_index": int(idx),
                                    "source_conv_id": conv_ids[idx],
                                    "response_rank": int(rank),
                                    "emotion_context": eval_df.iloc[idx]["emotion_context"],
                                    "situation": eval_df.iloc[idx]["situation"],
                                    "seeker_post": eval_df.iloc[idx]["situation"],
                                    "generated_response": text,
                                    "response_post": text,
                                    "response_token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                                    "model_name": args.model_name,
                                }
                            )

        generated_df = pd.DataFrame(records).sort_values(
            ["alpha", "eval_index", "response_rank"]
        ).reset_index(drop=True)
        generated_df.to_csv(out_generated_csv, index=False)
        print(f"[done] generated rows={len(generated_df)} ({out_generated_csv})", flush=True)

    if args.skip_classification:
        meta = {
            "task": "ex8_ood_validation",
            "mode": "generation_only",
            "model_name": args.model_name,
            "vector_info": vector_info,
            "eval_size": int(len(eval_df)),
            "alphas": [float(a) for a in args.alphas],
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        out_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[skip] classification", flush=True)
        return

    print("[phase] EPITOME classification", flush=True)
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

    baseline_er = float(classified_df[np.isclose(classified_df["alpha"], 0.0)]["ER_label"].mean())
    summary_df = summarize_by_alpha(classified_df=classified_df, baseline_er=baseline_er)
    summary_df.to_csv(out_summary_csv, index=False)

    expected_generated = len(args.alphas) * len(eval_df) * int(args.n_responses)
    metadata = {
        "task": "ex8_ood_validation",
        "dataset_source": args.ed_source,
        "model_name": args.model_name,
        "vector_info": vector_info,
        "negative_emotions": negative_emotions,
        "eval_size": int(len(eval_df)),
        "target_layer": int(args.target_layer),
        "alphas": [float(a) for a in args.alphas],
        "baseline_er_alpha0": baseline_er,
        "expected_generated": int(expected_generated),
        "actual_generated": int(len(generated_df)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out_meta_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY (ER/IP/EX_mean by alpha)", flush=True)
    print("=" * 60, flush=True)
    print(summary_df[["alpha", "ER_mean", "IP_mean", "EX_mean", "delta_vs_alpha0"]].to_string(index=False), flush=True)
    print(f"\n[done] summary: {out_summary_csv}", flush=True)
    print(f"[done] metadata: {out_meta_json}", flush=True)


if __name__ == "__main__":
    main()
