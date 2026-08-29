#!/usr/bin/env python3
"""Ex20: WikiText-103 perplexity benchmark for steered Llama models.

Evaluates:
- baseline model (no steering)
- ER/EX/IP steering vectors at one target layer
- alpha sweep for each mechanism

Primary output:
- per-condition perplexity on WikiText-103 (or a local text file)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_llama31_multi_response_activation_task1 import (
    get_input_device,
    resolve_torch_dtype,
    set_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Older Linux runtimes can fail when huggingface_hub tries Xet-backed downloads.
# Force regular HTTP download path unless the user explicitly overrides it.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex20: language benchmark (WikiText-103 perplexity)")

    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--torch-dtype", type=str, default="bfloat16",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=12)

    p.add_argument("--target-layer", type=int, default=15)
    p.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 1.0, 2.0])
    p.add_argument("--mechanisms", type=str, default="ER,EX,IP")

    p.add_argument(
        "--er-vector-path",
        type=Path,
        default=Path("outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_layer_vectors.npz"),
    )
    p.add_argument(
        "--ex-vector-path",
        type=Path,
        default=Path("outputs/ex12_filtered_ex/ex12_filtered_ex_llama31_n200_layer_vectors.npz"),
    )
    p.add_argument(
        "--ip-vector-path",
        type=Path,
        default=Path("outputs/ex13_filtered_ip/ex13_filtered_ip_llama31_n200_layer_vectors.npz"),
    )
    p.add_argument("--normalize-loaded-vectors", action="store_true")

    p.add_argument("--dataset-name", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-v1")
    p.add_argument("--dataset-split", type=str, default="test")
    p.add_argument("--local-text-path", type=Path, default=None)
    p.add_argument("--max-text-rows", type=int, default=0,
                   help="Use first N rows from dataset text. 0 means all rows.")

    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max-eval-tokens", type=int, default=0,
                   help="Evaluate only the first N tokens after tokenization. 0 means all.")
    p.add_argument("--add-special-tokens", action="store_true")

    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex20_language_benchmark"))
    p.add_argument("--run-name", type=str, default="ex20_wikitext103_ppl_llama31")

    return p.parse_args()


def resolve_path(base: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (base / maybe_relative)


def parse_mechanisms(text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for tok in (text or "").split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in {"ER", "EX", "IP"}:
            raise ValueError(f"Unknown mechanism: {t}. Expected subset of ER,EX,IP")
        if t not in seen:
            out.append(t)
            seen.add(t)
    if not out:
        raise ValueError("--mechanisms must include at least one of ER,EX,IP")
    return out


def get_transformer_layers(model: AutoModelForCausalLM):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture")


class LayerVectorSteerer(AbstractContextManager):
    """Add coeff * vector to hidden states at one transformer layer."""

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
        # For perplexity evaluation under teacher forcing, apply steering to all
        # token positions so each next-token prediction is affected.
        steered = hidden + (self.coeff * vec)
        return (steered,) + rest if rest is not None else steered

    def __enter__(self):
        self.handle = self.target_layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


def load_text_rows(
    local_text_path: Optional[Path],
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    max_text_rows: int,
) -> Tuple[List[str], Dict[str, str]]:
    if local_text_path is not None:
        if not local_text_path.exists():
            raise FileNotFoundError(f"local text path not found: {local_text_path}")
        rows = local_text_path.read_text(encoding="utf-8").splitlines()
        if max_text_rows > 0:
            rows = rows[:max_text_rows]
        return rows, {"source": "local", "local_text_path": str(local_text_path)}

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "datasets package is required for HF dataset loading. "
            "Install it or pass --local-text-path."
        ) from e

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if "text" not in ds.column_names:
        raise RuntimeError(f"Dataset split has no 'text' column: {ds.column_names}")
    rows = [str(x) if x is not None else "" for x in ds["text"]]
    if max_text_rows > 0:
        rows = rows[:max_text_rows]
    return rows, {"source": "hf", "dataset_name": dataset_name, "dataset_config": dataset_config, "dataset_split": dataset_split}


def select_model_max_length(model, requested_max_length: int) -> int:
    max_candidates: List[int] = []
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ["max_position_embeddings", "n_positions", "seq_length"]:
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                max_candidates.append(val)
    if max_candidates:
        return int(min(requested_max_length, min(max_candidates)))
    return int(requested_max_length)


def load_layer_vector(npz_path: Path, layer: int, normalize: bool) -> Tuple[np.ndarray, float]:
    if not npz_path.exists():
        raise FileNotFoundError(f"vector file not found: {npz_path}")
    with np.load(npz_path) as z:
        if "layers" not in z or "vectors" not in z:
            raise RuntimeError(f"Invalid vector npz format: {npz_path}")
        layers = np.asarray(z["layers"]).astype(int)
        vectors = np.asarray(z["vectors"]).astype(np.float32)

    idx = np.where(layers == int(layer))[0]
    if len(idx) == 0:
        raise RuntimeError(f"Layer {layer} not found in {npz_path}. Available layers: {layers.tolist()}")
    vec = vectors[int(idx[0])].astype(np.float32)
    raw_norm = float(np.linalg.norm(vec))
    if normalize and raw_norm > 0.0:
        vec = vec / raw_norm
    return vec, raw_norm


def evaluate_perplexity(
    model: AutoModelForCausalLM,
    input_device: torch.device,
    input_ids_cpu: torch.Tensor,
    max_length: int,
    stride: int,
    max_eval_tokens: int,
) -> Dict[str, float]:
    if input_ids_cpu.ndim != 2:
        raise ValueError(f"input_ids_cpu must be 2D, got shape={tuple(input_ids_cpu.shape)}")

    eval_ids = input_ids_cpu
    if max_eval_tokens > 0 and eval_ids.shape[1] > max_eval_tokens:
        eval_ids = eval_ids[:, :max_eval_tokens]

    seq_len = int(eval_ids.shape[1])
    if seq_len < 2:
        raise RuntimeError(f"Tokenized sequence too short for perplexity: seq_len={seq_len}")

    nll_sum = 0.0
    token_count = 0
    prev_end = 0
    steps = 0

    for begin in range(0, seq_len, int(stride)):
        end = min(begin + int(max_length), seq_len)
        trg_len = end - prev_end
        if trg_len <= 0:
            break

        chunk = eval_ids[:, begin:end].to(input_device)
        labels = chunk.clone()
        labels[:, :-trg_len] = -100

        with torch.inference_mode():
            out = model(input_ids=chunk, labels=labels, use_cache=False, return_dict=True)

        loss = float(out.loss.detach().float().cpu().item())
        nll_sum += loss * trg_len
        token_count += trg_len
        prev_end = end
        steps += 1
        if end >= seq_len:
            break

    if token_count <= 0:
        raise RuntimeError("No tokens were evaluated for perplexity.")

    mean_nll = nll_sum / float(token_count)
    ppl = float(math.exp(mean_nll)) if mean_nll < 80 else float("inf")
    return {
        "ppl": ppl,
        "mean_nll": float(mean_nll),
        "n_eval_tokens": int(token_count),
        "n_steps": int(steps),
        "seq_len_used": int(seq_len),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    persona_project = args.persona_project.resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    er_vec_path = resolve_path(persona_project, args.er_vector_path).resolve()
    ex_vec_path = resolve_path(persona_project, args.ex_vector_path).resolve()
    ip_vec_path = resolve_path(persona_project, args.ip_vector_path).resolve()
    local_text_path = resolve_path(persona_project, args.local_text_path).resolve() if args.local_text_path else None

    rows, dataset_meta = load_text_rows(
        local_text_path=local_text_path,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        max_text_rows=int(args.max_text_rows),
    )
    if not rows:
        raise RuntimeError("No text rows available for evaluation.")

    eval_text = "\n\n".join(rows)
    if not eval_text.strip():
        raise RuntimeError("Evaluation text is empty after loading rows.")

    print(f"[data] rows={len(rows)} source={dataset_meta.get('source')}", flush=True)
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

    context_len = select_model_max_length(model, int(args.max_length))
    stride = int(args.stride)
    if stride <= 0:
        raise ValueError("--stride must be >= 1")
    if context_len <= 1:
        raise ValueError("--max-length must be >= 2")

    tokenized = tokenizer(
        eval_text,
        return_tensors="pt",
        add_special_tokens=bool(args.add_special_tokens),
    )
    input_ids_cpu = tokenized["input_ids"].cpu()
    print(f"[tokenize] seq_len={input_ids_cpu.shape[1]} context_len={context_len} stride={stride}", flush=True)

    mechanisms = parse_mechanisms(args.mechanisms)
    alpha_values = [float(a) for a in args.alphas]

    vec_file_map: Dict[str, Path] = {
        "ER": er_vec_path,
        "EX": ex_vec_path,
        "IP": ip_vec_path,
    }
    vectors: Dict[str, np.ndarray] = {}
    vector_raw_norms: Dict[str, float] = {}
    for mech in mechanisms:
        vec_np, raw_norm = load_layer_vector(
            npz_path=vec_file_map[mech],
            layer=int(args.target_layer),
            normalize=bool(args.normalize_loaded_vectors),
        )
        vectors[mech] = vec_np
        vector_raw_norms[mech] = raw_norm

    hidden_size = getattr(model.config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        for mech, vec in vectors.items():
            if int(vec.shape[0]) != int(hidden_size):
                raise RuntimeError(
                    f"Vector dim mismatch for {mech}: vec_dim={vec.shape[0]} model_hidden={hidden_size}"
                )

    conditions: List[dict] = [{"condition_id": "baseline", "mechanism": "BASE", "alpha": 0.0}]
    for mech in mechanisms:
        for alpha in alpha_values:
            sign = "+" if alpha > 0 else ""
            conditions.append({
                "condition_id": f"{mech}_a{sign}{alpha:g}",
                "mechanism": mech,
                "alpha": float(alpha),
            })

    rows_out: List[dict] = []
    for i, cond in enumerate(conditions, start=1):
        mech = cond["mechanism"]
        alpha = float(cond["alpha"])
        cond_id = cond["condition_id"]
        print(f"[eval {i}/{len(conditions)}] {cond_id}", flush=True)

        context = nullcontext()
        if mech in vectors and not np.isclose(alpha, 0.0):
            vec_t = torch.tensor(vectors[mech], dtype=torch.float32)
            context = LayerVectorSteerer(model, int(args.target_layer), vec_t, alpha)

        with context:
            metrics = evaluate_perplexity(
                model=model,
                input_device=input_device,
                input_ids_cpu=input_ids_cpu,
                max_length=context_len,
                stride=stride,
                max_eval_tokens=int(args.max_eval_tokens),
            )

        rows_out.append({
            "condition_id": cond_id,
            "mechanism": mech,
            "alpha": alpha,
            "target_layer": int(args.target_layer),
            "ppl": float(metrics["ppl"]),
            "mean_nll": float(metrics["mean_nll"]),
            "n_eval_tokens": int(metrics["n_eval_tokens"]),
            "n_steps": int(metrics["n_steps"]),
            "seq_len_used": int(metrics["seq_len_used"]),
            "vector_path": "" if mech == "BASE" else str(vec_file_map[mech]),
            "vector_raw_norm": np.nan if mech == "BASE" else float(vector_raw_norms[mech]),
            "vector_normed_on_load": bool(args.normalize_loaded_vectors),
            "dataset_source": str(dataset_meta.get("source", "")),
            "dataset_name": str(dataset_meta.get("dataset_name", "")),
            "dataset_config": str(dataset_meta.get("dataset_config", "")),
            "dataset_split": str(dataset_meta.get("dataset_split", "")),
            "local_text_path": str(dataset_meta.get("local_text_path", "")),
        })

    out_csv = output_dir / f"{args.run_name}_results.csv"
    out_json = output_dir / f"{args.run_name}_metadata.json"

    df = pd.DataFrame(rows_out)
    base_row = df[df["condition_id"] == "baseline"]
    if not base_row.empty:
        base_ppl = float(base_row.iloc[0]["ppl"])
        df["delta_ppl_vs_baseline"] = df["ppl"] - base_ppl
        df["ratio_ppl_vs_baseline"] = df["ppl"] / base_ppl if base_ppl != 0 else np.nan
    else:
        df["delta_ppl_vs_baseline"] = np.nan
        df["ratio_ppl_vs_baseline"] = np.nan

    order_map = {"BASE": 0, "ER": 1, "EX": 2, "IP": 3}
    df["__order"] = df["mechanism"].map(order_map).fillna(99).astype(int)
    df = df.sort_values(["__order", "alpha"]).drop(columns=["__order"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "task": "ex20_language_benchmark_ppl",
        "run_name": args.run_name,
        "model_name": args.model_name,
        "target_layer": int(args.target_layer),
        "alphas": alpha_values,
        "mechanisms": mechanisms,
        "max_length": int(args.max_length),
        "stride": int(args.stride),
        "max_eval_tokens": int(args.max_eval_tokens),
        "add_special_tokens": bool(args.add_special_tokens),
        "dataset_meta": dataset_meta,
        "vector_paths": {k: str(v) for k, v in vec_file_map.items()},
        "normalize_loaded_vectors": bool(args.normalize_loaded_vectors),
        "n_conditions": int(len(conditions)),
        "result_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] results: {out_csv}", flush=True)
    print(f"[done] metadata: {out_json}", flush=True)
    print("SUMMARY (ppl)", flush=True)
    print(df[["condition_id", "ppl", "delta_ppl_vs_baseline", "ratio_ppl_vs_baseline"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
