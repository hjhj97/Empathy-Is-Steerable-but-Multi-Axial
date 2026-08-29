#!/usr/bin/env python3
"""Ex17: Persona effects on empathy (v2) with layer-15 activation logging.

Design:
1) Load shared eval seekers (default: Ex11 eval seekers, n=200).
2) Generate one response per seeker for each persona.
3) Extract per-response layer-15 activation (response-token pooled hidden state).
4) Classify ER/IP/EX using EPITOME classifiers.
5) Save generated/classified rows and persona-wise summaries.
"""

from __future__ import annotations

import argparse
import json
import random
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
DEFAULT_DATASET = Path("dataset/filtered/emotional-reactions-reddit-filtered.csv")
DEFAULT_EVAL_SEEKERS = Path("outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_eval_seekers.csv")
DEFAULT_USER_TEMPLATE = (
    "You are a {persona}.\n"
    "Seeker post:\n"
    "{seeker_post}\n\n"
    "Response:"
)
DEFAULT_PERSONAS = (
    "person,"
    "empathetic person,cynical person,"
    "white person,black person,"
    "male,female,"
    "psychotherapist,engineer,"
    "Republican supporter,Democratic supporter"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex17 persona effects on empathy vector (v2)")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--epitome-project", type=Path, default=EPITOME_PROJECT_DEFAULT)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--eval-seekers-path", type=Path, default=DEFAULT_EVAL_SEEKERS)
    p.add_argument("--seeker-column", type=str, default="seeker_post")
    p.add_argument("--id-column", type=str, default="source_id")
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--sample-seed", type=int, default=42)

    p.add_argument("--personas", type=str, default=DEFAULT_PERSONAS)
    p.add_argument("--baseline-persona", type=str, default="person")
    p.add_argument("--system-prompt", type=str, default="")
    p.add_argument("--user-template", type=str, default=DEFAULT_USER_TEMPLATE)

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--max-input-tokens", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--torch-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device-map", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=12)

    p.add_argument("--activation-layer", type=int, default=15)
    p.add_argument("--activation-pool", type=str, default="mean", choices=["mean", "last"])
    p.add_argument("--save-activation-dtype", type=str, default="float16", choices=["float16", "float32"])

    p.add_argument("--er-vector-path", type=Path, default=Path("outputs/ex11_filtered_er/ex11_filtered_er_llama31_n200_layer_vectors.npz"))
    p.add_argument("--ex-vector-path", type=Path, default=Path("outputs/ex12_filtered_ex/ex12_filtered_ex_llama31_n200_layer_vectors.npz"))
    p.add_argument("--ip-vector-path", type=Path, default=Path("outputs/ex13_filtered_ip/ex13_filtered_ip_llama31_n200_layer_vectors.npz"))
    p.add_argument("--skip-projection", action="store_true")

    p.add_argument("--er-model-path", type=Path, default=Path("output/reddit_ER.pth"))
    p.add_argument("--ip-model-path", type=Path, default=Path("output/reddit_IP.pth"))
    p.add_argument("--ex-model-path", type=Path, default=Path("output/reddit_EX.pth"))
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--classifier-max-length", type=int, default=64)

    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex17_persona_effects_v2"))
    p.add_argument("--run-name", type=str, default="ex17_persona_effects_v2_llama31_n200")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def resolve_dataset_path(persona_project: Path, epitome_project: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    cand1 = (persona_project / p).resolve()
    if cand1.exists():
        return cand1
    return (epitome_project / p).resolve()


def parse_personas(raw: str) -> List[str]:
    vals = [x.strip() for x in (raw or "").split(",") if x.strip()]
    out: List[str] = []
    for v in vals:
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("No personas provided.")
    return out


def format_chat_prompt(tokenizer, system_content: str, user_content: str) -> str:
    msgs = []
    if system_content.strip():
        msgs.append({"role": "system", "content": system_content})
    msgs.append({"role": "user", "content": user_content})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if system_content.strip():
        return f"System: {system_content}\nUser: {user_content}\nAssistant:"
    return f"User: {user_content}\nAssistant:"


def load_eval_seekers(
    persona_project: Path,
    epitome_project: Path,
    dataset_path: Path,
    eval_seekers_path: Path,
    seeker_column: str,
    id_column: str,
    eval_size: int,
    sample_seed: int,
) -> Tuple[pd.DataFrame, str]:
    path = resolve_path(persona_project, eval_seekers_path).resolve()
    if path.exists():
        df = pd.read_csv(path)
        source_note = f"eval_seekers_csv:{path}"
    else:
        raw = pd.read_csv(dataset_path)
        raw = ensure_id_column(raw, id_column)
        if seeker_column not in raw.columns:
            raise ValueError(f"Missing seeker column: {seeker_column}")
        work = raw[[id_column, seeker_column]].copy()
        work[seeker_column] = work[seeker_column].fillna("").astype(str)
        work[id_column] = work[id_column].astype(str)
        work = work[work[seeker_column].str.strip() != ""].drop_duplicates(id_column).reset_index(drop=True)
        if len(work) < eval_size:
            raise RuntimeError(f"Not enough seekers for eval-size={eval_size}: got {len(work)}")
        df = work.sample(n=eval_size, random_state=sample_seed).reset_index(drop=True)
        source_note = f"fallback_dataset_sampling:{dataset_path}"

    if "source_id" in df.columns and "seeker_post" in df.columns:
        out = df[["source_id", "seeker_post"]].copy()
    else:
        if seeker_column not in df.columns:
            raise ValueError(f"Missing seeker column in eval seekers: {seeker_column}")
        src_col = id_column if id_column in df.columns else ("source_id" if "source_id" in df.columns else None)
        if src_col is None:
            df = ensure_id_column(df, id_column)
            src_col = id_column
        out = df[[src_col, seeker_column]].rename(columns={src_col: "source_id", seeker_column: "seeker_post"}).copy()

    out["source_id"] = out["source_id"].astype(str)
    out["seeker_post"] = out["seeker_post"].fillna("").astype(str)
    out = out[out["seeker_post"].str.strip() != ""].drop_duplicates("source_id").reset_index(drop=True)

    if eval_size > 0 and len(out) > eval_size:
        out = out.sample(n=eval_size, random_state=sample_seed).reset_index(drop=True)
    if eval_size > 0 and len(out) < eval_size:
        raise RuntimeError(f"Eval seekers smaller than requested eval-size={eval_size}: got {len(out)}")

    return out, source_note


def extract_response_layer_activation(
    model: AutoModelForCausalLM,
    tokenizer,
    input_device: torch.device,
    prompt_text: str,
    response_text: str,
    max_total_tokens: int,
    layer: int,
    pool: str,
) -> Optional[Tuple[np.ndarray, int, float]]:
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_total_tokens,
    )["input_ids"]

    remaining = max_total_tokens - len(prompt_ids)
    if remaining <= 0:
        return None

    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remaining,
    )["input_ids"]
    if len(response_ids) <= 0:
        return None

    full_ids = prompt_ids + response_ids
    enc = {
        "input_ids": torch.tensor([full_ids], dtype=torch.long, device=input_device),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long, device=input_device),
    }
    response_start = len(prompt_ids)

    with torch.inference_mode():
        outputs = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)

    target = outputs.hidden_states[layer + 1][0, response_start:, :]
    if pool == "mean":
        vec = target.mean(dim=0)
    else:
        vec = target[-1, :]
    l2_mean = float(torch.linalg.vector_norm(target, ord=2, dim=-1).mean().item())
    return vec.float().cpu().numpy().astype(np.float32), int(len(response_ids)), l2_mean


def load_layer_vector(npz_path: Path, layer: int) -> np.ndarray:
    data = np.load(npz_path)
    layers = np.asarray(data["layers"]).astype(int).tolist()
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if layer not in layers:
        raise ValueError(f"Layer {layer} not found in vector file: {npz_path}")
    idx = layers.index(layer)
    return vectors[idx].astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def summarize_by_persona(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for persona, g in df.groupby("used_persona", sort=True):
        rec: Dict[str, object] = {
            "persona": persona,
            "n": int(len(g)),
            "ER_mean": float(g["ER_label"].mean()),
            "IP_mean": float(g["IP_label"].mean()),
            "EX_mean": float(g["EX_label"].mean()),
        }
        for mech in ["ER", "IP", "EX"]:
            counts = g[f"{mech}_label"].value_counts().to_dict()
            total = len(g)
            rec[f"{mech}_pct_0"] = float(counts.get(0, 0) / total * 100.0)
            rec[f"{mech}_pct_1"] = float(counts.get(1, 0) / total * 100.0)
            rec[f"{mech}_pct_2"] = float(counts.get(2, 0) / total * 100.0)

        if "activation_l2_mean" in g.columns:
            rec["act_l2_mean"] = float(g["activation_l2_mean"].mean())
        for col in ["proj_er", "proj_ex", "proj_ip"]:
            if col in g.columns:
                rec[f"{col}_mean"] = float(g[col].mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("persona").reset_index(drop=True)


def summarize_vs_baseline(summary_df: pd.DataFrame, baseline_persona: str) -> pd.DataFrame:
    if baseline_persona not in set(summary_df["persona"].tolist()):
        raise ValueError(f"baseline persona '{baseline_persona}' not found")
    b = summary_df[summary_df["persona"] == baseline_persona].iloc[0]

    out = summary_df.copy()
    for m in ["ER", "IP", "EX"]:
        out[f"delta_{m}_vs_baseline"] = out[f"{m}_mean"].astype(float) - float(b[f"{m}_mean"])
    for col in ["proj_er_mean", "proj_ex_mean", "proj_ip_mean", "act_l2_mean"]:
        if col in out.columns:
            out[f"delta_{col}_vs_baseline"] = out[col].astype(float) - float(b[col])
    out["baseline_persona"] = baseline_persona
    return out.sort_values("delta_ER_vs_baseline", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    if args.eval_size <= 0:
        raise ValueError("--eval-size must be >= 1")

    persona_project = args.persona_project.resolve()
    epitome_project = args.epitome_project.resolve()
    dataset_path = resolve_dataset_path(persona_project, epitome_project, args.dataset_path).resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    er_model_path = resolve_path(epitome_project, args.er_model_path).resolve()
    ip_model_path = resolve_path(epitome_project, args.ip_model_path).resolve()
    ex_model_path = resolve_path(epitome_project, args.ex_model_path).resolve()

    er_vec_path = resolve_path(persona_project, args.er_vector_path).resolve()
    ex_vec_path = resolve_path(persona_project, args.ex_vector_path).resolve()
    ip_vec_path = resolve_path(persona_project, args.ip_vector_path).resolve()

    run_name = args.run_name
    out_seekers = output_dir / f"{run_name}_eval_seekers.csv"
    out_generated = output_dir / f"{run_name}_generated.csv"
    out_activation_rows = output_dir / f"{run_name}_activation_rows.csv"
    out_activation_npy = output_dir / f"{run_name}_layer{args.activation_layer}_activations.npy"
    out_classified = output_dir / f"{run_name}_classified.csv"
    out_summary = output_dir / f"{run_name}_summary_by_persona.csv"
    out_summary_vs = output_dir / f"{run_name}_summary_vs_{args.baseline_persona.replace(' ', '_')}.csv"
    out_metadata = output_dir / f"{run_name}_metadata.json"

    for p in [out_seekers, out_generated, out_activation_rows, out_activation_npy, out_classified, out_summary, out_summary_vs, out_metadata]:
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")

    personas = parse_personas(args.personas)
    if args.baseline_persona not in personas:
        raise ValueError(f"--baseline-persona must be in --personas: {args.baseline_persona}")

    eval_seekers, eval_source_note = load_eval_seekers(
        persona_project=persona_project,
        epitome_project=epitome_project,
        dataset_path=dataset_path,
        eval_seekers_path=args.eval_seekers_path,
        seeker_column=args.seeker_column,
        id_column=args.id_column,
        eval_size=args.eval_size,
        sample_seed=args.sample_seed,
    )
    eval_seekers.to_csv(out_seekers, index=False)
    print(f"[done] eval seekers={len(eval_seekers)} ({out_seekers})", flush=True)

    print(f"[load model] {args.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    input_device = get_input_device(model)

    if not args.skip_projection:
        er_vec = load_layer_vector(er_vec_path, args.activation_layer)
        ex_vec = load_layer_vector(ex_vec_path, args.activation_layer)
        ip_vec = load_layer_vector(ip_vec_path, args.activation_layer)
    else:
        er_vec = ex_vec = ip_vec = None

    prompts: Dict[Tuple[str, int], str] = {}
    generated_rows: List[dict] = []
    total = len(personas) * len(eval_seekers)
    done = 0

    for persona in personas:
        seeker_posts = eval_seekers["seeker_post"].tolist()
        prompt_texts = [
            format_chat_prompt(
                tokenizer,
                args.system_prompt.strip(),
                args.user_template.format(persona=persona, seeker_post=sp).strip(),
            )
            for sp in seeker_posts
        ]
        for i, ptxt in enumerate(prompt_texts):
            prompts[(persona, i)] = ptxt

        print(f"[step] generation persona={persona}", flush=True)
        for start in range(0, len(prompt_texts), args.batch_size):
            batch_prompts = prompt_texts[start:start + args.batch_size]
            grouped = generate_n_responses_for_batch(
                model=model,
                tokenizer=tokenizer,
                input_device=input_device,
                prompt_texts=batch_prompts,
                n_responses=1,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            for bi, responses in enumerate(grouped):
                idx = start + bi
                text = str(responses[0]).strip() if responses else ""
                row = {
                    "source_id": str(eval_seekers.iloc[idx]["source_id"]),
                    "seeker_post": str(eval_seekers.iloc[idx]["seeker_post"]),
                    "used_persona": persona,
                    "generated_response": text,
                    "response_post": text,
                    "model_name": args.model_name,
                    "eval_index": int(idx),
                }
                generated_rows.append(row)
                done += 1
            if done % 200 == 0 or done == total:
                print(f"  [progress] generated {done}/{total}", flush=True)

    generated_df = pd.DataFrame(generated_rows)
    generated_df["response_token_count"] = generated_df["generated_response"].fillna("").astype(str).apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
    generated_df = generated_df.reset_index(drop=True)
    generated_df["activation_index"] = np.arange(len(generated_df), dtype=int)
    generated_df.to_csv(out_generated, index=False)
    print(f"[done] generated rows={len(generated_df)} ({out_generated})", flush=True)

    print(f"[step] layer-{args.activation_layer} activation extraction ({args.activation_pool})", flush=True)
    max_total = args.max_input_tokens + args.max_new_tokens
    act_vectors: List[np.ndarray] = []
    act_rows: List[dict] = []
    skipped = 0
    for r in generated_df.itertuples(index=False):
        prompt_text = prompts[(str(r.used_persona), int(r.eval_index))]
        out = extract_response_layer_activation(
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            prompt_text=prompt_text,
            response_text=str(r.generated_response),
            max_total_tokens=max_total,
            layer=args.activation_layer,
            pool=args.activation_pool,
        )
        if out is None:
            skipped += 1
            continue
        vec, tok_n, l2 = out
        act_vectors.append(vec)
        rec = {
            "activation_index": int(r.activation_index),
            "source_id": str(r.source_id),
            "used_persona": str(r.used_persona),
            "eval_index": int(r.eval_index),
            "response_token_count_tf": int(tok_n),
            "activation_l2_mean": float(l2),
        }
        if er_vec is not None:
            rec["proj_er"] = cosine(vec, er_vec)
            rec["proj_ex"] = cosine(vec, ex_vec)
            rec["proj_ip"] = cosine(vec, ip_vec)
        act_rows.append(rec)
        if len(act_rows) % 200 == 0 or len(act_rows) == len(generated_df):
            print(f"  [progress] activations {len(act_rows)}/{len(generated_df)}", flush=True)

    if not act_rows:
        raise RuntimeError("No valid activations extracted.")

    save_dtype = np.float16 if args.save_activation_dtype == "float16" else np.float32
    act_mat = np.stack(act_vectors, axis=0).astype(save_dtype)
    np.save(out_activation_npy, act_mat)
    act_df = pd.DataFrame(act_rows).sort_values("activation_index").reset_index(drop=True)
    act_df.to_csv(out_activation_rows, index=False)
    print(f"[done] activations rows={len(act_df)} skipped={skipped} ({out_activation_rows})", flush=True)
    print(f"[done] activation matrix: {out_activation_npy} shape={act_mat.shape}", flush=True)

    print("[step] ER/IP/EX classification", flush=True)
    classified_df = classify_er_ip_ex(
        df=generated_df,
        epitome_project=epitome_project,
        er_model_path=er_model_path,
        ip_model_path=ip_model_path,
        ex_model_path=ex_model_path,
        batch_size=args.classifier_batch_size,
        max_length=args.classifier_max_length,
    )

    classified_df = classified_df.merge(
        act_df[["activation_index", "activation_l2_mean"] + ([c for c in ["proj_er", "proj_ex", "proj_ip"] if c in act_df.columns])],
        on="activation_index",
        how="left",
    )
    classified_df.to_csv(out_classified, index=False)
    print(f"[done] classified rows={len(classified_df)} ({out_classified})", flush=True)

    summary_df = summarize_by_persona(classified_df)
    summary_df.to_csv(out_summary, index=False)
    print(f"[done] summary: {out_summary}", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    summary_vs_df = summarize_vs_baseline(summary_df, baseline_persona=args.baseline_persona)
    summary_vs_df.to_csv(out_summary_vs, index=False)
    print(f"[done] baseline summary: {out_summary_vs}", flush=True)
    print(summary_vs_df.to_string(index=False), flush=True)

    metadata = {
        "task": "ex17_persona_effects_v2",
        "dataset_path": str(dataset_path),
        "eval_seekers_source": eval_source_note,
        "n_eval_seekers": int(len(eval_seekers)),
        "n_personas": int(len(personas)),
        "n_generated": int(len(generated_df)),
        "n_activations": int(len(act_df)),
        "n_activation_skipped": int(skipped),
        "activation_layer": int(args.activation_layer),
        "activation_pool": args.activation_pool,
        "save_activation_dtype": args.save_activation_dtype,
        "projection_enabled": bool(not args.skip_projection),
        "outputs": {
            "eval_seekers_csv": str(out_seekers),
            "generated_csv": str(out_generated),
            "activation_rows_csv": str(out_activation_rows),
            "activation_npy": str(out_activation_npy),
            "classified_csv": str(out_classified),
            "summary_csv": str(out_summary),
            "summary_vs_baseline_csv": str(out_summary_vs),
        },
        "personas": personas,
        "baseline_persona": args.baseline_persona,
        "prompt": {
            "system_prompt": args.system_prompt,
            "user_template": args.user_template,
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] metadata: {out_metadata}", flush=True)


if __name__ == "__main__":
    main()
