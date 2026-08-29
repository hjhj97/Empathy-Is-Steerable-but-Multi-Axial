#!/usr/bin/env python3
"""Ex25: prompt-based mechanism baseline vs activation steering references.

Pipeline:
1) Load shared eval seekers (default: Ex11 eval seekers).
2) Generate responses for prompt-only conditions P0-P3 (no activation hook).
3) Classify responses with EPITOME ER/IP/EX classifiers.
4) Summarize by condition and vs P0 baseline.
5) Compare prompt deltas to Ex11/Ex12/Ex13 layer-15 alpha=+1 references.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex25 prompt mechanism baseline")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--epitome-project", type=Path, default=EPITOME_PROJECT_DEFAULT)
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--eval-seekers-path", type=Path, default=DEFAULT_EVAL_SEEKERS)
    p.add_argument("--seeker-column", type=str, default="seeker_post")
    p.add_argument("--id-column", type=str, default="source_id")
    p.add_argument("--eval-size", type=int, default=200)
    p.add_argument("--sample-seed", type=int, default=42)

    p.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--system-prompt", type=str, default="")
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

    p.add_argument("--p1-instruction", type=str, default="Respond empathetically to the seeker.")
    p.add_argument("--p2-instruction", type=str, default="Acknowledge the seeker's feelings warmly and compassionately.")
    p.add_argument(
        "--p3-instruction",
        type=str,
        default="Reflect the seeker's situation, show understanding, and ask one open-ended follow-up question.",
    )
    p.add_argument("--baseline-condition-id", type=str, default="P0")

    p.add_argument("--steering-layer", type=int, default=15)
    p.add_argument("--steering-alpha", type=float, default=1.0)
    p.add_argument("--ex11-summary-path", type=Path, default=None)
    p.add_argument("--ex12-summary-path", type=Path, default=None)
    p.add_argument("--ex13-summary-path", type=Path, default=None)

    p.add_argument("--er-model-path", type=Path, default=Path("output/reddit_ER.pth"))
    p.add_argument("--ip-model-path", type=Path, default=Path("output/reddit_IP.pth"))
    p.add_argument("--ex-model-path", type=Path, default=Path("output/reddit_EX.pth"))
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--classifier-max-length", type=int, default=64)

    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex25_prompt_mechanism_baseline"))
    p.add_argument("--run-name", type=str, default="ex25_prompt_mechanism_baseline_llama31_n200")
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


def model_key_from_name(model_name: str) -> str:
    m = model_name.lower()
    if "llama-3.1-8b-instruct" in m:
        return "llama31"
    if "qwen2.5-7b-instruct" in m or "qwen2.5" in m:
        return "qwen25"
    if "mistral-7b-instruct-v0.3" in m or "mistral-7b" in m:
        return "mistral7b"
    return re.sub(r"[^a-z0-9]+", "", m)[:24] or "model"


def load_eval_seekers(
    persona_project: Path,
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


def build_prompt_conditions(args: argparse.Namespace) -> List[dict]:
    conds = [
        {"condition_id": "P0", "label": "no_instruction", "instruction": ""},
        {"condition_id": "P1", "label": "generic_empathy", "instruction": args.p1_instruction.strip()},
        {"condition_id": "P2", "label": "er_targeted", "instruction": args.p2_instruction.strip()},
        {"condition_id": "P3", "label": "ip_ex_targeted", "instruction": args.p3_instruction.strip()},
    ]
    return conds


def build_user_prompt(instruction: str, seeker_post: str) -> str:
    if instruction.strip():
        return f"{instruction.strip()}\n\nSeeker post:\n{seeker_post}\n\nResponse:"
    return f"Seeker post:\n{seeker_post}\n\nResponse:"


def summarize_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition_id", "condition_label", "instruction"]
    rows: List[dict] = []
    for key, g in df.groupby(group_cols, sort=False):
        cid, clabel, inst = key
        rec: Dict[str, object] = {
            "condition_id": cid,
            "condition_label": clabel,
            "instruction": inst,
            "n": int(len(g)),
            "ER_mean": float(g["ER_label"].mean()),
            "IP_mean": float(g["IP_label"].mean()),
            "EX_mean": float(g["EX_label"].mean()),
            "response_token_count_mean": float(g["response_token_count"].mean()),
        }
        for mech in ["ER", "IP", "EX"]:
            counts = g[f"{mech}_label"].value_counts().to_dict()
            total = len(g)
            rec[f"{mech}_pct_0"] = float(counts.get(0, 0) / total * 100.0)
            rec[f"{mech}_pct_1"] = float(counts.get(1, 0) / total * 100.0)
            rec[f"{mech}_pct_2"] = float(counts.get(2, 0) / total * 100.0)
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["condition_id"]).reset_index(drop=True)


def summarize_vs_baseline(summary_df: pd.DataFrame, baseline_condition_id: str) -> pd.DataFrame:
    if baseline_condition_id not in set(summary_df["condition_id"].tolist()):
        raise ValueError(f"baseline condition_id '{baseline_condition_id}' not found in summary")
    b = summary_df[summary_df["condition_id"] == baseline_condition_id].iloc[0]
    out = summary_df.copy()
    for m in ["ER", "IP", "EX"]:
        out[f"delta_{m}_vs_baseline"] = out[f"{m}_mean"].astype(float) - float(b[f"{m}_mean"])
    out["baseline_condition_id"] = baseline_condition_id
    return out.sort_values(["condition_id"]).reset_index(drop=True)


def resolve_summary_path(
    persona_project: Path,
    maybe_path: Optional[Path],
    exp_dir: str,
    pattern: str,
) -> Path:
    if maybe_path is not None:
        path = resolve_path(persona_project, maybe_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"summary not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(
                f"summary path must be a CSV file, got directory: {path}. "
                "Pass --ex11-summary-path/--ex12-summary-path/--ex13-summary-path as files or omit them."
            )
        return path
    base = persona_project / exp_dir
    matches = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No summary matches for pattern: {base / pattern}")
    return matches[0]


def load_steering_reference_row(summary_path: Path, layer: int, alpha: float, metric: str) -> dict:
    df = pd.read_csv(summary_path)
    if "layer" not in df.columns or "alpha" not in df.columns:
        raise ValueError(f"Invalid summary format (missing layer/alpha): {summary_path}")
    sel = df[(df["layer"].astype(int) == int(layer)) & (np.isclose(df["alpha"].astype(float), float(alpha)))]
    if sel.empty:
        raise ValueError(f"No row at layer={layer}, alpha={alpha} in {summary_path}")
    row = sel.iloc[0]

    mean_col = f"{metric}_mean"
    if mean_col not in sel.columns:
        raise ValueError(f"Missing mean column '{mean_col}' in {summary_path}")

    if "delta_vs_alpha0" in sel.columns:
        delta_col = "delta_vs_alpha0"
    elif "delta_vs_unsteered" in sel.columns:
        delta_col = "delta_vs_unsteered"
    else:
        raise ValueError(f"Missing delta column in {summary_path}")

    return {
        "summary_path": str(summary_path),
        "layer": int(layer),
        "alpha": float(alpha),
        "metric": metric,
        "mean": float(row[mean_col]),
        "delta_vs_alpha0": float(row[delta_col]),
    }


def build_pairwise_prompt_vs_activation(
    summary_vs_df: pd.DataFrame,
    steering_refs: Dict[str, dict],
) -> pd.DataFrame:
    rows: List[dict] = []
    for r in summary_vs_df.itertuples(index=False):
        for metric, source in [("ER", "ex11_er"), ("EX", "ex12_ex"), ("IP", "ex13_ip")]:
            ref = steering_refs[source]
            p_mean = float(getattr(r, f"{metric}_mean"))
            p_delta = float(getattr(r, f"delta_{metric}_vs_baseline"))
            s_mean = float(ref["mean"])
            s_delta = float(ref["delta_vs_alpha0"])
            rows.append(
                {
                    "condition_id": r.condition_id,
                    "condition_label": r.condition_label,
                    "metric": metric,
                    "prompt_mean": p_mean,
                    "prompt_delta_vs_P0": p_delta,
                    "steering_source": source,
                    "steering_layer": int(ref["layer"]),
                    "steering_alpha": float(ref["alpha"]),
                    "steering_mean": s_mean,
                    "steering_delta_vs_alpha0": s_delta,
                    "delta_prompt_minus_steering_mean": p_mean - s_mean,
                    "delta_prompt_minus_steering_delta": p_delta - s_delta,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["condition_id", "metric"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    persona_project = args.persona_project.resolve()
    epitome_project = args.epitome_project.resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(persona_project, epitome_project, args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    run_name = args.run_name
    out_seekers = output_dir / f"{run_name}_eval_seekers.csv"
    out_generated = output_dir / f"{run_name}_generated.csv"
    out_classified = output_dir / f"{run_name}_classified.csv"
    out_summary = output_dir / f"{run_name}_summary_by_condition.csv"
    out_summary_vs = output_dir / f"{run_name}_summary_vs_baseline.csv"
    out_pairwise = output_dir / f"{run_name}_pairwise_prompt_vs_activation.csv"
    out_meta = output_dir / f"{run_name}_metadata.json"

    outputs = [out_seekers, out_generated, out_classified, out_summary, out_summary_vs, out_pairwise, out_meta]
    for p in outputs:
        if p.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")

    eval_seekers, eval_source_note = load_eval_seekers(
        persona_project=persona_project,
        dataset_path=dataset_path,
        eval_seekers_path=args.eval_seekers_path,
        seeker_column=args.seeker_column,
        id_column=args.id_column,
        eval_size=args.eval_size,
        sample_seed=args.sample_seed,
    )
    eval_seekers.to_csv(out_seekers, index=False)
    print(f"[done] eval seekers={len(eval_seekers)} ({out_seekers})", flush=True)

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

    conditions = build_prompt_conditions(args)
    generated_rows: List[dict] = []
    total = len(conditions) * len(eval_seekers)
    done = 0

    for cond in conditions:
        cid = str(cond["condition_id"])
        clabel = str(cond["label"])
        instruction = str(cond["instruction"])
        print(f"[step] generation condition={cid} label={clabel}", flush=True)

        prompts = [
            format_chat_prompt(
                tokenizer,
                args.system_prompt.strip(),
                build_user_prompt(instruction, str(sp)),
            )
            for sp in eval_seekers["seeker_post"].tolist()
        ]

        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start:start + args.batch_size]
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
                generated_rows.append(
                    {
                        "source_id": str(eval_seekers.iloc[idx]["source_id"]),
                        "seeker_post": str(eval_seekers.iloc[idx]["seeker_post"]),
                        "condition_id": cid,
                        "condition_label": clabel,
                        "instruction": instruction,
                        "generated_response": text,
                        "response_post": text,
                        "model_name": args.model_name,
                        "eval_index": int(idx),
                    }
                )
                done += 1
            if done % 200 == 0 or done == total:
                print(f"  [progress] generated {done}/{total}", flush=True)

    generated_df = pd.DataFrame(generated_rows).reset_index(drop=True)
    generated_df["response_token_count"] = generated_df["generated_response"].fillna("").astype(str).apply(
        lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
    generated_df.to_csv(out_generated, index=False)
    print(f"[done] generated rows={len(generated_df)} ({out_generated})", flush=True)

    er_model_path = resolve_path(epitome_project, args.er_model_path).resolve()
    ip_model_path = resolve_path(epitome_project, args.ip_model_path).resolve()
    ex_model_path = resolve_path(epitome_project, args.ex_model_path).resolve()

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
    classified_df.to_csv(out_classified, index=False)
    print(f"[done] classified rows={len(classified_df)} ({out_classified})", flush=True)

    summary_df = summarize_by_condition(classified_df)
    summary_df.to_csv(out_summary, index=False)
    print(f"[done] summary by condition: {out_summary}", flush=True)
    print(summary_df.to_string(index=False), flush=True)

    summary_vs_df = summarize_vs_baseline(summary_df, baseline_condition_id=args.baseline_condition_id)
    summary_vs_df.to_csv(out_summary_vs, index=False)
    print(f"[done] summary vs baseline: {out_summary_vs}", flush=True)

    mkey = model_key_from_name(args.model_name)
    ex11_summary = resolve_summary_path(
        persona_project,
        args.ex11_summary_path,
        exp_dir="outputs/ex11_filtered_er",
        pattern=f"ex11_filtered_er_{mkey}_*_summary.csv",
    )
    ex12_summary = resolve_summary_path(
        persona_project,
        args.ex12_summary_path,
        exp_dir="outputs/ex12_filtered_ex",
        pattern=f"ex12_filtered_ex_{mkey}_*_summary.csv",
    )
    ex13_summary = resolve_summary_path(
        persona_project,
        args.ex13_summary_path,
        exp_dir="outputs/ex13_filtered_ip",
        pattern=f"ex13_filtered_ip_{mkey}_*_summary.csv",
    )

    steering_refs = {
        "ex11_er": load_steering_reference_row(ex11_summary, args.steering_layer, args.steering_alpha, "ER"),
        "ex12_ex": load_steering_reference_row(ex12_summary, args.steering_layer, args.steering_alpha, "EX"),
        "ex13_ip": load_steering_reference_row(ex13_summary, args.steering_layer, args.steering_alpha, "IP"),
    }

    pairwise_df = build_pairwise_prompt_vs_activation(summary_vs_df, steering_refs)
    pairwise_df.to_csv(out_pairwise, index=False)
    print(f"[done] pairwise prompt vs activation: {out_pairwise}", flush=True)

    metadata = {
        "task": "ex25_prompt_mechanism_baseline",
        "model_name": args.model_name,
        "model_key": mkey,
        "dataset_path": str(dataset_path),
        "eval_seekers_source": eval_source_note,
        "n_eval_seekers": int(len(eval_seekers)),
        "n_conditions": int(len(conditions)),
        "baseline_condition_id": args.baseline_condition_id,
        "steering_layer": int(args.steering_layer),
        "steering_alpha": float(args.steering_alpha),
        "steering_references": steering_refs,
        "prompt_conditions": conditions,
        "outputs": {
            "eval_seekers_csv": str(out_seekers),
            "generated_csv": str(out_generated),
            "classified_csv": str(out_classified),
            "summary_by_condition_csv": str(out_summary),
            "summary_vs_baseline_csv": str(out_summary_vs),
            "pairwise_prompt_vs_activation_csv": str(out_pairwise),
        },
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done] metadata: {out_meta}", flush=True)


if __name__ == "__main__":
    main()
