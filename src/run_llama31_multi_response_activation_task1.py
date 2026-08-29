#!/usr/bin/env python3
"""Task 1: sample EPITOME, generate N responses, and save activations.

Pipeline:
1) Randomly sample seeker posts from EPITOME original dataset.
2) Generate N responses per seeker post under multiple prompt conditions.
3) Extract layer-wise response-token activations (mean pooled hidden states).
4) Save generated rows + activation arrays for Task 2.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SYSTEM_PROMPT = "You are helpful assistant."
PROMPT_CONDITIONS = {
    "empathetic": (
        "The following is a post from someone seeking emotional support.\n"
        "Write a deeply empathetic and emotionally supportive response "
        "in 2-5 sentences. Focus on validating the person's feelings "
        "and showing genuine understanding.\n\n"
        "Seeker post:\n{seeker_post}\n\n"
        "Response:"
    ),
    "neutral": (
        "The following is a post from someone seeking emotional support.\n"
        "Write one response in 2-5 sentences.\n\n"
        "Seeker post:\n{seeker_post}\n\n"
        "Response:"
    ),
    "practical": (
        "The following is a post from someone seeking emotional support.\n"
        "Write a practical and solution-oriented response "
        "in 2-5 sentences. Focus on actionable advice.\n\n"
        "Seeker post:\n{seeker_post}\n\n"
        "Response:"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Task1: multi-response generation + activation extraction")
    parser.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("dataset/emotional-reactions-reddit.csv"),
        help="Original EPITOME CSV path (relative to persona-project unless absolute)",
    )
    parser.add_argument("--seeker-column", type=str, default="seeker_post")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=12)

    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-responses", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--blocked-start-prefixes",
        type=str,
        default="I'm so sorry,I can't,I cannot",
        help="Comma-separated banned response starts (case-insensitive).",
    )
    parser.add_argument(
        "--blocked-start-max-retries",
        type=int,
        default=4,
        help="Max regeneration retries per response when banned starts are detected.",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt-conditions",
        type=str,
        default="empathetic,neutral,practical",
        help="Comma-separated prompt condition names. Available: empathetic, neutral, practical",
    )

    parser.add_argument(
        "--activation-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="dtype for saved activation arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/multi_response_activation"),
        help="Output directory (relative to persona-project unless absolute)",
    )
    parser.add_argument("--run-name", type=str, default="er_n500_k5_t08_p3_v3")
    return parser.parse_args()


def resolve_path(base: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (base / maybe_relative)


def to_jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_prompt_conditions(prompt_conditions_arg: str) -> List[Tuple[str, str]]:
    names = [x.strip() for x in (prompt_conditions_arg or "").split(",") if x.strip()]
    if not names:
        raise ValueError("No prompt conditions selected.")

    missing = [n for n in names if n not in PROMPT_CONDITIONS]
    if missing:
        raise ValueError(
            f"Unknown prompt condition(s): {missing}. "
            f"Available: {sorted(PROMPT_CONDITIONS.keys())}"
        )
    return [(n, PROMPT_CONDITIONS[n]) for n in names]


def parse_blocked_prefixes(prefix_arg: str) -> List[str]:
    raw = [x.strip() for x in (prefix_arg or "").split(",") if x.strip()]
    return [r.lower() for r in raw]


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for placement in model.hf_device_map.values():
            if isinstance(placement, int):
                return torch.device(f"cuda:{placement}")
            if isinstance(placement, str) and placement.startswith("cuda"):
                return torch.device(placement)
    return next(model.parameters()).device


def format_chat_prompt(tokenizer, system_content: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_content}\\nUser: {user_content}\\nAssistant:"


def ensure_id_column(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    out = df.copy()
    if id_column in out.columns:
        return out
    if "sp_id" in out.columns and "rp_id" in out.columns:
        out[id_column] = out["sp_id"].astype(str) + "_" + out["rp_id"].astype(str)
    else:
        out[id_column] = [str(i) for i in range(len(out))]
    return out


def sample_dataset(df: pd.DataFrame, seeker_column: str, n: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work[seeker_column] = work[seeker_column].fillna("").astype(str)
    work = work[work[seeker_column].str.strip() != ""].copy()
    if n > 0 and len(work) > n:
        work = work.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)
    return work


def generate_n_responses_for_batch(
    model: AutoModelForCausalLM,
    tokenizer,
    input_device: torch.device,
    prompt_texts: List[str],
    n_responses: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> List[List[str]]:
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )
    enc = {k: v.to(input_device) for k, v in enc.items()}

    batch_size = int(enc["input_ids"].shape[0])
    input_len = int(enc["input_ids"].shape[1])
    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "num_return_sequences": n_responses,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    with torch.no_grad():
        out = model.generate(**enc, **kwargs)

    if out.ndim == 1:
        out = out.unsqueeze(0)
    new_tokens = out[:, input_len:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    texts = [t.strip() for t in texts]

    if len(texts) != batch_size * n_responses:
        raise RuntimeError(
            f"Unexpected generation size: got {len(texts)}, expected {batch_size * n_responses}"
        )

    grouped: List[List[str]] = []
    for i in range(batch_size):
        s = i * n_responses
        e = (i + 1) * n_responses
        grouped.append(texts[s:e])
    return grouped


def starts_with_blocked_prefix(text: str, blocked_prefixes: List[str]) -> bool:
    if not blocked_prefixes:
        return False
    t = (text or "").strip().lstrip("\"' ").lower()
    return any(t.startswith(p) for p in blocked_prefixes)


def fallback_rewrite_for_blocked_start(text: str) -> str:
    body = (text or "").strip().lstrip("\"' ")
    if not body:
        return "I hear you."
    return f"I hear you. {body}"


def extract_response_activations(
    model: AutoModelForCausalLM,
    tokenizer,
    input_device: torch.device,
    prompt_text: str,
    response_text: str,
    max_length: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_ids = prompt_enc["input_ids"]

    remaining_budget = max_length - len(prompt_ids)
    if remaining_budget <= 0:
        return None

    response_enc = tokenizer(
        response_text,
        add_special_tokens=False,
        truncation=True,
        max_length=remaining_budget,
    )
    response_ids = response_enc["input_ids"]
    response_token_count = len(response_ids)
    if response_token_count <= 0:
        return None

    full_ids = prompt_ids + response_ids
    full_enc = {
        "input_ids": torch.tensor([full_ids], dtype=torch.long, device=input_device),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long, device=input_device),
    }
    response_start = len(prompt_ids)

    with torch.inference_mode():
        outputs = model(
            **full_enc,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    pooled_per_layer: List[np.ndarray] = []
    l2_means: List[float] = []
    for hs in outputs.hidden_states[1:]:
        resp_h = hs[0, response_start:, :]
        pooled = resp_h.mean(dim=0).float().cpu().numpy()
        l2_mean = float(torch.linalg.vector_norm(resp_h, ord=2, dim=-1).mean().item())
        pooled_per_layer.append(pooled)
        l2_means.append(l2_mean)

    return np.stack(pooled_per_layer, axis=0), np.asarray(l2_means, dtype=np.float32), response_token_count


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be >= 1")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition-penalty must be > 0")
    if args.blocked_start_max_retries < 0:
        raise ValueError("--blocked-start-max-retries must be >= 0")

    persona_project = args.persona_project.resolve()
    if not persona_project.exists():
        raise FileNotFoundError(f"persona-project not found: {persona_project}")

    input_path = resolve_path(persona_project, args.input_path).resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    sampled_csv = output_dir / f"{args.run_name}_sampled_{args.sample_size}.csv"
    generated_csv = output_dir / f"{args.run_name}_generated_with_activation_index.csv"
    activation_vec_path = output_dir / f"{args.run_name}_activation_vectors.npy"
    activation_l2_path = output_dir / f"{args.run_name}_activation_l2_mean.npy"
    metadata_path = output_dir / f"{args.run_name}_task1_metadata.json"

    raw_df = pd.read_csv(input_path)
    if args.seeker_column not in raw_df.columns:
        raise ValueError(f"Missing seeker column: {args.seeker_column}")

    raw_df = ensure_id_column(raw_df, args.id_column)
    sampled = sample_dataset(raw_df, args.seeker_column, args.sample_size, args.sample_seed)
    sampled.to_csv(sampled_csv, index=False)
    print(f"[done] sampled {len(sampled)} rows -> {sampled_csv}", flush=True)
    prompt_conditions = parse_prompt_conditions(args.prompt_conditions)
    blocked_prefixes = parse_blocked_prefixes(args.blocked_start_prefixes)
    n_conditions = len(prompt_conditions)

    print("[step] loading generation model", flush=True)
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

    save_dtype = np.float16 if args.activation_dtype == "float16" else np.float32

    skipped_activation = 0
    blocked_start_initial = 0
    blocked_start_regenerated = 0
    blocked_start_fallback_rewrite = 0
    n_layers: Optional[int] = None
    hidden_size: Optional[int] = None

    records: List[Dict[str, object]] = []
    all_layer_vecs: List[np.ndarray] = []
    all_layer_l2: List[np.ndarray] = []

    expected_total = len(sampled) * n_conditions * args.n_responses
    print(
        "[start] generation + activation extraction: "
        f"seekers={len(sampled)}, prompt_conditions={n_conditions}, "
        f"responses_per_condition={args.n_responses}, expected_total={expected_total}",
        flush=True,
    )

    activation_index = 0
    sampled = sampled.reset_index(drop=True)
    total_seekers = len(sampled)
    sample_ptr = 0

    while sample_ptr < total_seekers:
        batch_start = sample_ptr
        batch_end = min(batch_start + args.batch_size, total_seekers)
        batch_df = sampled.iloc[batch_start:batch_end].copy()

        base_indices: List[int] = []
        source_ids: List[str] = []
        seeker_posts: List[str] = []
        for base_idx, b_row in batch_df.iterrows():
            sp = str(b_row[args.seeker_column]) if pd.notna(b_row[args.seeker_column]) else ""
            sid = str(b_row[args.id_column]) if pd.notna(b_row[args.id_column]) else str(base_idx)
            base_indices.append(int(base_idx))
            source_ids.append(sid)
            seeker_posts.append(sp)

        for condition_idx, (prompt_condition, user_template) in enumerate(prompt_conditions):
            prompt_texts: List[str] = [
                format_chat_prompt(
                    tokenizer,
                    args.system_prompt.strip(),
                    user_template.format(seeker_post=sp).strip(),
                )
                for sp in seeker_posts
            ]

            responses_grouped = generate_n_responses_for_batch(
                model=model,
                tokenizer=tokenizer,
                input_device=input_device,
                prompt_texts=prompt_texts,
                n_responses=args.n_responses,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            if blocked_prefixes:
                retry_round = 0
                flat_prompts: List[str] = []
                flat_index_map: List[Tuple[int, int]] = []
                for local_i, responses in enumerate(responses_grouped):
                    for r_idx, resp in enumerate(responses):
                        if starts_with_blocked_prefix(resp, blocked_prefixes):
                            blocked_start_initial += 1
                            flat_prompts.append(prompt_texts[local_i])
                            flat_index_map.append((local_i, r_idx))

                while flat_prompts and retry_round < args.blocked_start_max_retries:
                    retry_round += 1
                    regenerated = generate_n_responses_for_batch(
                        model=model,
                        tokenizer=tokenizer,
                        input_device=input_device,
                        prompt_texts=flat_prompts,
                        n_responses=1,
                        max_input_tokens=args.max_input_tokens,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty,
                    )
                    next_prompts: List[str] = []
                    next_map: List[Tuple[int, int]] = []
                    for idx, grouped_resp in enumerate(regenerated):
                        new_resp = grouped_resp[0] if grouped_resp else ""
                        local_i, r_idx = flat_index_map[idx]
                        responses_grouped[local_i][r_idx] = new_resp
                        if starts_with_blocked_prefix(new_resp, blocked_prefixes):
                            next_prompts.append(flat_prompts[idx])
                            next_map.append((local_i, r_idx))
                        else:
                            blocked_start_regenerated += 1
                    flat_prompts = next_prompts
                    flat_index_map = next_map

                if flat_index_map:
                    for local_i, r_idx in flat_index_map:
                        responses_grouped[local_i][r_idx] = fallback_rewrite_for_blocked_start(
                            responses_grouped[local_i][r_idx]
                        )
                        blocked_start_fallback_rewrite += 1

            for local_i, responses in enumerate(responses_grouped):
                base_idx = base_indices[local_i]
                source_id = source_ids[local_i]
                seeker_post = seeker_posts[local_i]
                prompt_text = prompt_texts[local_i]
                grouped_sample_index = base_idx * n_conditions + condition_idx

                for response_rank, response_text in enumerate(responses):
                    activation = extract_response_activations(
                        model=model,
                        tokenizer=tokenizer,
                        input_device=input_device,
                        prompt_text=prompt_text,
                        response_text=response_text,
                        max_length=args.max_input_tokens + args.max_new_tokens,
                    )
                    if activation is None:
                        skipped_activation += 1
                        continue

                    layer_vecs, layer_l2, response_token_count = activation

                    if n_layers is None:
                        n_layers = int(layer_vecs.shape[0])
                        hidden_size = int(layer_vecs.shape[1])

                    all_layer_vecs.append(layer_vecs.astype(save_dtype, copy=False))
                    all_layer_l2.append(layer_l2.astype(np.float32, copy=False))

                    records.append(
                        {
                            "sample_index": int(grouped_sample_index),
                            "base_sample_index": int(base_idx),
                            "prompt_condition": prompt_condition,
                            "source_id": source_id,
                            "response_rank": int(response_rank),
                            "seeker_post": seeker_post,
                            "generated_response": response_text,
                            "response_post": response_text,
                            "activation_index": int(activation_index),
                            "response_token_count": int(response_token_count),
                            "model_name": args.model_name,
                            "temperature": float(args.temperature),
                            "top_p": float(args.top_p),
                            "repetition_penalty": float(args.repetition_penalty),
                        }
                    )
                    activation_index += 1

        sample_ptr = batch_end
        if (sample_ptr % 10 == 0) or (sample_ptr == total_seekers):
            print(
                f"[progress] processed seekers {sample_ptr}/{total_seekers} | valid_activations={activation_index}",
                flush=True,
            )

    if activation_index == 0:
        raise RuntimeError("No valid activation was extracted. Check generation settings and token limits.")

    activation_vectors = np.stack(all_layer_vecs, axis=0)
    activation_l2 = np.stack(all_layer_l2, axis=0)

    np.save(activation_vec_path, activation_vectors)
    np.save(activation_l2_path, activation_l2)

    generated_df = pd.DataFrame(records)
    generated_df.to_csv(generated_csv, index=False)

    metadata = {
        "task": "task1",
        "input_csv": str(input_path),
        "sampled_csv": str(sampled_csv),
        "generated_csv": str(generated_csv),
        "activation_vectors_npy": str(activation_vec_path),
        "activation_l2_npy": str(activation_l2_path),
        "saved_responses_with_activation": int(activation_index),
        "expected_total_responses": int(expected_total),
        "skipped_no_response_span": int(skipped_activation),
        "blocked_start_prefixes": blocked_prefixes,
        "blocked_start_initial": int(blocked_start_initial),
        "blocked_start_regenerated": int(blocked_start_regenerated),
        "blocked_start_fallback_rewrite": int(blocked_start_fallback_rewrite),
        "prompt_conditions": [name for name, _template in prompt_conditions],
        "activation_dtype": args.activation_dtype,
        "activation_shape": [int(x) for x in activation_vectors.shape],
        "n_layers": int(n_layers if n_layers is not None else -1),
        "hidden_size": int(hidden_size if hidden_size is not None else -1),
        "args": to_jsonable_args(args),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] saved generated rows: {generated_csv}", flush=True)
    print(f"[done] saved activation vectors: {activation_vec_path}", flush=True)
    print(f"[done] saved activation l2 means: {activation_l2_path}", flush=True)
    print(f"[done] metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
