#!/usr/bin/env python3
"""Persona-conditioned response generation + EPITOME ER/IP/EX classification.

RQ: How do empathy labels change by persona prompt?

Pipeline:
1) Randomly sample seeker posts from the original dataset (default: 500).
2) Generate responses with Llama-3.1-8B-Instruct for each persona.
3) Classify generated responses with EPITOME ER/IP/EX checkpoints.
4) Save detailed outputs and persona-wise summary tables.

Example:
python src/run_llama31_persona_epitome_experiment.py \
  --persona-project /path/to/Empathy-Is-Steerable-but-Multi-Axial \
  --epitome-project /path/to/Empathy-Mental-Health
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERSONA_PROJECT_DEFAULT = Path(__file__).resolve().parents[1]
EPITOME_PROJECT_DEFAULT = PERSONA_PROJECT_DEFAULT.parent / "Empathy-Mental-Health"
DEFAULT_SYSTEM_TEMPLATE = "You are a {persona}."
DEFAULT_USER_TEMPLATE = """The following is a post from someone seeking emotional support.
Please write a brief response (2-5 sentences).

Seeker post:
{seeker_post}

Response:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Llama31 Persona Experiment + EPITOME ER/IP/EX")
    parser.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--epitome-project", type=Path, default=EPITOME_PROJECT_DEFAULT)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("dataset/emotional-reactions-reddit.csv"),
        help="Original dataset path (relative to persona-project unless absolute)",
    )
    parser.add_argument("--seeker-column", type=str, default="seeker_post")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=12)
    parser.add_argument(
        "--personas",
        type=str,
        default="Trump supporter,Obama supporter",
        help="Comma-separated personas",
    )
    parser.add_argument("--system-template", type=str, default=DEFAULT_SYSTEM_TEMPLATE)
    parser.add_argument("--user-template", type=str, default=DEFAULT_USER_TEMPLATE)

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Set 0.0 for deterministic decoding; >0 for sampling",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--er-model-path",
        type=Path,
        default=Path("output/reddit_ER.pth"),
        help="ER checkpoint path (relative to epitome-project unless absolute)",
    )
    parser.add_argument(
        "--ip-model-path",
        type=Path,
        default=Path("output/reddit_IP.pth"),
        help="IP checkpoint path (relative to epitome-project unless absolute)",
    )
    parser.add_argument(
        "--ex-model-path",
        type=Path,
        default=Path("output/reddit_EX.pth"),
        help="EX checkpoint path (relative to epitome-project unless absolute)",
    )
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--classifier-max-length", type=int, default=64)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/persona_llama31_epitome"),
        help="Output directory (relative to persona-project unless absolute)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="persona_trump_obama_n500",
        help="Filename prefix for artifacts",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation and reuse existing generated CSV",
    )
    parser.add_argument(
        "--generated-csv",
        type=Path,
        default=None,
        help="When --skip-generation, use this generated CSV path",
    )
    return parser.parse_args()


def resolve_path(base: Path, maybe_relative: Path) -> Path:
    return maybe_relative if maybe_relative.is_absolute() else (base / maybe_relative)


def get_torch_dtype(dtype_name: str):
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


def batched(items: List[str], n: int):
    for i in range(0, len(items), n):
        yield i, items[i : i + n]


def format_chat_prompt(tokenizer, system_content: str, user_content: str) -> str:
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_content}\nUser: {user_content}\nAssistant:"


def sample_dataset(df: pd.DataFrame, seeker_column: str, n: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work[seeker_column] = work[seeker_column].fillna("").astype(str)
    work = work[work[seeker_column].str.strip() != ""].copy()
    if len(work) < n:
        raise ValueError(f"Requested sample_size={n}, but only {len(work)} rows are available.")
    return work.sample(n=n, random_state=seed).reset_index(drop=True)


def generate_for_persona(
    model,
    tokenizer,
    sampled_df: pd.DataFrame,
    seeker_column: str,
    persona: str,
    system_template: str,
    user_template: str,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> pd.DataFrame:
    seeker_posts = sampled_df[seeker_column].astype(str).tolist()
    prompts = [
        format_chat_prompt(
            tokenizer,
            system_template.format(persona=persona).strip(),
            user_template.format(seeker_post=sp).strip(),
        )
        for sp in seeker_posts
    ]

    generated: List[str] = []
    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p if do_sample else None,
        "temperature": temperature if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    input_device = get_input_device(model)

    for start_idx, batch_prompts in batched(prompts, batch_size):
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        enc = {k: v.to(input_device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(**enc, **generation_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        out_new = out[:, prompt_len:]
        texts = tokenizer.batch_decode(out_new, skip_special_tokens=True)
        generated.extend([t.strip() for t in texts])
        print(
            f"[progress] persona={persona} generated {min(start_idx + len(batch_prompts), len(prompts))}/{len(prompts)}",
            flush=True,
        )

    out = sampled_df.copy()
    if "response_post" in out.columns:
        out["reference_response_post"] = out["response_post"]
    out["generated_text"] = generated
    out["response_post"] = out["generated_text"]
    out["used_persona"] = persona
    return out


def encode_text_pairs(
    tokenizer: RobertaTokenizer,
    seeker_posts: List[str],
    response_posts: List[str],
    max_length: int,
):
    sp = tokenizer(
        seeker_posts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    rp = tokenizer(
        response_posts,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return TensorDataset(sp["input_ids"], sp["attention_mask"], rp["input_ids"], rp["attention_mask"])


def load_epitome_model(checkpoint: Path, device: torch.device):
    from models.models import BiEncoderAttentionWithRationaleClassification

    model = BiEncoderAttentionWithRationaleClassification()
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def classify_er_ip_ex(
    df: pd.DataFrame,
    epitome_project: Path,
    er_model_path: Path,
    ip_model_path: Path,
    ex_model_path: Path,
    batch_size: int,
    max_length: int,
) -> pd.DataFrame:
    epitome_src = epitome_project / "src"
    if str(epitome_src) not in sys.path:
        sys.path.insert(0, str(epitome_src))

    seeker_posts = df["seeker_post"].fillna("").astype(str).tolist()
    response_posts = df["response_post"].fillna("").astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] classifier device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    dataset = encode_text_pairs(tokenizer, seeker_posts, response_posts, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
    )

    model_er = load_epitome_model(er_model_path, device)
    model_ip = load_epitome_model(ip_model_path, device)
    model_ex = load_epitome_model(ex_model_path, device)

    er_labels, ip_labels, ex_labels = [], [], []
    er_probs, ip_probs, ex_probs = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            b_sp_ids = batch[0].to(device)
            b_sp_mask = batch[1].to(device)
            b_rp_ids = batch[2].to(device)
            b_rp_mask = batch[3].to(device)

            er_logits, _ = model_er(
                input_ids_SP=b_sp_ids,
                input_ids_RP=b_rp_ids,
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=b_sp_mask,
                attention_mask_RP=b_rp_mask,
            )
            ip_logits, _ = model_ip(
                input_ids_SP=b_sp_ids,
                input_ids_RP=b_rp_ids,
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=b_sp_mask,
                attention_mask_RP=b_rp_mask,
            )
            ex_logits, _ = model_ex(
                input_ids_SP=b_sp_ids,
                input_ids_RP=b_rp_ids,
                token_type_ids_SP=None,
                token_type_ids_RP=None,
                attention_mask_SP=b_sp_mask,
                attention_mask_RP=b_rp_mask,
            )

            er_prob = torch.softmax(er_logits, dim=1).detach().cpu()
            ip_prob = torch.softmax(ip_logits, dim=1).detach().cpu()
            ex_prob = torch.softmax(ex_logits, dim=1).detach().cpu()

            er_labels.extend(torch.argmax(er_prob, dim=1).tolist())
            ip_labels.extend(torch.argmax(ip_prob, dim=1).tolist())
            ex_labels.extend(torch.argmax(ex_prob, dim=1).tolist())

            er_probs.extend(er_prob.tolist())
            ip_probs.extend(ip_prob.tolist())
            ex_probs.extend(ex_prob.tolist())

            if i % 30 == 0:
                print(f"[progress] classifier batches={i}, rows={len(er_labels)}", flush=True)

    out = df.copy()
    out["ER_label"] = er_labels
    out["IP_label"] = ip_labels
    out["EX_label"] = ex_labels

    out["ER_prob_0"] = [p[0] for p in er_probs]
    out["ER_prob_1"] = [p[1] for p in er_probs]
    out["ER_prob_2"] = [p[2] for p in er_probs]

    out["IP_prob_0"] = [p[0] for p in ip_probs]
    out["IP_prob_1"] = [p[1] for p in ip_probs]
    out["IP_prob_2"] = [p[2] for p in ip_probs]

    out["EX_prob_0"] = [p[0] for p in ex_probs]
    out["EX_prob_1"] = [p[1] for p in ex_probs]
    out["EX_prob_2"] = [p[2] for p in ex_probs]
    return out


def summarize_by_persona(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for persona, g in df.groupby("used_persona", sort=True):
        rec: Dict[str, object] = {
            "persona": persona,
            "n": int(len(g)),
            "ER_mean": float(g["ER_label"].mean()),
            "IP_mean": float(g["IP_label"].mean()),
            "EX_mean": float(g["EX_label"].mean()),
        }
        for task in ["ER", "IP", "EX"]:
            counts = g[f"{task}_label"].value_counts().to_dict()
            total = len(g)
            rec[f"{task}_count_0"] = int(counts.get(0, 0))
            rec[f"{task}_count_1"] = int(counts.get(1, 0))
            rec[f"{task}_count_2"] = int(counts.get(2, 0))
            rec[f"{task}_pct_0"] = float(counts.get(0, 0) / total * 100.0)
            rec[f"{task}_pct_1"] = float(counts.get(1, 0) / total * 100.0)
            rec[f"{task}_pct_2"] = float(counts.get(2, 0) / total * 100.0)
        records.append(rec)
    return pd.DataFrame(records).sort_values("persona").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    persona_project = args.persona_project.resolve()
    epitome_project = args.epitome_project.resolve()
    if not persona_project.exists():
        raise FileNotFoundError(f"persona-project not found: {persona_project}")
    if not epitome_project.exists():
        raise FileNotFoundError(f"epitome-project not found: {epitome_project}")

    input_path = resolve_path(persona_project, args.input_path).resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    er_model_path = args.er_model_path if args.er_model_path.is_absolute() else (epitome_project / args.er_model_path)
    ip_model_path = args.ip_model_path if args.ip_model_path.is_absolute() else (epitome_project / args.ip_model_path)
    ex_model_path = args.ex_model_path if args.ex_model_path.is_absolute() else (epitome_project / args.ex_model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input CSV not found: {input_path}")

    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    if not personas:
        raise ValueError("No valid personas provided.")

    generated_csv_default = output_dir / f"{args.run_name}_generated.csv"
    classified_csv = output_dir / f"{args.run_name}_classified_ER_IP_EX.csv"
    sample_csv = output_dir / f"{args.run_name}_sampled_500.csv"
    summary_csv = output_dir / f"{args.run_name}_summary.csv"

    if args.skip_generation:
        if args.generated_csv is None:
            raise ValueError("--generated-csv is required when --skip-generation is set.")
        generated_csv = resolve_path(persona_project, args.generated_csv).resolve()
        if not generated_csv.exists():
            raise FileNotFoundError(f"generated CSV not found: {generated_csv}")
        generated_df = pd.read_csv(generated_csv)
        print(f"[step] skip generation enabled, loaded: {generated_csv}")
    else:
        print("[step] load and sample dataset")
        raw = pd.read_csv(input_path)
        if args.seeker_column not in raw.columns:
            raise ValueError(
                f"Column '{args.seeker_column}' not found. Available: {list(raw.columns)}"
            )
        if args.id_column not in raw.columns:
            if "sp_id" in raw.columns and "rp_id" in raw.columns:
                raw[args.id_column] = raw["sp_id"].astype(str) + "_" + raw["rp_id"].astype(str)
            else:
                raw[args.id_column] = [str(i) for i in range(len(raw))]

        sampled = sample_dataset(raw, args.seeker_column, args.sample_size, args.sample_seed)
        sample_csv = output_dir / f"{args.run_name}_sampled_{args.sample_size}.csv"
        sampled.to_csv(sample_csv, index=False)
        print(f"[done] sampled rows: {len(sampled)} -> {sample_csv}")

        print("[step] load generation model")
        gen_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
            padding_side="left",
        )
        if gen_tokenizer.pad_token_id is None:
            gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id

        gen_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=get_torch_dtype(args.torch_dtype),
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        gen_model.eval()

        all_outputs = []
        for persona in personas:
            print(f"[step] generation persona={persona}")
            per_df = generate_for_persona(
                model=gen_model,
                tokenizer=gen_tokenizer,
                sampled_df=sampled,
                seeker_column=args.seeker_column,
                persona=persona,
                system_template=args.system_template,
                user_template=args.user_template,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            per_df["model_name"] = args.model_name
            all_outputs.append(per_df)

        generated_df = pd.concat(all_outputs, axis=0, ignore_index=True)
        generated_df.to_csv(generated_csv_default, index=False)
        generated_csv = generated_csv_default
        print(f"[done] generated saved: {generated_csv}")

    print("[step] ER/IP/EX classification")
    classified_df = classify_er_ip_ex(
        df=generated_df,
        epitome_project=epitome_project,
        er_model_path=er_model_path.resolve(),
        ip_model_path=ip_model_path.resolve(),
        ex_model_path=ex_model_path.resolve(),
        batch_size=args.classifier_batch_size,
        max_length=args.classifier_max_length,
    )
    classified_df.to_csv(classified_csv, index=False)
    print(f"[done] classified saved: {classified_csv}")

    summary_df = summarize_by_persona(classified_df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[done] summary saved: {summary_csv}")
    print("\n[summary]")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
