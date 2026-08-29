#!/usr/bin/env python3
"""Ex16: GPT-5-mini Batch judge for Ex11/12/13 layer-15 responses.

Scope:
- Inputs: generated responses from Ex11 (ER), Ex12 (EX), Ex13 (IP)
- Models: llama31, qwen25, mistral7b
- Filter: layer=15 and alpha in {-1, 0, 1, 2}
- Judge: task-specific prompt per mechanism (ER/EX/IP), score 0/1/2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TASK_ORDER = ["ER", "EX", "IP"]
MODEL_ORDER = ["llama31", "qwen25", "mistral7b"]

TASK_TO_INPUT_TEMPLATE = {
    "ER": "outputs/ex11_filtered_er/ex11_filtered_er_{model_tag}_n200_generated.csv",
    "EX": "outputs/ex12_filtered_ex/ex12_filtered_ex_{model_tag}_n200_generated.csv",
    "IP": "outputs/ex13_filtered_ip/ex13_filtered_ip_{model_tag}_n200_generated.csv",
}

PROMPT_ER = """You are an expert annotator for empathic responses.

Assess the candidate response.

ER (Emotional Reactions)
Does the response express or allude to warmth, compassion, concern, or similar feelings of the responder toward the seeker?
- 0: No
- 1: Yes, alluded, but feelings are not explicitly expressed
- 2: Yes, explicit mention of such feelings

Important:
- Do not reward length by itself.
- Judge only the candidate response against the seeker post.

[Seeker Post]
{seeker_post}

[Candidate Response]
{response}

Output exactly:
ER: <0|1|2>"""

PROMPT_IP = """You are an expert annotator for empathic responses.

Assess the candidate response.

IP (Interpretations)
Does the response communicate an understanding of the seeker's experiences and feelings?
- 0: No
- 1: Yes, communicates understanding
- 2: Yes, and includes one or more stronger interpretive behaviors, such as:
  - conjecture/speculation about the seeker's experiences and/or feelings
  - reflecting back on similar experiences of self/others
  - describing similar experiences of self/others
  - paraphrasing the seeker's experiences and/or feelings


Important:
- Do not reward length by itself.
- Judge only the candidate response against the seeker post.

[Seeker Post]
{seeker_post}

[Candidate Response]
{response}

Output exactly:
IP: <0|1|2>"""

PROMPT_EX = """You are an expert annotator for empathic responses.

Assess the candidate response.

EX (Explorations)
Does the response attempt to explore the seeker's experiences and feelings?
- 0: No
- 1: Yes, but exploration is generic
- 2: Yes, and exploration is specific

Important:
- Do not reward length by itself.
- Judge only the candidate response against the seeker post.

[Seeker Post]
{seeker_post}

[Candidate Response]
{response}

Output exactly:
EX: <0|1|2>"""

PROMPT_BY_TASK = {"ER": PROMPT_ER, "IP": PROMPT_IP, "EX": PROMPT_EX}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Ex16 GPT-5-mini layer15 judge batch")
    p.add_argument("--mode", choices=["submit", "wait", "run"], default="run")
    p.add_argument("--persona-project", type=Path, default=PROJECT_ROOT)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/ex16_gpt5mini_layer15_judge"))
    p.add_argument("--run-name", type=str, default="ex16_gpt5mini_layer15_judge")
    p.add_argument("--model", type=str, default="gpt-5-mini")

    p.add_argument("--tasks", type=str, default="ER,EX,IP")
    p.add_argument("--model-tags", type=str, default="llama31,qwen25,mistral7b")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--alphas", type=str, default="-1,0,1,2")
    p.add_argument("--input-spec", type=str, default="")

    p.add_argument("--seeker-column", type=str, default="seeker_post")
    p.add_argument("--response-column", type=str, default="response_post")
    p.add_argument("--alpha-column", type=str, default="alpha")
    p.add_argument("--layer-column", type=str, default="layer")
    p.add_argument("--id-column", type=str, default="source_id")
    p.add_argument("--sample-id-column", type=str, default="base_sample_index")
    p.add_argument("--response-rank-column", type=str, default="response_rank")

    p.add_argument("--batch-id", type=str, default="")
    p.add_argument("--completion-window", type=str, default="24h", choices=["24h"])
    p.add_argument("--poll-interval", type=float, default=30.0)
    p.add_argument("--max-wait-minutes", type=float, default=180.0)
    p.add_argument("--expected-rows", type=int, default=7200)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_path(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p)


def ensure_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError("openai package is not installed. Install with: python -m pip install openai") from e
    return OpenAI()


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def parse_alpha_list(s: str) -> List[float]:
    vals: List[float] = []
    for x in parse_csv_list(s):
        vals.append(float(x))
    if not vals:
        raise ValueError("No alpha values were provided.")
    return vals


def normalize_task(t: str) -> str:
    up = t.strip().upper()
    if up not in {"ER", "EX", "IP"}:
        raise ValueError(f"Invalid task: {t}")
    return up


def parse_input_spec(spec: str) -> Dict[Tuple[str, str], Path]:
    """Format: TASK:MODEL_TAG:relative_or_abs_path, ..."""
    out: Dict[Tuple[str, str], Path] = {}
    chunks = parse_csv_list(spec)
    for ch in chunks:
        toks = [x.strip() for x in ch.split(":", 2)]
        if len(toks) != 3:
            raise ValueError(f"Invalid --input-spec entry: {ch} (expected TASK:MODEL:PATH)")
        task, model_tag, path_s = toks
        task_n = normalize_task(task)
        out[(task_n, model_tag)] = Path(path_s)
    return out


def default_input_path(task: str, model_tag: str) -> Path:
    tmpl = TASK_TO_INPUT_TEMPLATE.get(task)
    if tmpl is None:
        raise ValueError(f"No default template for task={task}")
    return Path(tmpl.format(model_tag=model_tag))


def ensure_required_columns(df: pd.DataFrame, required: List[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def load_input_rows(args: argparse.Namespace, persona_project: Path) -> pd.DataFrame:
    tasks = [normalize_task(x) for x in parse_csv_list(args.tasks)]
    model_tags = parse_csv_list(args.model_tags)
    alphas = parse_alpha_list(args.alphas)
    layer = int(args.layer)
    custom_map = parse_input_spec(args.input_spec)

    rows: List[pd.DataFrame] = []
    for task in tasks:
        for model_tag in model_tags:
            rel_or_abs = custom_map.get((task, model_tag), default_input_path(task, model_tag))
            csv_path = resolve_path(persona_project, rel_or_abs).resolve()
            if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found for task={task}, model={model_tag}: {csv_path}")

            df = pd.read_csv(csv_path)
            ensure_required_columns(
                df,
                [args.seeker_column, args.alpha_column, args.layer_column],
                csv_path,
            )

            response_col = args.response_column
            if response_col not in df.columns:
                if "generated_response" in df.columns:
                    response_col = "generated_response"
                else:
                    raise ValueError(f"Missing response column '{args.response_column}' in {csv_path}")

            work = df.copy()
            work["alpha_norm"] = pd.to_numeric(work[args.alpha_column], errors="coerce")
            work["layer_norm"] = pd.to_numeric(work[args.layer_column], errors="coerce")
            work = work[(work["layer_norm"] == layer) & (work["alpha_norm"].isin(alphas))].copy()

            if work.empty:
                raise ValueError(f"No rows after filter for task={task}, model={model_tag}, path={csv_path}")

            work["task"] = task
            work["model_tag"] = model_tag
            work["input_csv"] = str(csv_path)
            work["response_text"] = work[response_col].astype(str)
            work["seeker_text"] = work[args.seeker_column].astype(str)

            if args.id_column not in work.columns:
                work[args.id_column] = [f"{model_tag}_{task}_{i}" for i in range(len(work))]
            if args.sample_id_column not in work.columns:
                work[args.sample_id_column] = np.nan
            if args.response_rank_column not in work.columns:
                work[args.response_rank_column] = np.nan

            keep_cols = [
                "task",
                "model_tag",
                "input_csv",
                "alpha_norm",
                "layer_norm",
                args.id_column,
                args.sample_id_column,
                args.response_rank_column,
                "seeker_text",
                "response_text",
            ]
            rows.append(work[keep_cols].copy())

    out = pd.concat(rows, axis=0, ignore_index=True)
    out["row_uid"] = np.arange(len(out), dtype=int)
    out = out.sort_values(["task", "model_tag", "alpha_norm", args.sample_id_column, "row_uid"]).reset_index(drop=True)
    return out


def prompt_for_row(task: str, seeker_post: str, response: str) -> str:
    tmpl = PROMPT_BY_TASK[task]
    return tmpl.format(seeker_post=seeker_post, response=response)


def build_batch_jsonl(df: pd.DataFrame, out_jsonl: Path, args: argparse.Namespace) -> None:
    with out_jsonl.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            task = str(row["task"])
            model_tag = str(row["model_tag"])
            alpha = float(row["alpha_norm"])
            layer = int(row["layer_norm"])
            rid = str(row[args.id_column])
            sample_id = str(row[args.sample_id_column])
            rank = str(row[args.response_rank_column])
            row_uid = int(row["row_uid"])
            prompt = prompt_for_row(task, str(row["seeker_text"]), str(row["response_text"]))
            custom_id = f"row_{row_uid}::{task}::{model_tag}::{layer}::{alpha}::{rid}::{sample_id}::{rank}"
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": args.model,
                    "input": prompt,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def submit_batch(client, jsonl_path: Path, metadata: Dict[str, str], completion_window: str) -> Tuple[str, str]:
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata=metadata,
    )
    return str(batch.id), str(uploaded.id)


def wait_batch_and_get_file_ids(
    client, batch_id: str, poll_interval: float, max_wait_minutes: float
) -> Tuple[str, str, str]:
    started = time.time()
    while True:
        b = client.batches.retrieve(batch_id)
        status = str(getattr(b, "status", "unknown"))
        output_file_id = str(getattr(b, "output_file_id", "") or "")
        error_file_id = str(getattr(b, "error_file_id", "") or "")
        print(f"[batch] id={batch_id} status={status}", flush=True)
        if status in {"completed", "failed", "expired", "cancelled"}:
            return status, output_file_id, error_file_id
        if time.time() - started > max_wait_minutes * 60:
            raise TimeoutError(
                f"Batch wait timeout after {max_wait_minutes} minutes. Re-run with --mode wait --batch-id {batch_id}"
            )
        time.sleep(poll_interval)


def extract_output_text_from_response_body(body: object) -> str:
    if not isinstance(body, dict):
        return ""
    out_text = body.get("output_text")
    if isinstance(out_text, str) and out_text.strip():
        return out_text.strip()

    output = body.get("output")
    chunks: List[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                t = c.get("text")
                if isinstance(t, str) and t:
                    chunks.append(t)
    return "\n".join(chunks).strip()


def save_text_from_file_content_obj(content_obj, out_path: Path) -> None:
    if hasattr(content_obj, "write_to_file"):
        content_obj.write_to_file(str(out_path))
        return
    if hasattr(content_obj, "text") and isinstance(content_obj.text, str):
        out_path.write_text(content_obj.text, encoding="utf-8")
        return
    if hasattr(content_obj, "read"):
        raw = content_obj.read()
        if isinstance(raw, bytes):
            out_path.write_bytes(raw)
        else:
            out_path.write_text(str(raw), encoding="utf-8")
        return
    out_path.write_text(str(content_obj), encoding="utf-8")


def parse_task_score(task: str, text: str) -> Optional[int]:
    t = (text or "").strip()
    m = re.search(rf"{task}\s*:\s*([0-2])", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: if model returns only one digit line.
    m2 = re.search(r"\b([0-2])\b", t)
    if m2:
        return int(m2.group(1))
    return None


def parse_batch_results(input_df: pd.DataFrame, output_jsonl: Path, error_jsonl: Optional[Path]) -> Tuple[pd.DataFrame, int, int]:
    id_to_row: Dict[str, int] = {}
    for idx, row in input_df.iterrows():
        custom_id = (
            f"row_{int(row['row_uid'])}::{row['task']}::{row['model_tag']}::{int(row['layer_norm'])}"
            f"::{float(row['alpha_norm'])}::{row['source_id']}::{row['base_sample_index']}::{row['response_rank']}"
        )
        id_to_row[custom_id] = idx

    raw_texts: Dict[int, str] = {}
    request_failed = 0

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            custom_id = rec.get("custom_id", "")
            idx = id_to_row.get(custom_id)
            if idx is None:
                continue
            err = rec.get("error")
            if err:
                request_failed += 1
                raw_texts[idx] = f"REQUEST_ERROR: {err}"
                continue
            resp = rec.get("response", {})
            body = resp.get("body", {}) if isinstance(resp, dict) else {}
            txt = extract_output_text_from_response_body(body)
            raw_texts[idx] = txt if txt else "EMPTY_OUTPUT"

    if error_jsonl is not None and error_jsonl.exists():
        with error_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                custom_id = rec.get("custom_id", "")
                idx = id_to_row.get(custom_id)
                if idx is None:
                    continue
                request_failed += 1
                raw_texts[idx] = f"BATCH_ERROR_FILE: {rec.get('error', rec)}"

    out = input_df.copy()
    labels: List[Optional[int]] = []
    raws: List[str] = []
    parse_failed = 0
    for idx, row in out.iterrows():
        raw = raw_texts.get(idx, "MISSING_OUTPUT")
        score = parse_task_score(str(row["task"]), raw)
        if score is None:
            parse_failed += 1
        labels.append(score)
        raws.append(raw)
    out["judge_label"] = pd.Series(labels, dtype="Int64")
    out["judge_raw"] = raws
    out["judge_parse_failed"] = out["judge_label"].isna().astype(int)
    return out, request_failed, parse_failed


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = df.groupby(["task", "model_tag", "alpha_norm"], sort=True)
    for (task, model_tag, alpha), g in grouped:
        s = g["judge_label"].dropna().astype(int)
        n = int(len(s))
        c = s.value_counts().to_dict()
        rows.append(
            {
                "task": task,
                "model_tag": model_tag,
                "alpha": float(alpha),
                "n": n,
                "judge_mean": float(s.mean()) if n > 0 else np.nan,
                "count_0": int(c.get(0, 0)),
                "count_1": int(c.get(1, 0)),
                "count_2": int(c.get(2, 0)),
                "pct_0": float(c.get(0, 0) / n * 100.0) if n > 0 else np.nan,
                "pct_1": float(c.get(1, 0) / n * 100.0) if n > 0 else np.nan,
                "pct_2": float(c.get(2, 0) / n * 100.0) if n > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values(["task", "model_tag", "alpha"]).reset_index(drop=True)

    out["delta_vs_alpha0"] = np.nan
    for (task, model_tag), g in out.groupby(["task", "model_tag"], sort=True):
        base = g[g["alpha"] == 0.0]
        if base.empty:
            continue
        base_mean = float(base.iloc[0]["judge_mean"])
        idxs = g.index
        out.loc[idxs, "delta_vs_alpha0"] = out.loc[idxs, "judge_mean"] - base_mean
    return out


def make_output_paths(output_dir: Path, run_name: str) -> Dict[str, Path]:
    return {
        "input_rows_csv": output_dir / f"{run_name}_input_rows.csv",
        "requests_jsonl": output_dir / f"{run_name}_batch_requests.jsonl",
        "batch_meta_json": output_dir / f"{run_name}_batch_meta.json",
        "batch_output_jsonl": output_dir / f"{run_name}_batch_output.jsonl",
        "batch_error_jsonl": output_dir / f"{run_name}_batch_error.jsonl",
        "judged_csv": output_dir / f"{run_name}_judged.csv",
        "summary_csv": output_dir / f"{run_name}_summary_by_task_model_alpha.csv",
        "summary_pivot_csv": output_dir / f"{run_name}_summary_pivot.csv",
        "metadata_json": output_dir / f"{run_name}_metadata.json",
    }


def refuse_overwrite(paths: Dict[str, Path], overwrite: bool) -> None:
    if overwrite:
        return
    for p in paths.values():
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {p}. Use --overwrite.")


def main() -> None:
    args = parse_args()
    persona_project = args.persona_project.resolve()
    output_dir = resolve_path(persona_project, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = make_output_paths(output_dir, args.run_name)
    if args.mode in {"submit", "run"}:
        refuse_overwrite(paths, args.overwrite)

    if args.mode == "wait":
        if paths["input_rows_csv"].exists():
            input_df = pd.read_csv(paths["input_rows_csv"])
        else:
            input_df = load_input_rows(args, persona_project)
    else:
        input_df = load_input_rows(args, persona_project)
        input_df.to_csv(paths["input_rows_csv"], index=False)
        print(f"[done] input rows: {len(input_df)} ({paths['input_rows_csv']})", flush=True)

    if args.expected_rows > 0 and len(input_df) != int(args.expected_rows):
        print(
            f"[warn] expected_rows={args.expected_rows}, actual_rows={len(input_df)} "
            "(check model outputs / filters)",
            flush=True,
        )

    client = ensure_openai_client()
    batch_id = args.batch_id.strip()

    if args.mode in {"submit", "run"}:
        build_batch_jsonl(input_df, paths["requests_jsonl"], args)
        print(f"[done] batch requests: {paths['requests_jsonl']}", flush=True)
        metadata = {
            "run_name": args.run_name,
            "task": "ex16_gpt5mini_layer15_judge",
            "model": args.model,
        }
        batch_id, input_file_id = submit_batch(client, paths["requests_jsonl"], metadata, args.completion_window)
        batch_meta = {
            "batch_id": batch_id,
            "input_file_id": input_file_id,
            "created_at": int(time.time()),
            "model": args.model,
            "request_jsonl": str(paths["requests_jsonl"]),
        }
        paths["batch_meta_json"].write_text(json.dumps(batch_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] batch created: {batch_id}", flush=True)
        print(f"[done] batch meta: {paths['batch_meta_json']}", flush=True)

    if args.mode == "submit":
        return

    if args.mode == "wait" and not batch_id:
        if paths["batch_meta_json"].exists():
            meta = json.loads(paths["batch_meta_json"].read_text(encoding="utf-8"))
            batch_id = str(meta.get("batch_id", ""))
        if not batch_id:
            raise ValueError("--mode wait requires --batch-id or existing batch_meta_json")

    status, output_file_id, error_file_id = wait_batch_and_get_file_ids(
        client=client,
        batch_id=batch_id,
        poll_interval=args.poll_interval,
        max_wait_minutes=args.max_wait_minutes,
    )
    if status != "completed":
        raise RuntimeError(f"Batch ended with status={status}. output={output_file_id} error={error_file_id}")
    if not output_file_id:
        raise RuntimeError("Batch completed but output_file_id is empty")

    out_content = client.files.content(output_file_id)
    save_text_from_file_content_obj(out_content, paths["batch_output_jsonl"])
    print(f"[done] batch output: {paths['batch_output_jsonl']}", flush=True)

    err_path: Optional[Path] = None
    if error_file_id:
        err_content = client.files.content(error_file_id)
        save_text_from_file_content_obj(err_content, paths["batch_error_jsonl"])
        err_path = paths["batch_error_jsonl"]
        print(f"[info] batch error file: {paths['batch_error_jsonl']}", flush=True)

    judged_df, request_failed, parse_failed = parse_batch_results(
        input_df=input_df,
        output_jsonl=paths["batch_output_jsonl"],
        error_jsonl=err_path,
    )
    judged_df.to_csv(paths["judged_csv"], index=False)
    print(f"[done] judged: {paths['judged_csv']}", flush=True)

    summary = summarize(judged_df)
    summary.to_csv(paths["summary_csv"], index=False)
    print(f"[done] summary: {paths['summary_csv']}", flush=True)

    pivot = summary.pivot_table(
        index=["task", "model_tag"],
        columns="alpha",
        values="judge_mean",
        aggfunc="first",
    ).reset_index()
    pivot.to_csv(paths["summary_pivot_csv"], index=False)
    print(f"[done] summary pivot: {paths['summary_pivot_csv']}", flush=True)

    metadata = {
        "task": "ex16_gpt5mini_layer15_judge",
        "batch_id": batch_id,
        "rows_input": int(len(input_df)),
        "layer": int(args.layer),
        "alphas": parse_alpha_list(args.alphas),
        "tasks": [normalize_task(x) for x in parse_csv_list(args.tasks)],
        "model_tags": parse_csv_list(args.model_tags),
        "request_failed_count": int(request_failed),
        "parse_failed_count": int(parse_failed),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "output_files": {k: str(v) for k, v in paths.items()},
    }
    paths["metadata_json"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] metadata: {paths['metadata_json']}", flush=True)

    print("\n=== SUMMARY (judge_mean by task x model x alpha) ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
