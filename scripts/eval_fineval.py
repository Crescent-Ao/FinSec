#!/usr/bin/env python3
"""Local, answer-bearing FinEval validation-set A/B/C/D next-token evaluator.

This is a custom zero-shot validation protocol; it does not report official
FinEval test results. Test CSVs without answers are intentionally not loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


PROTOCOL = "fineval-academic-val-zero-shot-option-next-token-v1"
LABELS = "ABCD"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(data_dir: Path, limit: int) -> tuple[list[dict], str]:
    files = sorted(data_dir.glob("*.csv"))
    if len(files) != 34:
        raise ValueError(f"Expected 34 subject CSVs in academic/val, found {len(files)}: {data_dir}")
    file_hashes = [(p.name, sha256_file(p)) for p in files]
    dataset_hash = hashlib.sha256(json.dumps(file_hashes, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    rows: list[dict] = []
    for path in files:
        subject = path.stem.removesuffix("_val")
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"question", *LABELS, "answer"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"Missing required question/A-D/answer columns: {path}")
            subject_rows = []
            for row in reader:
                answer = (row.get("answer") or "").strip().upper()
                if answer not in set(LABELS):
                    continue
                question = (row.get("question") or "").strip()
                if not question or any(not (row.get(k) or "").strip() for k in LABELS):
                    continue
                subject_rows.append({"subject": subject, "id": str(row.get("id", len(subject_rows))),
                                     "question": question, "options": {k: row[k].strip() for k in LABELS},
                                     "answer": answer})
            if limit:
                subject_rows = subject_rows[:limit]
            rows.extend(subject_rows)
    if not rows:
        raise ValueError("No answer-bearing FinEval validation examples found")
    return rows, dataset_hash


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model ID or local model directory")
    p.add_argument("--adapter", help="Optional PEFT adapter directory")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing the 34 academic/val CSVs")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--limit-per-subject", type=int, default=0, help="0 evaluates all rows")
    p.add_argument("--max-length", type=int, default=2048)
    args = p.parse_args(argv)
    if args.limit_per_subject < 0 or args.max_length < 1:
        p.error("--limit-per-subject must be >= 0 and --max-length must be >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        rows, data_hash = load_rows(args.data_dir, args.limit_per_subject)
    except (OSError, ValueError, csv.Error) as e:
        raise SystemExit(f"Data error: {e}")

    # Verify resume identity before loading model weights.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "summary.json"
    metadata = {"protocol": PROTOCOL, "model": args.model,
                "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
                "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
                "model_index_sha256": sha256_file(Path(args.model) / "model.safetensors.index.json"),
                "model_config_sha256": sha256_file(Path(args.model) / "config.json"),
                "data_sha256": data_hash, "data_dir": str(args.data_dir.resolve()),
                "limit_per_subject": args.limit_per_subject, "max_length": args.max_length,
                "subjects": len({r["subject"] for r in rows}), "examples": len(rows)}
    meta_path = args.output_dir / "run.json"
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise SystemExit("Resume refused: protocol/model/adapter/data/config fingerprint differs from run.json")
    else:
        if predictions_path.exists() and predictions_path.stat().st_size:
            raise SystemExit("Resume refused: predictions.jsonl exists without matching run.json")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = {}
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                obj = json.loads(line)
                key = (obj["subject"], obj["id"])
                if key in done or obj.get("fingerprint") != data_hash:
                    raise SystemExit(f"Invalid or duplicate checkpoint record at line {line_num}")
                done[key] = obj
        expected = {(r["subject"], r["id"]) for r in rows}
        if not set(done).issubset(expected):
            raise SystemExit("Resume refused: checkpoint contains rows outside this data/config selection")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForImageTextToText, BitsAndBytesConfig
    except ImportError as e:
        raise SystemExit(f"Missing model dependencies: {e}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; evaluation requires a single CUDA GPU and will not fall back to CPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, enable_thinking=False)
    label_ids = {}
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"Option label {label!r} is not exactly one tokenizer token: {ids}")
        label_ids[label] = ids[0]
    if len(set(label_ids.values())) != 4:
        raise SystemExit(f"A/B/C/D do not map to four distinct token IDs: {label_ids}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=quant, torch_dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True, attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = torch.device("cuda:0")
    with predictions_path.open("a", encoding="utf-8") as out, torch.inference_mode():
        for index, row in enumerate(rows, 1):
            key = (row["subject"], row["id"])
            if key in done:
                continue
            prompt = "请回答以下单项选择题，仅输出正确选项字母A、B、C或D。\n" + row["question"] + "\n" + "\n".join(f"{k}. {row['options'][k]}" for k in LABELS)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            if encoded["input_ids"].shape[-1] > args.max_length:
                raise ValueError(f"Question exceeds max_length: {key}; refusing silent truncation")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded, logits_to_keep=1).logits[0, -1]
            scores = {k: float(logits[token_id].float().item()) for k, token_id in label_ids.items()}
            pred = max(LABELS, key=lambda k: scores[k])
            obj = {"subject": row["subject"], "id": row["id"], "prediction": pred,
                   "correct": pred == row["answer"], "scores": scores, "fingerprint": data_hash}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()
            done[key] = obj
            print(f"[{index}/{len(rows)}] {row['subject']} {row['id']} {pred}", flush=True)

    by_subject = {}
    correct = 0
    for row in rows:
        obj = done[(row["subject"], row["id"])]
        stat = by_subject.setdefault(row["subject"], {"correct": 0, "count": 0})
        stat["correct"] += int(obj["correct"])
        stat["count"] += 1
        correct += int(obj["correct"])
    for stat in by_subject.values():
        stat["accuracy"] = stat["correct"] / stat["count"]
    summary = {**metadata, "protocol_description": "Zero-shot prompt with question and A-D options; choose by raw next-token logits for the four verified single-token labels. No generated reasoning.",
               "result_scope": "academic/val only; custom protocol, not official FinEval test score",
               "micro_accuracy": correct / len(rows),
               "macro_accuracy": sum(s["accuracy"] for s in by_subject.values()) / len(by_subject),
               "by_subject": by_subject}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"micro_accuracy": summary["micro_accuracy"], "macro_accuracy": summary["macro_accuracy"], "examples": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
