"""Validate subagent-authored candidates; arithmetic checks are not expert review."""
import argparse
import ast
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "你是一名金融学习助手。请依据题设准确作答，给出必要的公式、单位或判断依据，解释保持简洁。"


def normalized(text, mask_numbers=False):
    text = unicodedata.normalize("NFKC", text).replace("−", "-").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff.+%\-/]", "", text)
    return re.sub(r"[0-9]+", "#", text) if mask_numbers else text


def arithmetic(expression):
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("invalid expression length")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 100:
        raise ValueError("expression too complex")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = Decimal(ast.get_source_segment(expression, node))
            if not result.is_finite() or abs(result) > Decimal("1e12"):
                raise ValueError("constant out of range")
            return result
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and abs(right) <= 60 and right == right.to_integral_value():
                return left ** int(right)
            if isinstance(node.op, ast.Pow) and right == Decimal("0.5") and left >= 0:
                return left.sqrt()
            if isinstance(node.op, ast.Pow) and right == Decimal("-0.5") and left > 0:
                return 1 / left.sqrt()
        raise ValueError("unsupported expression")

    with localcontext() as context:
        context.prec = 50
        value = evaluate(tree.body)
    if not value.is_finite() or abs(value) > Decimal("1e24"):
        raise ValueError("result out of range")
    return value


class SimilarityIndex:
    def __init__(self):
        self.entries = []
        self.postings = defaultdict(list)

    @staticmethod
    def shingles(text):
        return {text[i:i + 5] for i in range(len(text) - 4)}

    def add(self, key, text, split=None, group=None):
        tokens = self.shingles(text)
        index = len(self.entries)
        self.entries.append((key, text, tokens, split, group))
        for token in tokens:
            self.postings[token].append(index)

    def search(self, text, split=None, group=None, protected=False):
        tokens = self.shingles(text)
        counts = Counter()
        for token in tokens:
            counts.update(self.postings.get(token, []))
        matches = []
        for index, common in counts.items():
            key, other, other_tokens, other_split, other_group = self.entries[index]
            if not protected and (other_split == split or other_group == group):
                continue
            jaccard = common / (len(tokens) + len(other_tokens) - common)
            overlap = common / min(len(tokens), len(other_tokens))
            if min(len(text), len(other)) >= 40 and (jaccard >= 0.85 or (protected and overlap >= 0.94)):
                matches.append({"reference": key, "jaccard": round(jaccard, 4),
                                "overlap": round(overlap, 4)})
        return matches


def load_fineval(directory):
    index = SimilarityIndex()
    rows = 0
    for path in sorted(directory.rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            for number, row in enumerate(reader, 1):
                if len(row) < 6 or row[1].lower() in ("question", "题目"):
                    continue
                key = f"{path.relative_to(directory)}:{number}"
                index.add(key + ":question", normalized(row[1]))
                index.add(key + ":with_options", normalized(" ".join(row[1:6])))
                rows += 1
    if not rows:
        raise ValueError("No FinEval CSV questions loaded")
    return index, rows


def render(record):
    question = "【题设材料】\n" + record["context"] + "\n\n【问题】\n" + record["question"]
    if record["task_type"] == "mcq":
        question += "\n" + "\n".join(f"{key}. {record['options'][key]}" for key in "ABCD")
        question += "\n请选择唯一正确选项，并简要解释。"
    result = dict(record)
    result["messages"] = [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": question},
                          {"role": "assistant", "content": record["answer"]}]
    result["validation"] = {"schema": "passed", "arithmetic": "passed" if record["checks"] else "not_applicable",
                            "expert_review": "pending"}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineval-dir", type=Path)
    args = parser.parse_args()
    manifest_path = ROOT / "data_specs/pilot_v1_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    slots = {row["id"]: row for row in manifest["slots"]}
    output = ROOT / "data/pilot_luna_v1"
    records, errors, seen_ids = [], [], set()
    for path in sorted((output / "parts").glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            record = None
            try:
                record = json.loads(line)
                key = record["id"]
                assert key in slots, "unknown id"
                assert key not in seen_ids, "duplicate id"
                seen_ids.add(key)
                slot = slots[key]
                for field in ("id", "group_id", "split", "domain", "task_type"):
                    assert record[field] == slot[field], f"manifest mismatch: {field}"
                assert record["producer"] == {"requested_model": "gpt-6-luna", "agent": slot["owner"]}
                assert record["source"] == {"kind": "synthetic_authored", "reference": slot["group_id"]}
                assert record["review_status"] == "generated_unreviewed"
                for field in ("context", "question", "answer", "topic", "final_answer"):
                    assert isinstance(record[field], str) and record[field].strip(), f"empty {field}"
                assert len(re.findall(r"[\u4e00-\u9fff]", record["context"])) >= 8, "context is not Chinese"
                answer = record["final_answer"]
                assert answer.replace(",", "") in record["answer"].replace(",", ""), "final answer absent from explanation"
                if record["task_type"] == "mcq":
                    assert set(record["options"]) == set("ABCD")
                    assert all(isinstance(value, str) and value.strip() for value in record["options"].values())
                    assert len({normalized(value) for value in record["options"].values()}) == 4, "duplicate options"
                    assert answer in "ABCD" and len(answer) == 1
                    assert re.match(r"答案[：:]\s*" + answer, record["answer"]), "MCQ answer prefix mismatch"
                else:
                    assert record["options"] == {}
                assert isinstance(record["checks"], list)
                if record["task_type"] == "calculation":
                    assert record["checks"], "calculation lacks checks"
                    assert Decimal(answer) == Decimal(record["checks"][0]["expected"])
                for check in record["checks"]:
                    tolerance = Decimal(check["tolerance"])
                    assert tolerance.is_finite() and 0 <= tolerance <= Decimal("0.5")
                    expected = Decimal(check["expected"])
                    assert expected.is_finite()
                    display_tolerance = Decimal(1).scaleb(expected.as_tuple().exponent) / 2
                    effective_tolerance = min(tolerance, display_tolerance)
                    assert abs(arithmetic(check["expression"]) - expected) <= effective_tolerance, "arithmetic mismatch"
                    assert any(isinstance(n, ast.BinOp) for n in ast.walk(ast.parse(check["expression"]))), "constant-only check"
                records.append(record)
            except Exception as exc:
                errors.append({"file": path.name, "line": number, "id": record.get("id") if isinstance(locals().get("record"), dict) else None,
                               "reason": f"{type(exc).__name__}: {exc}"})
    for key in sorted(set(slots) - seen_ids):
        errors.append({"id": key, "reason": "missing slot"})
    exact = {}
    decisions = {}
    contexts = {}
    near = SimilarityIndex()
    protected, protected_rows = load_fineval(args.fineval_dir) if args.fineval_dir else (None, 0)
    fuzzy_matches, protected_matches = [], []
    for record in records:
        if record["task_type"] == "mcq":
            decision = (record["group_id"], tuple(sorted(normalized(v) for v in record["options"].values())),
                        normalized(record["options"][record["final_answer"]]))
            if decision in decisions:
                errors.append({"id": record["id"], "reason": "same-family repeated MCQ decision and options",
                               "reference": decisions[decision]})
            decisions[decision] = record["id"]
        text = normalized(record["context"] + record["question"] + "".join(record["options"].values()))
        if text in exact:
            errors.append({"id": record["id"], "reason": "exact duplicate input", "reference": exact[text]})
        exact[text] = record["id"]
        context = normalized(record["context"])
        if context in contexts and contexts[context][1] != record["split"]:
            errors.append({"id": record["id"], "reason": "context reused across splits", "reference": contexts[context][0]})
        contexts[context] = (record["id"], record["split"])
        masked = normalized(record["context"] + record["question"] + "".join(record["options"].values()), True)
        for match in near.search(masked, record["split"], record["group_id"]):
            fuzzy_matches.append({"id": record["id"], **match})
        near.add(record["id"], masked, record["split"], record["group_id"])
        if protected:
            for match in protected.search(text, protected=True):
                protected_matches.append({"id": record["id"], **match})
    if fuzzy_matches:
        errors.append({"reason": "cross-split near-duplicate inputs", "count": len(fuzzy_matches)})
    if protected_matches:
        errors.append({"reason": "possible FinEval overlap", "count": len(protected_matches)})
    report = {"dataset": "pilot_luna_v1", "status": "failed" if errors else "candidate_checks_passed",
              "requested_model": "gpt-6-luna", "rows": len(records),
              "families": len({row["group_id"] for row in records}),
              "splits": dict(Counter(row["split"] for row in records)),
              "domains": dict(Counter(row["domain"] for row in records)),
              "task_types": dict(Counter(row["task_type"] for row in records)),
              "mcq_answer_letters": dict(Counter(row["final_answer"] for row in records if row["task_type"] == "mcq")),
              "checks_evaluated": sum(len(row["checks"]) for row in records),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "arithmetic_scope": "expression/result consistency only; question semantics still need review",
              "arithmetic_tolerance_policy": "minimum of declared tolerance and half the last displayed decimal unit",
              "review_status": "generated_unreviewed", "errors": errors,
              "cross_split_near_matches": fuzzy_matches, "fineval_matches": protected_matches,
              "fineval_questions_checked": protected_rows,
              "fineval_check_limit": "lexical matching cannot exclude semantic paraphrases or pretraining contamination"}
    (ROOT / "reports/pilot_luna_v1_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ("errors", "cross_split_near_matches", "fineval_matches")}, ensure_ascii=False, indent=2))
    if errors:
        print(json.dumps(errors[:15], ensure_ascii=False, indent=2))
        raise SystemExit(1)
    for split in ("train", "validation", "privacy_nonmembers"):
        rows = sorted((render(row) for row in records if row["split"] == split), key=lambda row: row["id"])
        (output / f"{split}.candidates.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    main()
