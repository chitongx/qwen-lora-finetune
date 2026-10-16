"""Offline, agent-authored HRP evaluation preparation; no model or network dependencies."""
import argparse
import json
from pathlib import Path


def _validate_cases(cases):
    if not cases:
        raise ValueError("cases must not be empty")
    ids, questions = set(), set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case id must be a nonempty string")
        if case_id in ids:
            raise ValueError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        question = case.get("question")
        if not isinstance(question, str) or not question.strip() or question in questions:
            raise ValueError(f"case {case_id}: question must be nonempty and unique")
        questions.add(question)
        source = case.get("provenance", {})
        if (source.get("author") != "agent" or source.get("status") != "unreviewed"
                or source.get("source_file") != "data/hrp_qa_seed.jsonl"
                or source.get("source_indexing") != "zero_based"
                or source.get("not_for_training") is not True
                or case.get("manual_review_required") is not True):
            raise ValueError(f"case {case_id}: unreviewed agent provenance and manual review are required")
        indices = case.get("source_seed_indices")
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or not 0 <= index < 33 for index in indices)):
            raise ValueError(f"case {case_id}: source_seed_indices must reference the 33 seed rows")
        points = case.get("rubric_points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"case {case_id}: rubric_points must not be empty")
        point_ids = set()
        for point in points:
            if not point.get("id") or point["id"] in point_ids or not point.get("description"):
                raise ValueError(f"case {case_id}: rubric point ids must be unique with descriptions")
            point_ids.add(point["id"])
            aliases = point.get("aliases")
            if (not isinstance(aliases, list) or not aliases
                    or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)):
                raise ValueError(f"case {case_id}: aliases must contain nonempty strings")


def load_cases(path):
    """Load JSONL evaluation cases; source indices are zero-based, never training input."""
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    _validate_cases(cases)
    return cases


def repeated_four_gram_excess_ratio(answer):
    """(Number of 4-gram windows - unique windows) / windows after stripping whitespace."""
    text = "".join(answer.split())
    windows = [text[index:index + 4] for index in range(max(0, len(text) - 3))]
    return (len(windows) - len(set(windows))) / len(windows) if windows else 0.0


def evaluate_cases(cases, generate_fn):
    """Run each question once; generate_fn returns answer and observed token metadata.

    Required generation fields: answer (full str), new_tokens (int), max_new_tokens
    (positive int), at_token_limit (bool, consistent with the observed token counts).
    Literal alias matches cannot check negation, reasoning, ordering or factual truth.
    """
    cases = list(cases)
    _validate_cases(cases)
    rows = []
    for case in cases:
        generated = generate_fn(case["question"])
        if not isinstance(generated, dict) or not isinstance(generated.get("answer"), str):
            raise ValueError(f"case {case['id']}: answer must be full text in a dictionary")
        for field in ("new_tokens", "max_new_tokens"):
            if type(generated.get(field)) is not int:
                raise ValueError(f"case {case['id']}: observed {field} is required as an integer")
        new, limit = generated["new_tokens"], generated["max_new_tokens"]
        if limit <= 0 or not 0 <= new <= limit:
            raise ValueError(f"case {case['id']}: invalid new_tokens/max_new_tokens")
        at_limit = generated.get("at_token_limit")
        if type(at_limit) is not bool or at_limit != (new >= limit):
            raise ValueError(f"case {case['id']}: at_token_limit must agree with observed token counts")
        answer = generated["answer"]
        matches = []
        for point in case["rubric_points"]:
            aliases = [alias for alias in point["aliases"] if alias.casefold() in answer.casefold()]
            matches.append(dict(point, matched=bool(aliases), matched_aliases=aliases))
        rows.append({"id": case["id"], "category": case["category"],
                     "question": case["question"], "answer": answer,
                     "source_seed_indices": case["source_seed_indices"],
                     "provenance": case["provenance"], "rubric_matches": matches,
                     "repeated_four_gram_excess_ratio": repeated_four_gram_excess_ratio(answer),
                     "generation": {"new_tokens": new, "max_new_tokens": limit,
                                    "at_token_limit": at_limit},
                     "truncation_status": "token_limit_reached" if at_limit else "not_indicated",
                     "manual_review_required": True, "manual_review_status": "pending"})
    total = sum(len(row["rubric_matches"]) for row in rows)
    matched = sum(point["matched"] for row in rows for point in row["rubric_matches"])
    return {"schema_version": 2, "prepared_data_unreviewed": True,
            "metric_definitions": {
                "literal_rubric_match_rate": "Matched alias groups / all rubric groups; case-insensitive literal substring proxy, not factual accuracy. Negation, reasoning, ordering and business correctness need manual review.",
                "repeated_four_gram_excess_ratio": "After removing all whitespace, (all overlapping character 4-gram windows - distinct windows) / all windows; zero if fewer than four characters. Repeated terminology can increase this signal without an error.",
                "at_token_limit_cases": "Observed new_tokens equals max_new_tokens; a possible truncation signal, not proof that an answer is incomplete.",
                "manual_review_pending_cases": "All cases are agent-authored, unreviewed derivations of the existing seed knowledge; no business validation or independent external facts are claimed."},
            "summary": {"cases": len(rows), "matched_rubric_points": matched,
                        "total_rubric_points": total, "literal_rubric_match_rate": matched / total,
                        "mean_repeated_four_gram_excess_ratio": sum(row["repeated_four_gram_excess_ratio"] for row in rows) / len(rows),
                        "at_token_limit_cases": sum(row["generation"]["at_token_limit"] for row in rows),
                        "manual_review_pending_cases": len(rows)},
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Check existing answer records offline; never loads a model.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True, help="JSON array or JSONL with id and generation fields")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        protected = Path(__file__).resolve().parent / "results" / "evaluation.json"
        if args.output.resolve() == protected.resolve():
            raise ValueError("results/evaluation.json is reserved for the original experiment")
        cases = load_cases(args.cases)
        text = args.answers.read_text(encoding="utf-8")
        records = ([json.loads(line) for line in text.splitlines() if line.strip()]
                   if args.answers.suffix.lower() == ".jsonl" else json.loads(text))
        if not isinstance(records, list):
            raise ValueError("answers must be a JSON array or JSONL records")
        answers = {}
        for record in records:
            answer_id = record.get("id")
            if not isinstance(answer_id, str) or not answer_id.strip():
                raise ValueError("answer id must be a nonempty string")
            if answer_id in answers:
                raise ValueError(f"duplicate answer id: {answer_id}")
            answers[answer_id] = record
        case_ids = {case["id"] for case in cases}
        if answers.keys() - case_ids:
            raise ValueError(f"unknown answer ids: {sorted(answers.keys() - case_ids)}")
        if case_ids - answers.keys():
            raise ValueError(f"missing answer ids: {sorted(case_ids - answers.keys())}")
        ids_by_question = {case["question"]: case["id"] for case in cases}
        report = evaluate_cases(cases, lambda question: answers[ids_by_question[question]])
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
