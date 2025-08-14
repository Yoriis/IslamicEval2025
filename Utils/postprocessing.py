import os
import json
import re

input_dir = "input_directory"
output_dir = "output_directory"
tag = "team_name"

os.makedirs(output_dir, exist_ok=True)

def extract_passage_ids(raw_output, max_passages=20):
    # Clean markdown and whitespace fences
    cleaned_output = str(raw_output).strip().replace("```json", "").replace("```", "").strip()

    # Attempt to parse as JSON list
    try:
        parsed = json.loads(cleaned_output)
        if isinstance(parsed, list):
            passage_ids = parsed
        else:
            passage_ids = []
    except json.JSONDecodeError:
        # Fallback: extract using regex
        passage_ids = re.findall(r"\d+:\d+-\d+", cleaned_output)

    # Sanitize all entries and truncate to top max_passages
    return [
        str(pid).strip().replace("\n", "").strip('"').strip("'")
        for pid in passage_ids if str(pid).strip()
    ][:max_passages]

def is_valid_pid(pid):
    return bool(re.match(r"^\d+:\d+-\d+$", pid))

def format_tsv_line(qid, pid, rank, score, tag):
    if not is_valid_pid(pid):
        return None
    return f"{qid}\tQ0\t{pid}\t{rank}\t{score:.4f}\t{tag}"

for file in os.listdir(input_dir):
    if file.endswith(".json"):
        with open(os.path.join(input_dir, file), "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Failed to parse {file}, skipping...")
                continue

        base_name = os.path.splitext(file)[0]
        out_path = os.path.join(output_dir, f"{base_name}_run.tsv")

        with open(out_path, "w", encoding="utf-8") as out:
            for result in all_results:
                qid = result.get("q_id")
                raw_output = result.get("model_output", "")
                scores_raw = result.get("scores", "[]")

                passage_ids = extract_passage_ids(raw_output)

                try:
                    scores = json.loads(scores_raw)
                    score_map = {str(entry["id"]).strip(): float(entry["score"]) for entry in scores}
                except:
                    score_map = {}

                # Handle special case for no answer
                if not passage_ids or passage_ids == ["-1"]:
                    out.write(f"{qid}\tQ0\t-1\t1\t0.4500\t{tag}\n")
                else:
                    rank = 1
                    for pid in passage_ids:
                        score = score_map.get(pid, 0.4500)
                        line = format_tsv_line(qid, pid, rank, score, tag)
                        if line:
                            out.write(line + "\n")
                            rank += 1
