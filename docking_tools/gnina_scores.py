import csv
import re


TAG_PATTERN = re.compile(r">\s*<([^>]+)>")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_sdf_records(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        content = handle.read()
    records = [rec for rec in content.split("$$$$") if rec.strip()]
    return records


def extract_tag_value(record, tag_name):
    lines = record.splitlines()
    for idx, line in enumerate(lines):
        tag_match = TAG_PATTERN.search(line)
        if not tag_match:
            continue
        if tag_match.group(1).strip() != tag_name:
            continue
        for j in range(idx + 1, len(lines)):
            value_line = lines[j].strip()
            if not value_line:
                continue
            if value_line.startswith(">"):
                break
            number_match = NUMBER_PATTERN.search(value_line)
            if number_match:
                return float(number_match.group(0))
            break
    return None


def extract_gnina_scores(fragment, input_file, output_csv):
    rows = []
    for pose_index, record in enumerate(parse_sdf_records(input_file), start=1):
        row = {
            "fragment": str(fragment),
            "pose": pose_index,
            "minimized_affinity": extract_tag_value(record, "minimizedAffinity"),
            "cnn_score": extract_tag_value(record, "CNNscore"),
            "cnn_affinity": extract_tag_value(record, "CNNaffinity"),
        }
        rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["fragment", "pose", "minimized_affinity", "cnn_score", "cnn_affinity"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


fragment = int(input("fragment number: "))
input_file = input("fragment file: ").strip()
output_csv = input("csv file: ").strip()

scores = extract_gnina_scores(fragment, input_file, output_csv)
print(f"Extracted {len(scores)} GNINA-scored poses.")
