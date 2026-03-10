import re
import csv

def extract_vina_scores(fragment, input_file, output_csv):
    vina_scores = []

    # Regular expression to capture the first float after "VINA RESULT:"
    pattern = re.compile(r"VINA RESULT:\s*([-+]?\d*\.\d+|\d+)")

    with open(input_file, "r") as f:
        for line in f:
            if "VINA RESULT" in line:
                match = pattern.search(line)
                if match:
                    score = float(match.group(1))
                    vina_scores.append(score)

    # Append results to CSV file
    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fragment', 'binding_score'])
        for score in vina_scores:
            writer.writerow([str(fragment), score])

    return vina_scores


# Example usage
fragment = int(input('fragment number: '))
input_file = input('fragment file: ')
output_csv = input('csv file: ')

scores = extract_vina_scores(fragment, input_file, output_csv)
print("Extracted scores:", scores)
