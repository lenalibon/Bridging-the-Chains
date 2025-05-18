import json

input_file = './roscoe/roscoe_data/generated/gsm8k.json'
output_file = './example.json'

def transform_entry(entry):
    return {
        "question": entry["premise"],
        "ground_truth": entry["hypothesis"].replace("IGNORE THIS. Ground truth here for reference. ", "", 1),
        "answer": entry["gpt-3"]
    }

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if line.strip():  # Skip empty lines
            original = json.loads(line)
            transformed = transform_entry(original)
            json.dump(transformed, outfile, ensure_ascii=False)
            outfile.write('\n')