import json
import re
import argparse

def extract_last_number(text):
    matches = re.findall(r'[-+]?\d[\d,]*\.?\d*', text)
    return matches[-1] if matches else None

def clean_true_answer(true_answer):
    return re.sub(r'####\s*([-+]?\d[\d,]*\.?\d*)', r'A: \1', true_answer)

def add_final_answer_field(sample):
    true_answer = sample.get("true_answer", "")
    reasoning = sample.get("reasoning", "")

    sample["true_answer"] = clean_true_answer(true_answer.strip())

    if reasoning.strip().endswith("A:") or re.search(r"A:\s*\d+", reasoning):
        return sample

    final_number_reasoning = extract_last_number(reasoning)
    if final_number_reasoning:
        sample["reasoning"] = reasoning.strip() + f"\nA: {final_number_reasoning}"
    else:
        sample["reasoning"] = reasoning.strip() + "\nA: oo"
    
    return sample

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            sample = json.loads(line)
            updated = add_final_answer_field(sample)
            outfile.write(json.dumps(updated, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace #### with A: and append final answer to reasoning.")
    parser.add_argument("input_file", help="Path to the input .jsonl file")
    parser.add_argument("output_file", help="Path to save the output .jsonl file")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)
    print(f"Processed file saved to {args.output_file}")
