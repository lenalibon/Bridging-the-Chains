import os
import json

def merge_jsonl_files(input_folder, output_file):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # skip empty lines
                            try:
                                json_obj = json.loads(line)  # validate JSON
                                outfile.write(json.dumps(json_obj) + '\n')
                            except json.JSONDecodeError:
                                print(f"Skipping invalid JSON line in {filename}: {line}")

# Example usage
input_folder = './results_copy'  # Change this to your actual folder path
output_file = './b1.jsonl'   # Output file name
merge_jsonl_files(input_folder, output_file)