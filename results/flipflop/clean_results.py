import json
import os
from collections import Counter

# Function to remove the last '.' from the 'answer' field in all JSONL files in a folder
def process_jsonlines_folder(folder_path):
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as infile:
                lines = infile.readlines()

            with open(file_path, 'w') as outfile:
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        # Modify the 'answer' field by removing the last '.' if it exists
                        if "A" in data['answer']:
                            data["answer"] = data["answer"].replace("A", "0")
                        elif "B" in data['answer']:
                            data['answer'] = data["answer"].replace("B", "1")
                        #if '.' in data['answer']:
                            #data['answer'] = data['answer'].replace('.', '')
                        # Write the updated data back to the file
                        outfile.write(json.dumps(data) + '\n')


def count_last_valid_tokens(folder_path):
    token_counter = Counter()
    total_samples = 0

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip():
                        data = json.loads(line)
                        # Count the 'last_valid_token' field
                        if 'last_valid_token' in data:
                            token_counter[data['last_valid_token']] += 1
                            total_samples += 1

    # Calculate proportions
    proportions = {key: count / total_samples for key, count in token_counter.items()}

    return token_counter, proportions


def count_last_valid_tokens_in_files(file_paths):
    token_counter = Counter()
    total_samples = 0

    # Iterate through the provided list of file paths
    for file_path in file_paths:
        if os.path.isfile(file_path) and file_path.endswith('.jsonl'):
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip():
                        data = json.loads(line)
                        # Count the 'last_valid_token' field
                        if 'last_valid_token' in data:
                            token_counter[data['last_valid_token']] += 1
                            total_samples += 1

    # Calculate proportions
    proportions = {key: count / total_samples for key, count in token_counter.items()}

    return token_counter, proportions


# List of file paths
input_folder = ("llama3.1_70B-instruct/distance-qa/s5")
process_jsonlines_folder(input_folder)

print("Processed ", input_folder)
