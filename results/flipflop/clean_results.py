import json
import os

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
                        if 'answer' in data and data['answer'].endswith('.'):
                            data['answer'] = data['answer'][:-1]
                        # Write the updated data back to the file
                        outfile.write(json.dumps(data) + '\n')

# Input folder path
input_folder = 'llama3.1_70B/sparse/s1'  # Replace with your input folder path

# Call the function to process the folder
process_jsonlines_folder(input_folder)

print(f"Processed files in {input_folder}")
