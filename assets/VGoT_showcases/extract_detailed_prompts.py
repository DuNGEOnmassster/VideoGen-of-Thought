import os
import json

def extract_prompts(base_dir):
    # Recursively traverse the directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'image_prompt_pairs.json':
                json_file_path = os.path.join(root, file)
                try:
                    # Read the JSON file
                    with open(json_file_path, 'r') as json_file:
                        data = json.load(json_file)
                    # Extract the prompt field
                    prompts = [item['prompt'] for item in data]
                    # Define the output file path
                    output_file = os.path.join(root, 'detailed_prompts.txt')
                    # Write to the output file
                    with open(output_file, 'w') as outfile:
                        for prompt in prompts:
                            outfile.write(f"{prompt}\n")
                    print(f"Prompts extracted from {json_file_path} and saved to {output_file}")
                except Exception as e:
                    print(f"Error processing {json_file_path}: {e}")

# The current directory
base_dir = '.'

# Call the function
extract_prompts(base_dir)
print("All prompts have been extracted and saved to their respective directories.")