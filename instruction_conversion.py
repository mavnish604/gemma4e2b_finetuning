import json
import re

def preprocess_abhijit_data(input_file, output_file):
    # Pattern to identify when a new question starts (e.g., "Malik s says", "Anand asks")
    pattern = re.compile(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?(?:\s\d+)?)\s+(?:says|asks)(?::|\s+)')
    
    qa_pairs = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            
            # Split the transcript into segments based on the questioner's name
            segments = pattern.split(text)
            # segments[0] is usually the intro; we skip it
            for i in range(1, len(segments), 2):
                name = segments[i]
                # The content after the name contains the question + the answer
                content = segments[i+1].strip()
                
                # We format this as an instruction to capture the "Ask Abhijit" persona
                qa_pairs.append({
                    "instruction": f"Answer this question in the style of Abhijit Chavda.",
                    "input": content[:200], # Rough heuristic for the question part
                    "output": content       # The full response including the question repetition
                })

    with open(output_file, 'w') as f:
        for entry in qa_pairs:
            f.write(json.dumps(entry) + "\n")

preprocess_abhijit_data("gemma_full_style_dataset.jsonl", "abhijit_instruct.jsonl")