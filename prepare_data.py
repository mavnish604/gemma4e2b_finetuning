import os
import re
import json

# 1. Folder ka naam jahan aapki saari txt files hain
extract_path = 'extracted_transcripts' 
output_file = 'gemma_raw_style_dataset.jsonl'

processed_data = []

print(f"Scanning folder: {extract_path}...")

# Poori directory scan karke data collect karna
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Cleaning: [Music] jaise tags aur extra spaces hatana
                cleaned = re.sub(r'\[.*?\]', '', content, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                # Agar file khali nahi hai toh add karein
                if len(cleaned) > 20: 
                    processed_data.append({"text": cleaned})
            except Exception as e:
                print(f"Error reading {file}: {e}")

# 2. JSONL file mein save karna
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in processed_data:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"\n✅ Done! Total {len(processed_data)} files process ho gayi hain.")
print(f"Aapki file yahan taiyar hai: {os.path.abspath(output_file)}")
