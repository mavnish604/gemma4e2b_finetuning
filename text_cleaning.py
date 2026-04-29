import os
import json
import torch
import textwrap
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id = "google/gemma-1.1-2b-it"

print(f"Loading {model_id} (Quantized 4-bit Mode)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

def generate_qa_direct(text_chunk):
    # Using chat template format to leverage instruct tuning properly
    messages = [
        {"role": "user", "content": f"""You are an expert AI data annotator. Read the following text transcript and extract exactly 2 Question and Answer pairs based ONLY on the text. The "instruction" field must contain a fully formed question asked by a user, and the "response" field must contain the correct answer derived from the text.
Respond STRICTLY in this JSON format:
[
    {{"instruction": "<write a specific question here>", "response": "<write the answer here>"}},
    {{"instruction": "<write a second specific question here>", "response": "<write the second answer here>"}}
]

TEXT:
{text_chunk}"""}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return generated_text

def chunk_text(text, max_words=500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

# Data loading logic
print("Scanning gemma_dataset_txt/ directory...")
txt_files = sorted(glob.glob("gemma_dataset_txt/*.txt"))

output_file = "gemma_finetune_dataset.jsonl"
progress_file = "text_cleaning_progress.json"

processed_files = set()
if os.path.exists(progress_file):
    with open(progress_file, "r") as f:
        processed_files = set(json.load(f))

print(f"Found {len(txt_files)} transcript files. Currently processed: {len(processed_files)}.")

added_count = 0

with open(output_file, "a", encoding="utf-8") as outfile:
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        if filename in processed_files:
            continue
            
        print(f"\nProcessing {filename}...")
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = list(chunk_text(full_text, max_words=500))

        
        file_added = 0
        for chunk in tqdm(chunks, desc=f"Chunks from {filename}"):
            try:
                response_text = generate_qa_direct(chunk)
                
                # Simple cleanup for JSON extraction
                clean_json = response_text
                if "```json" in response_text:
                    clean_json = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    clean_json = response_text.split("```")[1].split("```")[0].strip()
                
                # Extract JSON brackets if model adds extra text
                start = clean_json.find('[')
                end = clean_json.rfind(']') + 1
                if start != -1 and end != 0:
                    clean_json = clean_json[start:end]

                qa_pairs = json.loads(clean_json)
                for pair in qa_pairs:
                    if "instruction" in pair and "response" in pair:
                        json.dump(pair, outfile, ensure_ascii=False)
                        outfile.write("\n")
                        added_count += 1
                        file_added += 1
                
                outfile.flush()
                torch.cuda.empty_cache()
                
            except Exception as e:
                # Silently ignore format errors / hallucinations
                continue
                
        processed_files.add(filename)
        with open(progress_file, "w") as f:
            json.dump(list(processed_files), f)
            
        print(f"--> Extracted {file_added} pairs from {filename}.")

print(f"\n✅ All caught up! Added {added_count} total new pairs to {output_file}.")