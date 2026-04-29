import os
import json
import time
from itertools import cycle
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Find all Groq API keys in .env
api_keys = [v for k, v in os.environ.items() if k.startswith("GROQ_API_KEY")]
if not api_keys:
    raise ValueError("No GROQ_API_KEY found in .env file.")

# Initialize Groq clients for each key
clients = [Groq(api_key=key) for key in api_keys]
client_cycle = cycle(clients)

TXT_DIR = "gemma_dataset_txt"
OUTPUT_JSONL = "gemma_finetune_dataset.jsonl"
PROGRESS_FILE = "extraction_progress.json"
MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an expert data extractor. Your goal is to process the auto-generated transcript from an #AskAbhijit YouTube video and extract Question & Answer pairs suitable for fine-tuning an AI model.

RULES:
1. Extract ALL Q&A pairs from the provided text snippet.
2. The 'instruction' should be the clear question asked (prefix it with 'Q: ').
3. The 'response' should be Abhijit's exact answer. Preserve his original wording and speaking style exactly as recorded in the transcript. Do NOT restructure paragraphs or clean up his grammar. You may only fix obvious YouTube transcription spelling errors.
4. IGNORE sponsor messages, "please subscribe" calls, greetings, music, overlaps, or off-topic meta-chatter.
5. If the snippet contains no meaningful questions, return an empty JSON array `[]` (or empty qa_pairs).
6. RETURN ONLY a JSON Object with a single key 'qa_pairs' mapping to the array of Q&A objects. Do not return any other text.

REQUIRED JSON FORMAT:
{
  "qa_pairs": [
    {
        "instruction": "Q: What is the concept of a Nation?",
        "response": "A nation is a modern geopolitical concept..."
    }
  ]
}
"""

def chunk_text(text, max_words=800):
    """Splits text into chunks by word count because the API has context limits."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def process_chunk(chunk_text, client, max_retries=3):
    """Sends a chunk to Groq API and returns parsed JSON Q&A pairs."""
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Extract Q&A from this transcript chunk:\n\n{chunk_text}"}
                ],
                model=MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            
            try:
                data = json.loads(response_content)
                if isinstance(data, dict):
                    if 'qa_pairs' in data:
                        return data['qa_pairs']
                    # Fallback for other potential keys
                    for key in data:
                        if isinstance(data[key], list):
                            return data[key]
                    if 'instruction' in data:
                        return [data]
                    return []
                elif isinstance(data, list):
                    return data
                return []
            except json.JSONDecodeError:
                print(f"Failed to parse JSON on attempt {attempt + 1}")
                time.sleep(2)
                continue
                
        except Exception as e:
            print(f"API Error with current key: {e}")
            if "429" in str(e):
                print("Rate limit hit, switching key and waiting longer...")
                return "RETRY"
            time.sleep(5)
            
    return []

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_files):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_files), f)

def main():
    if not os.path.exists(TXT_DIR):
        print(f"Directory {TXT_DIR} not found. Please run download_transcript.py first.")
        return

    processed_files = load_progress()
    print(f"🚀 Resuming LLM Q&A Extraction using {len(clients)} keys...")
    
    file_list = sorted([f for f in os.listdir(TXT_DIR) if f.endswith('.txt')])
    total_files = len(file_list)
    total_qa = 0

    # Open in append mode to preserve previous work
    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as f_out:
        for filename in file_list:
            if filename in processed_files:
                print(f"⏩ Skipping already processed: {filename}")
                continue
                
            filepath = os.path.join(TXT_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f_in:
                text_content = f_in.read()

            print(f"Processing: {filename} ({len(text_content.split())} words)")
            
            chunks = list(chunk_text(text_content))
            video_qa_count = 0
            
            for i, chunk in enumerate(chunks):
                print(f"  -> Chunk {i+1}/{len(chunks)}...")
                
                success = False
                while not success:
                    current_client = next(client_cycle)
                    qa_pairs = process_chunk(chunk, current_client)
                    
                    if qa_pairs == "RETRY":
                        time.sleep(10) # Wait for rate limit to chill
                        continue
                    
                    for pair in qa_pairs:
                        if "instruction" in pair and "response" in pair:
                            json.dump(pair, f_out)
                            f_out.write("\n")
                            total_qa += 1
                            video_qa_count += 1
                    
                    success = True
                    # Small breath between requests to keep RPM healthy
                    time.sleep(1.0) 

            processed_files.add(filename)
            save_progress(processed_files)
            print(f"✅ Finished {filename}. Generated {video_qa_count} pairs (Total: {total_qa}).")

    print(f"🎉 Dataset generation complete! Created {total_qa} additional pairs in {OUTPUT_JSONL}.")

if __name__ == "__main__":
    main()
