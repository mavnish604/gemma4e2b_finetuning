# Abhijit Chavda — Gemma-4-E2B Fine-tune Project

A local fine-tuned AI model that speaks in the style of **Abhijit Chavda** — Indian science communicator, historian, and geopolitical analyst — running at **~50 tok/s** on an RTX 3050 6 GB laptop GPU.

---

## What We Built

A fully local, GPU-accelerated conversational AI that:
- Responds in Abhijit Chavda's distinct YouTube style ("Namaste dosto", realpolitik framing, India-first lens)
- Runs entirely offline with no API calls
- Loads in **~3.5 seconds** and generates at **~45–56 tok/s**
- Applies a custom LoRA fine-tune on top of Gemma-4-E2B

---

## Project Journey

### Phase 1 — Dataset Creation
- Downloaded transcripts from Abhijit Chavda's **#AskAbhijit** YouTube playlist using `yt-dlp`
- Raw `.vtt` subtitle files stored in `raw_subtitles/`
- Cleaned and processed transcripts using `text_cleaning.py` → `gemma_dataset_txt/`
- Used a local Gemma model to extract Q&A pairs from the transcripts via `generate_qa_dataset.py`
- Final training dataset used for PEFT: **`abhijit_instruct.jsonl`** (26 MB)
  - Format: `{"instruction": "Answer this question in the style of Abhijit Chavda.", "input": "...", "output": "..."}`
  - Outputs are **verbatim transcript responses** — full, unedited answers from Abhijit's videos
  - Much richer than the smaller Q&A extract; captures natural speech patterns, tangents, and long-form reasoning

### Phase 2 — LoRA Fine-tuning
- Fine-tuned **Google Gemma-4-E2B-it** using **Unsloth** on the extracted Q&A dataset
- Training config: LoRA rank=32, alpha=32, targeting all projection layers (`q_proj.linear`, `k_proj.linear`, etc.)
- Trained adapter saved to `abhijit_chavda_lora/` (268 MB PEFT adapter)
- The fine-tune captured Abhijit's **speaking style and tone** — realpolitik framing, passionate delivery, "Jai Hind" closings

### Phase 3 — Inference Attempts (HF Transformers)
- Initial approach: load merged model with HF `AutoModelForCausalLM` + `bitsandbytes` 4-bit NF4
- **Problem 1:** The LoRA merge was interrupted mid-save — only 5 of 7 shards written, no `model.safetensors.index.json`
- **Problem 2:** Even with base + LoRA loaded directly, generation speed was **~1–2 tok/s**
- **Root cause:** Gemma-4-E2B is a multimodal model (vision + audio + language). At 4-bit, the language model layers alone barely fit in 6 GB VRAM. `device_map="auto"` spilled language model layers to CPU RAM, causing per-token CPU↔GPU transfers — the bottleneck

### Phase 4 — Switching to llama.cpp
- Downloaded pre-built GGUF from `bartowski/google_gemma-4-E2B-it-GGUF` on HuggingFace
  - Format: `Q4_K_L` (4.1 GB) — fits entirely in 6 GB VRAM, all 35 layers on GPU
- Installed `llama-cpp-python` with CUDA 12.1 pre-built wheel
- Converted the PEFT LoRA adapter to GGUF format using `llama.cpp`'s `convert_lora_to_gguf.py`
  - Result: `gguf_models/abhijit_lora.gguf` (203 MB)
- **Speed result: ~45–56 tok/s** — 40x improvement over HF transformers
---

## Current Architecture

```
User question
     │
     ▼
System Prompt (India-first realpolitik persona)
     │
     ▼
GGUF Base Model: google_gemma-4-E2B-it-Q4_K_L.gguf  [GPU, all layers]
     +
GGUF LoRA Adapter: abhijit_lora.gguf                 [applied at runtime]
     │
     ▼
Streaming response @ ~50 tok/s
```

| Component | Role |
|-----------|------|
| Base GGUF (Q4_K_L) | Language understanding + generation |
| LoRA adapter (GGUF) | Abhijit's speaking style, tone, delivery |
| System prompt | Specific stances, India-first analytical lens |

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3050 6 GB Laptop |
| RAM | 23 GB |
| Disk | 55 GB NVMe (92% used) |
| CUDA | 12.0 |

---

## File Structure

```
gemma4e2b_finetuning/
├── inference_gguf.py              # ← Main inference script (use this)
├── generate_qa_dataset.py         # Dataset generation from transcripts
├── text_cleaning.py               # Transcript pre-processing
├── prepare_data.py                # Data preparation utilities
├── instruction_conversion.py      # Instruction format conversion
├── download_transcript.py         # YouTube transcript downloader
│
├── gguf_models/
│   ├── google_gemma-4-E2B-it-Q4_K_L.gguf   # Base model (4.1 GB)
│   └── abhijit_lora.gguf                     # Fine-tuned LoRA adapter (203 MB)
│
├── abhijit_chavda_lora/           # Original PEFT adapter (source of truth)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # (237 MB)
│   └── tokenizer files
│
├── raw_subtitles/                 # Raw .vtt transcripts from YouTube (392 MB)
├── gemma_dataset_txt/             # Cleaned transcript text files (39 MB)
├── abhijit_instruct.jsonl         # ← Dataset used for PEFT training (26 MB)
├── gemma_finetune_dataset.jsonl   # Q&A extract (1.4 MB)
├── gemma_full_style_dataset.jsonl # Extended style dataset (38 MB)
└── venv/                          # Python virtual environment
```

---

## Usage

```bash
cd /media/tst_imperial/Projects/gemma4e2b_finetuning
source venv/bin/activate

# Interactive REPL (with LoRA fine-tune)
python inference_gguf.py

# Single question
python inference_gguf.py --prompt "What do you think about Nehru?"

# Base model only (no LoRA)
python inference_gguf.py --no_lora

# Control generation length
python inference_gguf.py --max_new_tokens 200

# In-session commands
# /temp 0.9    → change temperature
# /tokens 500  → change max output length
```

---

## Key Findings

### What the LoRA learned
- ✅ Abhijit's **speaking style** — energetic pacing, CAPS for emphasis, "Namaste dosto"
- ✅ Structural patterns — numbered breakdowns, "Let me tell you", "Jai Hind" closings
- ✅ Realpolitik framing — "India's National Interest" as primary lens


### LoRA vs Base model difference
The LoRA is measurably different — compare the same question:
- **Base model:** moralistic framing, ends with YouTube CTA ("Let me know in the comments!")
- **LoRA model:** realpolitik framing, ends with "Jai Hind!", less diplomatic hedging

### Why llama.cpp is dramatically faster
HF Transformers with `device_map="auto"` spills layers to CPU RAM when VRAM is tight.
Every generated token triggers a CPU↔GPU transfer for offloaded layers → ~1 tok/s.
llama.cpp fits the entire Q4_K_L model (4.1 GB) within the 6 GB VRAM budget,
so all computation stays on GPU → ~50 tok/s.

---

## Next Steps (Planned)

### RAG Integration
Combine the fine-tuned model with **Retrieval-Augmented Generation** using the existing transcripts:
- Embed `gemma_dataset_txt/` into a vector store (ChromaDB/FAISS)
- At query time, retrieve relevant transcript chunks and inject as context
- Result: model responds grounded in Abhijit's **actual words** on any topic, not just trained stances

**Architecture:**
```
User question → Vector search (transcripts) → Top-3 chunks injected as context
→ System prompt + LoRA style → Grounded, factually accurate response
```

### Why RAG > Retraining for this use case
- Retraining requires thousands of topic-specific examples to reliably override base model priors
- RAG retrieves the exact thing Abhijit said in a specific video — zero memorization needed
- New videos = just add transcripts to the vector store, no retraining required
