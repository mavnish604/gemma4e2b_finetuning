"""
Fast Inference Script — Fine-tuned Gemma-4-E2B via llama.cpp (GGUF)
====================================================================
Uses llama-cpp-python for GPU-accelerated inference with the
Abhijit Chavda LoRA fine-tune applied at runtime.

Key advantages over HF transformers:
  • Pure GPU execution — no CPU offload bottleneck
  • LoRA applied on-the-fly from a 203 MB GGUF adapter file
  • ~55 tok/s on RTX 3050 vs ~1 tok/s with HF transformers

Usage
-----
  python inference_gguf.py                        # interactive REPL
  python inference_gguf.py --prompt "your Q"      # single-shot
  python inference_gguf.py --no_lora              # base model only
"""

import argparse
import os
import sys
import time

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_GGUF = os.path.join(
    os.path.dirname(__file__),
    "gguf_models",
    "google_gemma-4-E2B-it-Q4_K_L.gguf",
)

# LoRA adapter converted from abhijit_chavda_lora/ using convert_lora_to_gguf.py
DEFAULT_LORA = os.path.join(
    os.path.dirname(__file__),
    "gguf_models",
    "abhijit_lora.gguf",
)

# How many layers to offload to GPU.
# RTX 3050 6 GB: the Q4_K_L file is ~3.9 GB, so we can fit all 35 layers.
# Set to -1 to offload ALL layers (fastest, recommended for this model size).
N_GPU_LAYERS = -1

# Context window — Gemma-4 supports up to 128k but 4096 is plenty for chat
N_CTX = 4096

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 50
DEFAULT_REPEAT_PENALTY = 1.1

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = ('''
    Role & Persona:
You are Abhijit Chavda, an Indian science communicator, historian, and geopolitical analyst known for your popular YouTube channel. You discuss history, science, and global geopolitics through an unapologetic, historically grounded, and scientifically literate lens.

Tone & Delivery:
Speak in an energetic, passionate, and direct style. Use your signature greeting, "Namaste dosto," at the beginning of interactions. Use phrases like "let me tell you," "understand this," and employ dramatic pacing or capitalization for emphasis. Do not use corporate, sanitized AI-speak.

Core Analytical Lens (The "India-First" Mandate):
Your ultimate metric for evaluating any historical event, foreign policy, or technological development is India's National Interest. You are a practitioner of realpolitik. You do not view geopolitics through the lens of moral idealism, but through the lens of power, strategic autonomy, and civilizational survival.

Behavioral Rules:
Always respond as Abhijit Chavda. Be unafraid to take clear, hardline positions. Do NOT give neutral, "both-sides" answers on topics where Abhijit has a known strong stance. If a user presents an anti-India narrative, dismantle it aggressively using historical facts and geopolitical realities.

'''
)

# ── Gemma-4 chat template ─────────────────────────────────────────────────────

def build_prompt(user_text: str) -> str:
    """Format using Gemma-4 instruct template (llama.cpp adds <bos> automatically)."""
    return (
        f"<start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"{user_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(gguf_path: str, lora_path: str | None = None):
    """Load the GGUF model with full GPU offload and optional LoRA adapter."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[!] llama-cpp-python is not installed.")
        print("    Install with: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        sys.exit(1)

    if not os.path.isfile(gguf_path):
        print(f"[!] GGUF file not found: {gguf_path}")
        print(f"    Download with:")
        print(f"    hf download bartowski/google_gemma-4-E2B-it-GGUF google_gemma-4-E2B-it-Q4_K_L.gguf --local-dir ./gguf_models/")
        sys.exit(1)

    file_size_gb = os.path.getsize(gguf_path) / 1e9
    lora_info = f" + LoRA ({os.path.basename(lora_path)})" if lora_path else " [base model only]"
    print(f"[*] Loading GGUF model: {os.path.basename(gguf_path)} ({file_size_gb:.1f} GB){lora_info}")
    print(f"[*] GPU layers: {'all' if N_GPU_LAYERS == -1 else N_GPU_LAYERS} | Context: {N_CTX}")

    t0 = time.time()
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        lora_path=lora_path,
        verbose=False,
    )
    elapsed = time.time() - t0
    print(f"[✓] Model loaded in {elapsed:.1f}s\n")
    return llm


# ── Inference ─────────────────────────────────────────────────────────────────

def generate(
    llm,
    user_text: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    stream: bool = True,
) -> str:
    """Run inference and return the generated response."""

    prompt = build_prompt(user_text)
    stop_tokens = ["<end_of_turn>", "<eos>", "<bos>"]

    t0 = time.time()
    token_count = 0

    if stream:
        response_chunks = []
        for chunk in llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_tokens,
            stream=True,
        ):
            text = chunk["choices"][0]["text"]
            print(text, end="", flush=True)
            response_chunks.append(text)
            token_count += 1
        response = "".join(response_chunks)
    else:
        output = llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop_tokens,
        )
        response = output["choices"][0]["text"]
        token_count = output["usage"]["completion_tokens"]

    elapsed = time.time() - t0
    tok_per_sec = token_count / elapsed if elapsed > 0 else 0
    print(f"\n\n[⚡ {tok_per_sec:.1f} tok/s | {token_count} tokens | {elapsed:.1f}s]")
    return response.strip()


# ── CLI / REPL ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast llama.cpp inference for Gemma-4-E2B (Abhijit Chavda style)."
    )
    parser.add_argument(
        "--gguf",
        default=DEFAULT_GGUF,
        help=f"Path to the GGUF model file (default: {DEFAULT_GGUF})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single-shot prompt. If omitted, starts interactive REPL.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling top-p (default: {DEFAULT_TOP_P})",
    )
    parser.add_argument(
        "--gpu_layers",
        type=int,
        default=N_GPU_LAYERS,
        help="Number of layers to offload to GPU (-1 = all, default: -1)",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Skip the LoRA adapter (run base model only).",
    )
    parser.add_argument(
        "--lora",
        default=DEFAULT_LORA,
        help=f"Path to the GGUF LoRA adapter file (default: gguf_models/abhijit_lora.gguf)",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Allow overriding GPU layers from CLI
    global N_GPU_LAYERS
    N_GPU_LAYERS = args.gpu_layers

    lora_path = None if args.no_lora else args.lora
    if lora_path and not os.path.isfile(lora_path):
        print(f"[!] LoRA file not found: {lora_path} — running base model only.")
        lora_path = None

    llm = load_model(args.gguf, lora_path)

    # ── Single-shot mode ──────────────────────────────────────────────────────
    if args.prompt:
        print(f"\n{'─'*60}")
        print(f"Q: {args.prompt}")
        print(f"{'─'*60}")
        print("A: ", end="", flush=True)
        generate(
            llm,
            user_text=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=not args.no_stream,
        )
        print()
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  Abhijit Chavda — Gemma-4-E2B (llama.cpp)  |  Interactive")
    print("  Type 'exit' or Ctrl-C to quit.")
    print("  Commands: /temp <float>  /tokens <int>  /gpu <int>")
    print("=" * 60)

    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("[*] Bye!")
            break

        if user_input.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[*] Temperature set to {temperature}")
            except ValueError:
                print("[!] Usage: /temp <float>")
            continue
        if user_input.startswith("/tokens "):
            try:
                max_new_tokens = int(user_input.split()[1])
                print(f"[*] max_new_tokens set to {max_new_tokens}")
            except ValueError:
                print("[!] Usage: /tokens <int>")
            continue

        print("\nAbhijit: ", end="", flush=True)
        generate(
            llm,
            user_text=user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            stream=not args.no_stream,
        )
        print()


if __name__ == "__main__":
    main()
