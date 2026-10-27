"""Generate a text answer using a locally downloaded TelecomGPT-R1."""
import argparse
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent / "models" / "TelecomGPT-R1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Question for the model")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--check", action="store_true", help="Check dependencies/GPU without loading weights")
    args = parser.parse_args()
    if not args.check and not (args.prompt and args.prompt.strip()):
        parser.error("Provide a nonempty question or use --check.")
    if not 1 <= args.max_new_tokens <= 4096:
        parser.error("--max-new-tokens must be between 1 and 4096.")

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    print(f"PyTorch: {torch.__version__}; Transformers: {transformers.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA/ROCm GPU detected. Install a GPU-enabled PyTorch build and check drivers.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This integration requires a GPU supporting bfloat16.")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}; {props.total_memory / 2**30:.1f} GiB")
    if args.check:
        print("Dependency/GPU checks passed. This does not guarantee sufficient free VRAM.")
        return

    model_dir = args.model_dir.expanduser().resolve()
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise SystemExit(f"Model not found in {model_dir}. Run download_model.py first.")

    # The upstream config declares Qwen3_5ForConditionalGeneration.
    # Explicit bf16 avoids the float32 dtype recorded in that config.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    ).eval()
    messages = [
        {"role": "system", "content": "You are a telecom assistant. Answer in the user's language. State assumptions and uncertainty."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if inputs["input_ids"].shape[1] + args.max_new_tokens > 8192:
        raise SystemExit("Question plus output exceeds the 8192-token limit used by this integration.")
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    answer = outputs[0, inputs["input_ids"].shape[1]:]
    print(tokenizer.decode(answer, skip_special_tokens=True))


if __name__ == "__main__":
    main()
