#!/usr/bin/env python3
"""
Generate a real-FFT stress dataset for Qwen3-30B-A3B full fine-tuning.

Requirement:
  - cutoff_len is set to 1024.
  - Every sample MUST tokenize to > 7000 tokens so that after truncation to
    1024 every training step uses an identical sequence length (exactly 1024
    real tokens, no padding), giving clean, comparable TPS / memory numbers.

Output (written into this same directory, FFTtest/dataset/):
  fft_real_100.json      -- 100 alpaca-format samples (instruction/input/output)
  dataset_info.json      -- LLaMA-Factory dataset registration

Token length is VERIFIED with the real Qwen3 tokenizer when available; the
script regenerates (grows) any sample that falls below the token floor.
"""

import json
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR  # dataset lives directly in FFTtest/dataset/

DATASET_NAME = "fft_real_100"
NUM_SAMPLES = 100
TOKEN_FLOOR = 7000            # every sample must exceed this many tokens
PARAGRAPHS_PER_SAMPLE = 80    # initial size; grown automatically if too short
MODEL_PATH = "/mnt/data3/models/Qwen3-30B-A3B"

# ---------------------------------------------------------------------------
# Rich body paragraphs (~400-550 chars each). Reused with replacement so each
# sample is a randomised, long block of coherent technical prose.
# ---------------------------------------------------------------------------
BODY_PARAGRAPHS = [
    "The development of large language models has fundamentally transformed the landscape of "
    "artificial intelligence research. These systems, trained on vast corpora of text, exhibit "
    "remarkable capabilities in language understanding, generation, and reasoning. The scaling "
    "laws observed in transformer-based architectures suggest that increasing model size, data "
    "volume, and compute budget yields predictable improvements in downstream performance across "
    "a wide variety of tasks, from question answering to code generation.",

    "Mixture-of-Experts architectures represent a compelling approach to scaling neural networks "
    "efficiently. Instead of activating all parameters for every input token, MoE models route "
    "each token to a small subset of expert sub-networks, dramatically reducing the effective "
    "computation per token while maintaining a large total parameter count. This sparse activation "
    "pattern enables training models with hundreds of billions of parameters at a fraction of the "
    "FLOPs required by equivalent dense architectures, making large-scale training more accessible.",

    "Quantization techniques have become indispensable tools for deploying large models in "
    "memory-constrained environments. By representing weights with fewer bits-typically INT8, "
    "INT4, or even lower precision-quantization reduces both memory footprint and memory bandwidth "
    "requirements, often with minimal impact on model accuracy. Advanced quantization schemes "
    "such as GPTQ, AWQ, and SmoothQuant incorporate calibration data and activation statistics "
    "to minimize quantization error, preserving the performance of the full-precision model.",

    "The attention mechanism lies at the heart of modern transformer architectures. By computing "
    "pairwise compatibility scores between query and key vectors, attention allows each token to "
    "selectively aggregate information from all positions in the sequence. Grouped-query attention "
    "reduces the key-value cache memory by sharing key and value projections across multiple query "
    "heads, substantially cutting inference memory costs for long-context generation without "
    "significant accuracy degradation in practice.",

    "Full parameter fine-tuning remains the gold standard for adapting pretrained language models "
    "to specialized domains or tasks. Unlike parameter-efficient methods such as LoRA, full "
    "fine-tuning updates every weight in the model, allowing maximum flexibility and expressivity. "
    "However, it requires storing gradients and optimizer states for all parameters, leading to "
    "memory requirements that are typically three to four times the model size itself, necessitating "
    "distributed training strategies such as FSDP or DeepSpeed ZeRO.",

    "Fully Sharded Data Parallel training distributes model parameters, gradients, and optimizer "
    "states evenly across all participating GPUs. During the forward pass, each device gathers the "
    "parameters for the current layer from its peers, computes activations, and then discards the "
    "gathered weights. This all-gather and re-scatter pattern reduces per-device memory from O(N) "
    "to O(N/k) where k is the number of GPUs, enabling training of models that would otherwise not "
    "fit in the aggregate GPU memory of a single node.",

    "Gradient checkpointing trades computation for memory by discarding intermediate activations "
    "during the forward pass and recomputing them as needed during backpropagation. Rather than "
    "storing the full activation tensor for every layer, only a subset of checkpoint activations "
    "is retained; the remaining activations are regenerated by replaying the forward computation "
    "from the nearest checkpoint. This technique typically reduces activation memory by a factor "
    "proportional to the square root of the sequence length, enabling longer context training.",

    "The Adam optimizer and its variants dominate the training of large language models due to "
    "their adaptive per-parameter learning rates and robustness to gradient scale differences "
    "across layers. AdamW decouples weight decay from the gradient update, improving regularization "
    "behavior in high-dimensional parameter spaces. However, storing the first and second moment "
    "estimates doubles and triples the memory required beyond the parameters themselves, making "
    "memory-efficient optimizer implementations such as paged Adam important for extreme-scale runs.",

    "Hardware accelerators equipped with advanced matrix multiplication units have dramatically "
    "accelerated deep learning workloads. Intel's Advanced Matrix Extensions (AMX) instruction "
    "set extends traditional SIMD capabilities with tile-based matrix multiply-accumulate operations "
    "that operate on 16x16 or 32x32 blocks of bfloat16 or int8 values, delivering teraflops of "
    "throughput for neural network inference and training kernels on modern server CPUs, enabling "
    "cost-effective deployment of large MoE models on commodity hardware.",

    "NUMA-aware memory management is critical for maximizing throughput on multi-socket server "
    "platforms. Non-uniform memory access architectures allocate memory banks across physical "
    "CPU sockets, and accessing memory attached to a remote socket incurs significantly higher "
    "latency and lower bandwidth than local accesses. Pinning expert weights to specific NUMA "
    "nodes and binding worker threads to the corresponding cores eliminates cross-NUMA traffic, "
    "substantially improving effective memory bandwidth utilization for large expert weight loads.",

    "The router module in a mixture-of-experts transformer determines which experts process each "
    "token. A linear projection maps the hidden state to a logit vector over experts, from which "
    "the top-k experts are selected. The auxiliary load-balancing loss encourages uniform routing "
    "distributions, preventing expert collapse where a small fraction of experts handles the vast "
    "majority of tokens. Proper tuning of the load-balancing coefficient is essential for training "
    "stability and final model quality in large MoE models.",

    "Gradient norm clipping prevents exploding gradients by rescaling the gradient vector when its "
    "global L2 norm exceeds a specified threshold. This technique is particularly important in the "
    "early stages of training when weight initialization may produce large activations, and in "
    "models with many layers where gradient magnitudes can compound multiplicatively. The clipping "
    "threshold is typically set between 0.5 and 1.0, with careful monitoring of gradient norms "
    "throughout training to detect instability or mode collapse in generative models.",

    "Knowledge distillation transfers the capabilities of a large teacher model to a smaller "
    "student by training the student to match the teacher's output distribution rather than just "
    "the ground truth labels. The soft probability distributions produced by the teacher carry "
    "rich information about inter-class relationships and model uncertainty that hard labels alone "
    "cannot capture. Combining task-specific distillation with intermediate feature matching "
    "allows student models to achieve near-teacher performance with a fraction of the parameters.",

    "Instruction tuning adapts pretrained language models to follow natural language directives "
    "by fine-tuning on curated collections of task-instruction pairs. By exposing the model to "
    "diverse instruction formats across hundreds of tasks, instruction tuning generalizes the "
    "model's ability to interpret and respond to novel directives at inference time without "
    "additional task-specific training. Careful dataset curation, quality filtering, and format "
    "standardization are essential to achieving robust instruction-following behavior across domains.",

    "Tensor parallelism partitions individual weight matrices across multiple devices, allowing "
    "linear layers too large to fit on a single GPU to be computed collaboratively. Each device "
    "holds a row or column slice of the weight matrix, performs a local matrix multiply, and "
    "then participates in an all-reduce or all-gather to combine partial results. While effective "
    "for very large matrices, tensor parallelism introduces communication overhead proportional "
    "to the number of tensor-parallel ranks, making it most beneficial when bandwidth is ample.",

    "Continual learning poses fundamental challenges to neural network optimization because "
    "training on new tasks typically degrades performance on previously learned tasks, a "
    "phenomenon known as catastrophic forgetting. Rehearsal-based methods mitigate forgetting "
    "by maintaining a replay buffer of past examples, while regularization approaches penalize "
    "changes to weights deemed important for old tasks. Domain-incremental fine-tuning of large "
    "pretrained models generally exhibits less forgetting than training a randomly initialized network.",

    "Efficient attention approximations reduce the quadratic complexity of the standard "
    "self-attention mechanism. FlashAttention fuses the attention computation into a single "
    "kernel that tiles the query, key, and value matrices to maximize reuse within fast on-chip "
    "SRAM, avoiding materialization of the large quadratic attention weight matrix in slow HBM. "
    "This allows processing sequences tens of thousands of tokens long without the memory "
    "explosion that would otherwise occur, enabling reasoning over entire books or codebases.",

    "Tokenization strategies profoundly influence model capabilities, particularly for low-resource "
    "languages and specialized domains. Byte-pair encoding and unigram language model tokenizers "
    "learn a vocabulary of subword units that balance coverage of the training corpus with "
    "vocabulary size. Models trained with vocabulary sizes in the range of 100,000 to 200,000 "
    "tokens achieve good coverage of multilingual text while maintaining manageable embedding "
    "table sizes. SentencePiece processes raw Unicode without relying on whitespace conventions.",

    "Benchmark evaluation is a cornerstone of empirical progress in natural language processing. "
    "Standardized test suites such as MMLU, BIG-Bench, HellaSwag, and HumanEval provide "
    "reproducible metrics for comparing model capabilities across reasoning, knowledge, commonsense "
    "understanding, and code generation. However, benchmark saturation and contamination-where "
    "test examples inadvertently appear in training data-have spurred the development of dynamic "
    "evaluation protocols and held-out test sets that are periodically refreshed for integrity.",

    "Reinforcement learning from human feedback aligns language model behavior with human "
    "preferences by training a reward model on pairwise comparisons of model outputs and using "
    "it as a reward signal for policy optimization. Proximal policy optimization constrains "
    "the policy update to prevent catastrophic divergence from the reference model, while "
    "direct preference optimization frames alignment as a supervised regression problem that "
    "avoids explicit reward modeling, improving helpfulness, harmlessness, and honesty of assistants.",
]

INSTRUCTION_TEMPLATES = [
    "Provide a comprehensive and detailed analysis of the following technical document (sample {i}):",
    "Write an in-depth review and explanation of the content below, covering all key concepts (sample {i}):",
    "Elaborate extensively on the topics discussed in the following passage, adding technical depth (sample {i}):",
    "Generate a thorough academic response to the following text, discussing implications and context (sample {i}):",
    "Analyze and expand upon the following document with detailed technical commentary (sample {i}):",
]


def build_sample(idx: int, rng: random.Random, n_paragraphs: int) -> dict:
    instruction = rng.choice(INSTRUCTION_TEMPLATES).format(i=idx + 1)
    intro = (
        f"This document (index {idx + 1}, seed {rng.randint(10000, 99999)}) covers "
        f"a range of topics in machine learning, systems software, and hardware optimization. "
        f"The following sections provide detailed technical exposition, each expanding on a "
        f"distinct facet of large-scale model training and inference.\n\n"
    )
    paragraphs = rng.choices(BODY_PARAGRAPHS, k=n_paragraphs)
    output = intro + "\n\n".join(paragraphs)
    return {"instruction": instruction, "input": "", "output": output}


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        return tok
    except Exception as e:  # tokenizer optional; fall back to char heuristic
        print(f"[warn] could not load tokenizer ({e.__class__.__name__}); "
              f"using char/4 heuristic for length check.")
        return None


def _token_len(tok, sample: dict) -> int:
    text = sample["instruction"] + "\n" + sample["output"]
    if tok is not None:
        return len(tok(text, add_special_tokens=True)["input_ids"])
    return (len(text)) // 4  # rough fallback


def main():
    rng = random.Random(2026)
    tok = _load_tokenizer()

    samples = []
    for i in range(NUM_SAMPLES):
        n_par = PARAGRAPHS_PER_SAMPLE
        sample = build_sample(i, rng, n_par)
        # Grow the sample until it comfortably clears the token floor.
        guard = 0
        while _token_len(tok, sample) <= TOKEN_FLOOR and guard < 12:
            n_par += 16
            sample = build_sample(i, rng, n_par)
            guard += 1
        samples.append(sample)

    tok_lens = [_token_len(tok, s) for s in samples]
    min_tok, max_tok = min(tok_lens), max(tok_lens)
    char_lens = [len(s["instruction"]) + len(s["output"]) for s in samples]
    print(f"Generated {len(samples)} samples.")
    print(f"token length : min={min_tok}, max={max_tok}  (floor={TOKEN_FLOOR})")
    print(f"char  length : min={min(char_lens)}, max={max(char_lens)}")
    print(f"tokenizer    : {'REAL Qwen3' if tok is not None else 'char/4 heuristic'}")

    if min_tok <= TOKEN_FLOOR:
        print(f"ERROR: a sample fell below the {TOKEN_FLOOR}-token floor.")
        sys.exit(1)

    out_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Dataset written to: {out_path}")

    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    # Preserve any existing registrations in this shared dataset dir.
    dataset_info = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception:
            dataset_info = {}
    dataset_info[DATASET_NAME] = {"file_name": f"{DATASET_NAME}.json"}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Dataset info written to: {info_path}")


if __name__ == "__main__":
    main()
