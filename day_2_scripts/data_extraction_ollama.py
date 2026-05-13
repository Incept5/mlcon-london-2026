import ollama


def generate_response(prompt):
    try:
        response = ollama.generate(
            model="qwen3.5:4b",
            prompt=prompt,
            think=False,
            options={
                "num_ctx": 8192,
                "temperature": 0.7
            }
        )
        return response['response']
    except Exception as e:
        print("Error:", e)
        return None


def summarise(text):
    prompt = f"You are a summary assistant. Summarise the following text into very short sentence...\n{text}"
    return generate_response(prompt)


def extract(text, what):
    prompt = (f"You are a concise data extraction assistant. Extract {what} from the following text,"
              f"give the answer only, nothing else...\n{text}")
    return generate_response(prompt)


def main():
    # Source: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    text = """# Qwen3.6-35B-A3B Model Card

## Model Description

Qwen3.6-35B-A3B is the first open-weight variant of the Qwen3.6 series, built on community feedback to prioritize stability and real-world utility. It offers developers a more intuitive, responsive, and productive coding experience.

Type: Causal Language Model with Vision Encoder
Training Stage: Pre-training & Post-training
License: Apache 2.0

## Qwen3.6 Highlights

- Agentic Coding: Enhanced handling of frontend workflows and repository-level reasoning with greater fluency and precision
- Thinking Preservation: New option to retain reasoning context from historical messages, streamlining iterative development and reducing overhead

## Model Specifications

### Language Model
- Parameters: 35B total, 3B activated
- Hidden Dimension: 2048
- Token Embedding: 248,320 (Padded)
- Layers: 40
- Hidden Layout: 10 x (3 x (Gated DeltaNet -> MoE) + 1 x (Gated Attention -> MoE))
- Context Length: 262,144 tokens natively, extensible to 1,010,000 tokens

### Mixture of Experts
- Number of Experts: 256
- Activated Experts: 8 Routed + 1 Shared
- Expert Intermediate Dimension: 512

## Key Benchmark Results

### Coding Agent Benchmarks
- SWE-bench Verified: 73.4
- SWE-bench Multilingual: 67.2
- SWE-bench Pro: 49.5
- Terminal-Bench 2.0: 51.5
- QwenClawBench: 52.6
- NL2Repo: 29.4

### Knowledge & Reasoning
- MMLU-Pro: 85.2
- MMLU-Redux: 93.3
- GPQA: 86.0
- AIME26: 92.7
- C-Eval: 90.0

### Vision-Language Performance
- MMMU: 81.7
- RealWorldQA: 85.3
- OmniDocBench1.5: 89.9
- VideoMMU: 83.7
- RefCOCO (avg): 92.0

## Recommended Sampling Parameters

### Thinking Mode for General Tasks
- temperature: 1.0
- top_p: 0.95
- top_k: 20
- min_p: 0.0
- presence_penalty: 1.5
- repetition_penalty: 1.0

### Instruct Mode for General Tasks
- temperature: 0.7
- top_p: 0.8
- top_k: 20
- min_p: 0.0
- presence_penalty: 1.5
- repetition_penalty: 1.0

### Instruct Mode for Reasoning Tasks
- temperature: 1.0
- top_p: 1.0
- top_k: 40
- min_p: 0.0
- presence_penalty: 2.0
- repetition_penalty: 1.0

## Best Practices

1. Output Length: Use 32,768 tokens for most queries; 81,920 tokens for complex math/programming problems
2. Math: Include "Please reason step by step, and put your final answer within \\boxed{}."
3. Multiple-choice: Use JSON format with answer field containing choice letter
"""

    print("=== Summarization Demo ===")
    summary = summarise(text)
    print(f"Summary: {summary}\n")

    print("=== Data Extraction Demo ===")
    data = "Sampling parameters for Instruct Mode for General Tasks"
    extracted_info = extract(text, data)
    print(f"{data}: {extracted_info}")


if __name__ == "__main__":
    main()
