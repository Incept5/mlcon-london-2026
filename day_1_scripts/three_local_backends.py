import ollama
import lmstudio as lms
from mlx_vlm import load, generate as mlx_generate


def ollama_response(prompt):
    response = ollama.generate(model="qwen3.5:4b", prompt=prompt, think=False)
    return response["response"]


def lmstudio_response(prompt):
    model = lms.llm("qwen3.5-4b")
    return model.respond(prompt + " /no_think").content


def mlx_response(prompt):
    model, processor = load("/Users/jdavies/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit")
    formatted = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    return mlx_generate(model, processor, formatted, verbose=False).text


if __name__ == "__main__":
    q = "Hello"
    print("ollama   :", ollama_response(q))
    print("lmstudio :", lmstudio_response(q))
    print("mlx      :", mlx_response(q))
