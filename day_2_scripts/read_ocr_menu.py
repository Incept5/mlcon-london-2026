import ollama
import time
from pathlib import Path
from typing import List, Dict, Any

HERE = Path(__file__).parent

MODELS: List[str] = [
    "qwen3.5:4b",
]
IMAGE_PATH: str = str(HERE / "data" / "IMG_5.jpg")


def get_ocr_prompt(model_name: str) -> str:
    # DeepSeek-OCR uses a specialized "Free OCR." prompt; others use natural language
    if "deepseek-ocr" in model_name.lower():
        return "Free OCR."

    return "Read all text from this image. Output only the text exactly as written."


def warmup_model(model_name: str, image_path: str) -> None:
    print(f"Warming up model: {model_name}...")

    try:
        dummy_prompt = "What do you see in this image?"

        ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': dummy_prompt,
                'images': [image_path]
            }],
            think=False,
            options={'temperature': 0.0}
        )
        print(f"Model {model_name} loaded and ready")
    except Exception as e:
        print(f"Warmup failed for {model_name}: {e}")


def extract_text_from_image(model_name: str, image_path: str):
    prompt = get_ocr_prompt(model_name)

    start_time = time.time()
    response = ollama.chat(
        model=model_name,
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }],
        think=False,
        options={'temperature': 0.0}
    )
    inference_time = time.time() - start_time

    return response['message']['content'], inference_time


def print_header(title: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_result(model_name: str, text: str, inference_time: float, width: int = 60) -> None:
    print(f"\nModel: {model_name}")
    print("-" * width)
    print(text)
    print(f"\nInference time: {inference_time:.2f}s")
    print("=" * width)


def main() -> None:
    print_header("MENU OCR DEMO")
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {', '.join(MODELS)}")

    results: Dict[str, Any] = {}

    for model in MODELS:
        try:
            warmup_model(model, IMAGE_PATH)

            text, inference_time = extract_text_from_image(model, IMAGE_PATH)
            print_result(model, text, inference_time)

            results[model] = {'success': True, 'time': inference_time}

        except Exception as e:
            print(f"\nModel: {model}")
            print("-" * 60)
            print(f"ERROR: {e}")
            print("=" * 60)
            results[model] = {'success': False, 'error': str(e)}

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model, result in results.items():
        if result['success']:
            print(f"{model}: Success, {result['time']:.2f}s")
        else:
            print(f"{model}: ERROR - {result['error']}")
    print("="*60)


if __name__ == "__main__":
    main()
