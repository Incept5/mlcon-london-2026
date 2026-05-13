import json
import re
import time
from pathlib import Path
import lmstudio as lms
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent

MODELS = [
    "qwen3.5-2b@q4_0", "qwen3.5-2b@q8_0",
    "qwen3.5-4b@q4_0", "qwen3.5-4b@q8_0",
    "qwen3.5-9b@q4_0", "qwen3.5-9b@q8_0",
    "gemma-4-e2b-it@q4_0", "gemma-4-e2b-it@q8_0",
    "gemma-4-e4b-it@q4_0", "gemma-4-e4b-it@q8_0",
]
IMAGE_PATH = str(HERE / "data" / "RTS1UI9-1024x659.jpg")


def is_gemma(model_key):
    return "gemma" in model_key.lower()


def extract_boxes_from_text(text, model_key):
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    parsed = None
    if match:
        try:
            parsed = json.loads(match.group(1))
        except Exception:
            pass

    if parsed is None:
        pattern = r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        matches = re.findall(pattern, text)
        if matches:
            parsed = [{"bbox_2d": [int(x1), int(y1), int(x2), int(y2)], "label": "Beer"}
                      for x1, y1, x2, y2 in matches]

    if parsed is None:
        return None

    # Gemma outputs [y1, x1, y2, x2]; swap to [x1, y1, x2, y2]
    if is_gemma(model_key):
        for box in parsed:
            bb = box.get("bbox_2d")
            if bb and len(bb) == 4:
                box["bbox_2d"] = [bb[1], bb[0], bb[3], bb[2]]

    return parsed


def detect_beer(model_key, image_path):
    print(f"\n{'='*60}\nProcessing with model: {model_key}\n{'='*60}")

    model = lms.llm(model_key)
    image = lms.prepare_image(image_path)

    prompt = """Detect all beer glasses, mugs, or steins in this image.

Return ONLY a JSON array:
```json
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "Beer"}
]
```

Coordinates are normalized to 0-1000 where x1,y1 is top-left and x2,y2 is bottom-right. Use integers."""

    chat = lms.Chat()
    chat.add_user_message(prompt, images=[image])

    start = time.time()
    result = model.respond(chat, config={"temperature": 0.0})
    inference_time = time.time() - start

    output = result.content
    print(f"\nModel Response:\n{output[:800]}")
    print(f"\nInference time: {inference_time:.2f}s")

    boxes = extract_boxes_from_text(output, model_key)
    if boxes:
        print(f"\nFound {len(boxes)} beer container(s)")
        return boxes, inference_time
    print("\nNo boxes detected")
    return [], inference_time


def draw_boxes(image_path, boxes, model_name):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size

    try:
        font = ImageFont.truetype("Arial.ttf", 20)
        title_font = ImageFont.truetype("Arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    for i, box in enumerate(boxes, 1):
        bbox = box.get('bbox_2d', [])
        if len(bbox) != 4:
            continue

        x1 = int(bbox[0] / 1000.0 * img_width)
        y1 = int(bbox[1] / 1000.0 * img_height)
        x2 = int(bbox[2] / 1000.0 * img_width)
        y2 = int(bbox[3] / 1000.0 * img_height)

        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = f"Beer {i}"
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle([text_bbox[0]-3, text_bbox[1]-3, text_bbox[2]+3, text_bbox[3]+3], fill=(0, 0, 0))
        draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)

    title_bbox = draw.textbbox((10, 10), model_name, font=title_font)
    draw.rectangle([title_bbox[0]-5, title_bbox[1]-5, title_bbox[2]+5, title_bbox[3]+5], fill=(0, 0, 0))
    draw.text((10, 10), model_name, fill=(255, 255, 0), font=title_font)

    model_suffix = model_name.replace(':', '-').replace('/', '-').replace('@', '-')
    output_path = image_path.replace('.jpg', f'-processed-{model_suffix}.jpg')
    image.save(output_path)
    print(f"\nSaved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BEER DETECTION DEMO (LM Studio)")
    print("="*60)
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {len(MODELS)}")

    results = {}
    for model in MODELS:
        try:
            boxes, inference_time = detect_beer(model, IMAGE_PATH)
            if boxes:
                output_path = draw_boxes(IMAGE_PATH, boxes, model)
                results[model] = {'boxes': len(boxes), 'output': output_path, 'time': inference_time}
            else:
                results[model] = {'boxes': 0, 'output': None, 'time': inference_time}
        except Exception as e:
            print(f"\nError with {model}: {e}")
            results[model] = {'error': str(e)}

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    for model, result in results.items():
        if 'error' in result:
            print(f"{model}: ERROR - {result['error']}")
        else:
            time_str = f"{result['time']:.2f}s" if 'time' in result else "N/A"
            print(f"{model}: {result['boxes']} detections, {time_str}")
    print("="*60)
