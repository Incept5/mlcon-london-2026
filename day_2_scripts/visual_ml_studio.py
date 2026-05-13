import requests
import json
import base64
import os
from pathlib import Path

HERE = Path(__file__).parent


def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None


def send_request(model, prompt, image_path=None):
    url = "http://localhost:1234/v1/chat/completions"

    user_content = [{"type": "text", "text": prompt}]

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }

        mime_type = mime_types.get(image_ext, 'image/jpeg')

        base64_image = encode_image_to_base64(image_path)
        if base64_image is None:
            raise Exception(f"Failed to encode image: {image_path}")

        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful visual assistant, give detailed and accurate answers."
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")


def main():
    data_dir = HERE / "data"
    images = [str(data_dir / f"IMG_{i}.jpg") for i in range(1, 13)]

    models = [
        "qwen3.5-4b",
    ]

    questions = [
        {"prompt": "Please output the equation for Markdown",
         "image": images[11],
         },
    ]

    for model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Testing model: {model_name}")
        print(f"{'=' * 60}")

        for i, question in enumerate(questions):
            prompt = question["prompt"]
            image_path = question["image"]

            print(f"\nQuestion {i + 1}: {prompt}")
            print(f"Image: {os.path.basename(image_path)}")
            print("-" * 40)

            try:
                response = send_request(model_name, prompt, image_path)

                if 'choices' in response and len(response['choices']) > 0:
                    content = response['choices'][0]['message']['content']
                    print(f"Response: {content}")
                else:
                    print("Error: Unexpected response format")
                    print(f"Raw response: {response}")

            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
