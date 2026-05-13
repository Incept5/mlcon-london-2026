import requests
import json
import base64
import os


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
    url = "http://localhost:11434/api/chat"

    images = []
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        base64_image = encode_image_to_base64(image_path)
        if base64_image is None:
            raise Exception(f"Failed to encode image: {image_path}")

        images.append(base64_image)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful visual assistant, give detailed and accurate answers."
            },
            {
                "role": "user",
                "content": prompt,
                "images": images
            }
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 1000
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {str(e)}")


def main():
    images = [
        "data/IMG_1.jpg",
        "data/IMG_2.jpg",
        "data/IMG_3.jpg",
        "data/IMG_4.jpg",
        "data/IMG_5.jpg",
        "data/IMG_6.jpg",
        "data/IMG_7.jpg",
        "data/IMG_8.jpg",
        "data/IMG_9.jpg",
        "data/IMG_10.jpg",
        "data/IMG_11.jpg",
        "data/IMG_12.jpg",
    ]

    models = [
        "qwen3.5:4b",
    ]

    questions = [
        {
            "prompt": "Using the graph, what is the Read noise for gain=50?",
            "image": images[0],
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

                if 'message' in response:
                    message = response['message']

                    if 'content' in message:
                        content = message['content']
                        if content:
                            print(f"Response: {content}")
                        else:
                            if 'thinking' in message and message['thinking']:
                                print("Response: (empty - model spent all tokens on thinking, consider increasing token limit)")
                            else:
                                print("Response: (empty)")
                else:
                    print("Error: Unexpected response format")
                    print(f"Raw response: {response}")

            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
