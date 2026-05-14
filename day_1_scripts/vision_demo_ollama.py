import ollama

response = ollama.generate(
    model="qwen3.5:4b",
    prompt="Describe this image and guess where it might have been taken.",
    images=["image.jpg"],
    think=False,
)
print(response["response"])
