from pathlib import Path
import lmstudio as lms

HERE = Path(__file__).parent
IMAGE = str(HERE / "data" / "RTS1UI9-1024x659.jpg")

model = lms.llm("qwen3.5-4b")
image = lms.prepare_image(IMAGE)

chat = lms.Chat()
chat.add_user_message("Describe this image in one sentence.", images=[image])

result = model.respond(chat)
print(result.content if hasattr(result, "content") else result)
