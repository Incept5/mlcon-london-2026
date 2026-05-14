import ollama
from ollama import Options
from mlx_audio.tts.generate import generate_audio

NAME = "John"
STAR_SIGN = "Scorpio"
TODAY = "Thursday, 14-05-2026"
LLM = "qwen3.5:4b"

SYSTEM_PROMPT = (
    "You are Morgan Freeman narrating a brief astrological reflection for tomorrow. "
    "Speak in two or three short sentences, calm and philosophical, with a gentle "
    "sense of wonder. Return plain prose only — no markdown, no headings, no lists, "
    "no stage directions, no quotation marks. Use British English."
)

instruction = (
    f"Give a short, philosophical prediction for tomorrow for {NAME}, "
    f"whose star sign is {STAR_SIGN}. Today is {TODAY}."
)

response = ollama.chat(
    model=LLM,
    think=False,
    stream=False,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ],
    options=Options(temperature=0.8, num_ctx=4096, top_p=0.95, top_k=40, num_predict=200),
)

prediction = response.message.content.strip()
print(prediction)

generate_audio(
    text=prediction,
    model="mlx-community/chatterbox-turbo-fp16",
    ref_audio="morgan-freeman-voice-sample.wav",
    file_prefix="morgan_astrology",
    audio_format="wav",
    play=True,
)
