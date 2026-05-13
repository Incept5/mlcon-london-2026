from pathlib import Path
from mlx_audio.tts.generate import generate_audio

HERE = Path(__file__).parent

generate_audio(
    text="In the spring of 2026, a quiet gathering took place in Amsterdam. They called it M L Con. And on that Tuesday morning, as the light came through the windows, everyone in the room understood that something remarkable was about to unfold.",
    model="mlx-community/chatterbox-turbo-fp16",
    ref_audio=str(HERE / "data" / "morgan-freeman-voice-sample.wav"),
    output_path=str(HERE / "output"),
    file_prefix="demo",
    play=True,
)
