from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import os
import soundfile as sf
from datetime import datetime

from pathlib import Path

app = FastAPI()

# 初始化模型
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL = "mlx-community/Spark-TTS-0.5B-fp16"

DEFAULT_PROMPT_PATH = Path("2.wav")

class SynthesisRequest(BaseModel):
    text: str
    prompt_text: str = None
    gender: str = None
    pitch: str = None
    speed: str = None
    prompt_path: str = None

@app.post("/synthesize")
async def synthesize(req: SynthesisRequest):
    # 路径处理
    prompt_audio_path = req.prompt_path or DEFAULT_PROMPT_PATH
    if not os.path.exists(prompt_audio_path):
        return {"error": f"Prompt audio not found: {prompt_audio_path}"}

    # 推理
    with torch.no_grad():
        wav = MODEL.inference(
            text=req.text,
            prompt_speech_path=prompt_audio_path,
            prompt_text=req.prompt_text,
            gender=req.gender,
            pitch=req.pitch,
            speed=req.speed,
        )

    # 保存输出
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = f"output/output_{timestamp}.wav"
    sf.write(out_path, wav, samplerate=16000)

    return FileResponse(out_path, media_type="audio/wav", filename=out_path)


@app.get("/synthesize")
async def synthesize_get(
    text: str = Query(..., description="The text to synthesize."),
    prompt_text: str = Query(None, description="Additional prompt text."),
    gender: str = Query(None, description="Gender for voice."),
    pitch: str = Query(None, description="Pitch for voice."),
    speed: str = Query(None, description="Speed for voice."),
    prompt_path: str = Query(None, description="Path to the prompt audio file."),
):
    # 路径处理
    prompt_audio_path = prompt_path or DEFAULT_PROMPT_PATH
    print(f"Path : {prompt_audio_path}")
    if not os.path.exists(prompt_audio_path):
        return {"error": f"Prompt audio not found: {prompt_audio_path}"}

    # 推理
    with torch.no_grad():
        wav = MODEL.inference(
            text=text,
            prompt_speech_path=prompt_audio_path,
            prompt_text=prompt_text,
            gender=None,
            pitch=pitch,
            speed=speed,
        )

    # 保存输出
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = f"output/output_{timestamp}.wav"
    sf.write(out_path, wav, samplerate=16000)

    return FileResponse(out_path, media_type="audio/wav", filename=out_path)
