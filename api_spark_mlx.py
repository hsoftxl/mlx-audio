
from mlx_audio.tts.generate import generate_audio

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn
import logging
from mlx_audio.tts.utils import load_model
import uuid
import os
import sys
import numpy as np
import soundfile as sf
from fastrtc import ReplyOnPause, Stream, get_stt_model

from pydantic import BaseModel

# Configure logging
def setup_logging(verbose: bool = False):
    level = logging.DEBUG  if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if verbose:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"

    logging.basicConfig(level=level, format=format_str)
    return logging.getLogger("mlx_audio_server")


logger = setup_logging()  # Will be updated with verbose setting in main()


# Load the model once on server startup.
# You can change the model path or pass arguments as needed.
# For performance, load once globally:
tts_model = None  # Will be loaded when the server starts
audio_player = None  # Will be initialized when the server starts
stt_model = get_stt_model()

app = FastAPI()

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logger.debug(f"Using output folder: {OUTPUT_FOLDER}")


# @app.post("/synthesize")
# async def synthesize(req: SynthesisRequest):


    # return FileResponse(out_path, media_type="audio/wav", filename=out_path)


class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    model: str = "mlx-community/Spark-TTS-0.5B-fp16"

@app.post("/tts")
def tts_endpoint(request: TTSRequest):
    """
    POST an x-www-form-urlencoded form with 'text' (and optional 'voice', 'speed', and 'model').
    We run TTS on the text, save the audio in a unique file,
    and return JSON with the filename so the client can retrieve it.
    """
    global tts_model

    if not request.text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    # Validate speed parameter
    if not (0.5 <= request.speed <= 2.0):
        return JSONResponse(
            {"error": "Speed must be between 0.5 and 2.0"}, status_code=400
        )
    speed_float = request.speed

    # Store current model repo_id for comparison
    current_model_repo_id = (
        getattr(tts_model, "repo_id", None) if tts_model is not None else None
    )

    # Load the model if it's not loaded or if a different model is requested
    if tts_model is None or current_model_repo_id != request.model:
        try:
            logger.debug(f"Loading TTS model from {request.model}")
            tts_model = load_model(request.model)
            logger.debug("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            return JSONResponse(
                {"error": f"Failed to load model: {str(e)}"}, status_code=500
            )

    # Generate the unique filename
    unique_id = str(uuid.uuid4())
    filename = f"tts_{unique_id}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    logger.debug(
        f"Generating TTS for text: '{request.text[:50]}...' with voice: {request.voice}, speed: {speed_float}, model: {request.model}"
    )
    logger.debug(f"Output file will be: {output_path}")

    # We'll use the high-level "model.generate" method:
    results = tts_model.generate(
        text=request.text,
        voice=request.voice,
        speed=speed_float, # Use the validated speed_float from the request
        lang_code=request.voice[0] if request.voice and len(request.voice) > 0 else "z", # Basic lang_code extraction
        sample_rate=16000,
        pitch=1,
        verbose=False,
    )

    # We'll just gather all segments (if any) into a single wav
    # It's typical for multi-segment text to produce multiple wave segments:
    audio_arrays = []
    for segment in results:
        audio_arrays.append(segment.audio)

    # If no segments, return error
    if not audio_arrays:
        logger.error("No audio segments generated")
        return JSONResponse({"error": "No audio generated"}, status_code=500)

    # Concatenate all segments
    cat_audio = np.concatenate(audio_arrays, axis=0)

    # Write the audio as a WAV
    try:
        sf.write(output_path, cat_audio, 16000)
        logger.debug(f"Successfully wrote audio file to {output_path}")

        # Verify the file exists
        if not os.path.exists(output_path):
            logger.error(f"File was not created at {output_path}")
            return JSONResponse(
                {"error": "Failed to create audio file"}, status_code=500
            )

        # Check file size
        file_size = os.path.getsize(output_path)
        logger.debug(f"File size: {file_size} bytes")

        if file_size == 0:
            logger.error("File was created but is empty")
            return JSONResponse(
                {"error": "Generated audio file is empty"}, status_code=500
            )

    except Exception as e:
        logger.error(f"Error writing audio file: {str(e)}")
        return JSONResponse(
            {"error": f"Failed to save audio: {str(e)}"}, status_code=500
        )

    return FileResponse(output_path, media_type="audio/wav", filename=output_path)


if __name__ == "__main__":
    # Setup logging based on a command-line argument or environment variable if needed
    # For now, keeping the default setup_logging() call
    # logger = setup_logging(verbose=True) # Example if you want to enable verbose logging
    logger.info("Starting MLX Audio API server...")
    uvicorn.run(app, host="0.0.0.0", port=8099)