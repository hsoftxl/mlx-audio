
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
import threading # Add threading import
import mlx.core as mx # Add mlx.core import
import gc # Add gc import

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
tts_model_lock = threading.Lock() # Initialize the lock

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

async def _generate_tts_audio(text: str, voice: str, speed: float, model_name: str):
    """Helper function to handle TTS generation, model loading/unloading, and file saving."""
    global tts_model

    if not text.strip():
        return JSONResponse({"error": "Text is empty"}, status_code=400)

    if not (0.5 <= speed <= 2.0):
        return JSONResponse(
            {"error": "Speed must be between 0.5 and 2.0"}, status_code=400
        )

    unique_id = str(uuid.uuid4())
    filename = f"tts_{unique_id}.wav"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    with tts_model_lock:
        if tts_model is not None:
            logger.debug(f"Unloading existing TTS model: {getattr(tts_model, 'repo_id', 'unknown')} to ensure fresh instance for every request.")
            old_model_to_remove = tts_model
            tts_model = None 
            del old_model_to_remove
            mx.clear_cache()
            logger.debug("MLX cache cleared during aggressive model unload.")
            gc.collect()
            mx.synchronize()
            logger.debug("Old TTS model presumed unloaded and GPU synchronized after cache clear and GC.")

        try:
            logger.debug(f"Loading TTS model from {model_name} for current request.")
            loaded_model_instance = load_model(model_name)
            tts_model = loaded_model_instance
            logger.debug("TTS model loaded successfully for current request.")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            tts_model = None
            return JSONResponse(
                {"error": f"Failed to load model: {str(e)}"}, status_code=500
            )
        
        logger.debug(
            f"Generating TTS for text: '{text[:50]}...' with voice: {voice}, speed: {speed}, model: {model_name}"
        )
        logger.debug(f"Output file will be: {output_path}")

        results = tts_model.generate(
            text=text,
            voice=voice,
            speed=speed, 
            lang_code=voice[0] if voice and len(voice) > 0 else "z",
            sample_rate=16000,
            pitch=1,
            verbose=False
        )

           # ref_audio="1.wav",
          #  ref_text="非要说的话,只能归结于理念不和吧",
        evaluated_mlx_chunks = []
        try:
            for segment_obj in results:
                chunk = segment_obj.audio
                mx.eval(chunk)
                evaluated_mlx_chunks.append(chunk)
        finally:
            if hasattr(results, 'close'):
                results.close()
            if 'results' in locals():
                del results

        if not evaluated_mlx_chunks:
            logger.error("No audio segments generated")
            gc.collect()
            mx.synchronize()
            return JSONResponse({"error": "No audio generated"}, status_code=500)

        mx.synchronize() 
        numpy_audio_segments = [np.array(chunk) for chunk in evaluated_mlx_chunks]
        del evaluated_mlx_chunks
        cat_audio = np.concatenate(numpy_audio_segments, axis=0)
        del numpy_audio_segments
        gc.collect() 
        mx.synchronize()

        try:
            sf.write(output_path, cat_audio, 16000)
            logger.debug(f"Successfully wrote audio file to {output_path}")
            if not os.path.exists(output_path):
                logger.error(f"File was not created at {output_path}")
                if 'cat_audio' in locals(): del cat_audio
                return JSONResponse({"error": "Failed to create audio file"}, status_code=500)
            file_size = os.path.getsize(output_path)
            logger.debug(f"File size: {file_size} bytes")
            if file_size == 0:
                logger.error("File was created but is empty")
                if 'cat_audio' in locals(): del cat_audio
                return JSONResponse({"error": "Generated audio file is empty"}, status_code=500)
        except Exception as e:
            logger.error(f"Error writing audio file: {str(e)}")
            if 'cat_audio' in locals(): del cat_audio
            return JSONResponse({"error": f"Failed to save audio: {str(e)}"}, status_code=500)
        finally:
            if 'cat_audio' in locals(): del cat_audio

    return FileResponse(output_path, media_type="audio/wav", filename=filename)

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    """
    POST endpoint for TTS, calls the shared generation logic.
    """
    return await _generate_tts_audio(request.text, request.voice, request.speed, request.model)

@app.get("/tts")
async def tts_get_endpoint(
    text: str = Query(..., description="The text to synthesize."),
    voice: str = Query("af_heart", description="Voice to use for synthesis."),
    speed: float = Query(1.0, ge=0.5, le=2.0, description="Speed of the speech (0.5 to 2.0)."),
    model: str = Query("mlx-community/Spark-TTS-0.5B-fp16", description="TTS model to use.")
):
    """
    GET endpoint for TTS, calls the shared generation logic.
    """
    return await _generate_tts_audio(text, voice, speed, model)


if __name__ == "__main__":
    # Setup logging based on a command-line argument or environment variable if needed
    # For now, keeping the default setup_logging() call
    # logger = setup_logging(verbose=True) # Example if you want to enable verbose logging
    logger.info("Starting MLX Audio API server...")
    uvicorn.run(app, host="0.0.0.0", port=8099)