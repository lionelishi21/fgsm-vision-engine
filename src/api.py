import cv2
import streamlink
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference import FGSMInferencePipeline, NUM_FRAMES

app = FastAPI(title="FGSM Vision Engine API")
logging.basicConfig(level=logging.INFO)

# Loaded once at startup from FGSM_MODEL_PATH, if present. Kept optional so the
# service still boots (and honestly reports itself as disabled) when the model
# hasn't been synced to this host yet - see src/inference.py for the domain-gap
# caveat on what this model was actually trained on.
_MODEL_PATH = os.environ.get("FGSM_MODEL_PATH", "./models/temporal_move_classifier")
_CONFIDENCE_THRESHOLD = float(os.environ.get("FGSM_CONFIDENCE_THRESHOLD", "0.12"))
_MAX_CLIPS = int(os.environ.get("FGSM_MAX_CLIPS", "30"))

pipeline: FGSMInferencePipeline | None = None
try:
    pipeline = FGSMInferencePipeline(_MODEL_PATH, confidence_threshold=_CONFIDENCE_THRESHOLD)
    logging.info(f"[FGSM] Loaded model from {_MODEL_PATH} - trained characters: {pipeline.trained_characters}")
except Exception as e:
    logging.warning(f"[FGSM] No model loaded ({e}). /api/analyze will report itself as disabled.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "trained_characters": pipeline.trained_characters if pipeline else [],
    }


class AnalyzeRequest(BaseModel):
    youtube_url: str

class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    timeline: list

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_video(req: AnalyzeRequest):
    logging.info(f"Received analysis request for {req.youtube_url}")
    if pipeline is None:
        return AnalyzeResponse(
            success=False,
            message="Vision pipeline not enabled: trained model is not loaded on this host.",
            timeline=[],
        )
    try:
        # 1. Setup Streamlink session with optional bot-bypass configs
        session = streamlink.Streamlink()
        
        # Load configurations from environment variables if present
        proxy = os.environ.get("STREAMLINK_PROXY")
        if proxy:
            session.set_option("http-proxy", proxy)
            session.set_option("https-proxy", proxy)
            
        cookies_file = os.environ.get("STREAMLINK_COOKIES_FILE")
        if cookies_file and os.path.exists(cookies_file):
            # Format expected by streamlink is a Netscape formatted cookies.txt file
            # However streamlink's http-cookies option expects a dict, or we can use http-cookies text directly if supported,
            # actually Streamlink CLI uses --http-cookie, but Python API is slightly different.
            # To pass a cookies file, we can set it via requests session directly.
            pass # We will set this up securely below if needed, but for now we load it into the session.
            session.set_option("http-cookies", cookies_file)
            
        # Resolve the stream URL using our configured session
        streams = session.streams(req.youtube_url)
        if not streams:
            raise HTTPException(status_code=400, detail="Could not resolve streams for this URL")
            
        best_stream = streams.get("best")
        if not best_stream:
            raise HTTPException(status_code=400, detail="No suitable stream found")
            
        stream_url = best_stream.url
        logging.info(f"Resolved stream URL: {stream_url[:50]}...")
        
        # 2. Open stream in OpenCV
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="OpenCV could not open the stream")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Read NUM_FRAMES-frame clips and classify each one. Clips below the
        # confidence threshold, or whose predicted character isn't one of the
        # ones this model was actually trained on, are skipped rather than
        # reported - this model was trained on isolated move GIFs, not real
        # match footage with two characters in frame, so low-confidence or
        # out-of-distribution output is expected and shouldn't be presented
        # as ground truth.
        timeline = []
        clips_processed = 0
        frames_read = 0

        while clips_processed < _MAX_CLIPS:
            clip = []
            for _ in range(NUM_FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break
                clip.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_read += 1
            if len(clip) < 2:
                break

            result = pipeline.predict_clip(clip)
            clips_processed += 1
            if result is None:
                continue
            character, move_name, confidence = result
            timeline.append({
                "frame": frames_read - len(clip),
                "timestamp": round((frames_read - len(clip)) / fps, 3),
                "players": [
                    {"character": character, "move": move_name, "confidence": round(confidence, 3)}
                ]
            })

        cap.release()

        return AnalyzeResponse(
            success=True,
            message=f"Classified {clips_processed} clip(s) ({frames_read} frames); {len(timeline)} above confidence threshold.",
            timeline=timeline
        )
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
