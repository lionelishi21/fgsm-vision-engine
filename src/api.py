import cv2
import streamlink
import logging
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Mocking the pipeline for the API file to ensure it can start without downloading heavy models immediately
# In a real run, this would be:
# from src.inference_v2 import FGSMInferencePipelineV2
# pipeline = FGSMInferencePipelineV2(...)

app = FastAPI(title="FGSM Vision Engine API")
logging.basicConfig(level=logging.INFO)

class AnalyzeRequest(BaseModel):
    youtube_url: str
    
class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    timeline: list

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_video(req: AnalyzeRequest):
    logging.info(f"Received analysis request for {req.youtube_url}")
    # The loop below returns placeholder detections, not model output. Callers
    # treat this timeline as ground truth, so only serve it when explicitly
    # running in mock mode for local development.
    if os.environ.get("FGSM_MOCK") != "true":
        return AnalyzeResponse(
            success=False,
            message="Vision pipeline not enabled: trained models are not wired into this service yet.",
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
            
        # For demonstration: process the first 30 frames (1 second at 30fps)
        frames_processed = 0
        max_frames = 30
        timeline = []
        
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # In real implementation: 
            # results = pipeline.process_frame(frames_processed, 30, [...detections...])
            # timeline.append(results)
            
            # Mocking the results for the API test
            timeline.append({
                "frame": frames_processed,
                "timestamp": round(frames_processed / 30.0, 3),
                "players": [
                    {"character": "ryu", "move": "standing_heavy_punch", "move_phase": "startup"}
                ]
            })
            
            frames_processed += 1
            
        cap.release()
        
        return AnalyzeResponse(
            success=True,
            message=f"Processed {frames_processed} frames successfully from stream.",
            timeline=timeline
        )
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
