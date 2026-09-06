from fastapi import FastAPI, HTTPException
import json
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FGSM Mock API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock", "mock_kinematics.json")

@app.get("/api/v1/kinematics/{sequence_id}")
async def get_kinematics(sequence_id: str):
    """
    Returns mock SMPL kinematic data for the What-If visualizer.
    """
    try:
        with open(MOCK_DATA_PATH, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Mock data not found. Run generate_mock_kinematics.py first.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
