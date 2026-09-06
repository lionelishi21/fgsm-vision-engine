import json
import math
import os

def generate_mock_smpl_data(num_frames=60):
    """
    Generates mock SMPL-like 3D kinematic data for two characters (P1 and P2).
    Simulates a simple 60 FPS interaction (e.g., a Hadoken animation).
    """
    frames = []
    for i in range(num_frames):
        # P1 (Attacker) doing a forward motion
        p1_x = -1.0 + (i / num_frames) * 0.2
        # P2 (Defender) taking a hit around frame 30
        p2_x = 1.0 + (max(0, i - 30) / 30) * 0.5
        
        # Simple arm motion for P1 (raising arms)
        p1_arm_angle = math.sin(i * 0.1) * 1.5
        
        frame_data = {
            "frame_idx": i,
            "timestamp_ms": int((i / 60.0) * 1000),
            "p1": {
                "character": "ryu",
                "root_position": [p1_x, 0.0, 0.0],
                "joints": {
                    "left_arm": [p1_arm_angle, 0.0, 0.0],
                    "right_arm": [p1_arm_angle, 0.0, 0.0],
                    # Mock other joints with 0s
                    "head": [0.0, 0.0, 0.0],
                    "left_leg": [0.0, 0.0, 0.0],
                    "right_leg": [0.0, 0.0, 0.0]
                },
                "status": "active" if i < 40 else "recovery"
            },
            "p2": {
                "character": "ken",
                "root_position": [p2_x, 0.0, 0.0],
                "joints": {
                    "left_arm": [0.0, 0.0, 0.0],
                    "right_arm": [0.0, 0.0, 0.0],
                    "head": [-0.5 if i > 30 else 0.0, 0.0, 0.0], # Head snaps back
                    "left_leg": [0.0, 0.0, 0.0],
                    "right_leg": [0.0, 0.0, 0.0]
                },
                "status": "idle" if i <= 30 else "hitstun"
            }
        }
        frames.append(frame_data)
        
    return {
        "metadata": {
            "fps": 60,
            "total_frames": num_frames,
            "sequence_name": "mock_hadoken_punish"
        },
        "frames": frames
    }

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock")
    os.makedirs(output_dir, exist_ok=True)
    
    mock_data = generate_mock_smpl_data(60)
    output_path = os.path.join(output_dir, "mock_kinematics.json")
    
    with open(output_path, "w") as f:
        json.dump(mock_data, f, indent=2)
        
    print(f"Generated mock kinematics data at {output_path}")
