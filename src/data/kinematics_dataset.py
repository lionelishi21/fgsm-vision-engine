import json
import torch
from torch.utils.data import Dataset
import os

class KinematicsMockDataset(Dataset):
    """
    Dataset loader for LHMM that parses the mock_kinematics.json file.
    Splits the 60-frame sequence into contextual windows.
    """
    def __init__(self, json_path, seq_len=16):
        self.seq_len = seq_len
        self.sequences = []
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Could not find mock data at {json_path}")
            
        with open(json_path, "r") as f:
            data = json.load(f)
            
        frames = data.get("frames", [])
        
        # Extract features (root_x, root_y, root_z, left_arm_x, right_arm_x, etc.) for P1
        # For simplicity, we just extract a fixed 15-dimensional vector per frame.
        # [p1_root_x, p1_root_y, p1_root_z, p1_la_x, p1_la_y, p1_la_z, p1_ra_x, p1_ra_y, p1_ra_z, p1_head_x...]
        
        all_features = []
        for frame in frames:
            p1 = frame["p1"]
            root = p1["root_position"]
            la = p1["joints"]["left_arm"]
            ra = p1["joints"]["right_arm"]
            head = p1["joints"]["head"]
            
            # 12 DoF for P1
            feature_vector = root + la + ra + head
            
            # P2
            p2 = frame["p2"]
            p2_root = p2["root_position"]
            
            # Let's say input dim is 15 (12 for P1 + 3 for P2 root)
            feature_vector.extend(p2_root)
            all_features.append(feature_vector)
            
        tensor_data = torch.tensor(all_features, dtype=torch.float32)
        
        # Create rolling windows
        num_frames = len(tensor_data)
        for i in range(num_frames - seq_len):
            window = tensor_data[i : i + seq_len + 1] # +1 to get the target for the last step
            self.sequences.append(window)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        window = self.sequences[idx]
        # Input sequence: frames 0 to seq_len-1
        # Target sequence: frames 1 to seq_len
        x = window[:-1]
        y = window[1:]
        return x, y

if __name__ == "__main__":
    # Test scaffolding
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mock", "mock_kinematics.json")
    dataset = KinematicsMockDataset(data_path, seq_len=16)
    print(f"Dataset loaded with {len(dataset)} sequences.")
    x, y = dataset[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}")
