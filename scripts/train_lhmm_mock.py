import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.lhmm_transformer import LHMMTransformer
from src.data.kinematics_dataset import KinematicsMockDataset

def train():
    print("Initializing LHMM Mock Training...")
    
    # Check for Apple Silicon GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock", "mock_kinematics.json")
    
    try:
        dataset = KinematicsMockDataset(data_path, seq_len=16)
    except FileNotFoundError:
        print("Mock data not found. Please generate it first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    input_dim = 15
    model = LHMMTransformer(input_dim=input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    print("Training Complete. LHMM mock scaffolding successful.")

if __name__ == "__main__":
    train()
