import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class LHMMTransformer(nn.Module):
    """
    Large Humanoid Movement Model (LHMM)
    An autoregressive Transformer Decoder for motion prediction.
    """
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Project raw joint kinematics to embedding space
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Decoder (using TransformerEncoder since we will apply causal mask manually)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project back to joint space
        self.output_proj = nn.Linear(d_model, input_dim)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates a causal mask for autoregressive training."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, input_dim]
        Returns:
            predicted_x: Tensor, shape [batch_size, seq_len, input_dim]
        """
        device = x.device
        seq_len = x.size(1)
        
        # Causal mask to prevent looking ahead
        mask = self.generate_square_subsequent_mask(seq_len, device)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer pass
        out = self.transformer(x, mask=mask, is_causal=True)
        
        # Project back to original kinematic space
        predicted = self.output_proj(out)
        return predicted

if __name__ == "__main__":
    # Test scaffolding
    model = LHMMTransformer(input_dim=15)
    dummy_input = torch.randn(2, 16, 15) # Batch of 2, seq len 16, 15 DoF
    output = model(dummy_input)
    print(f"Model output shape: {output.shape} (Expected: 2, 16, 15)")
