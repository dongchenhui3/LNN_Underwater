import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim=9, hidden_dim=256, output_dim=3):
        """
        cGAN Generator.
        Args:
            z_dim: Dimension of random noise vector.
            c_dim: Dimension of condition vector (Acc+Gyro+Vel = 9).
            hidden_dim: LSTM hidden dimension.
            output_dim: Output dimension (Mag noise = 3).
        """
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(z_dim + c_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Tanh limits output to [-1, 1], suitable for normalized data
        self.tanh = nn.Tanh()

    def forward(self, z, c):
        # z: (Batch, Seq, z_dim) - Random noise
        # c: (Batch, Seq, c_dim) - Condition (IMU)
        
        # Concatenate noise and condition
        input_gen = torch.cat([z, c], dim=-1)
        
        # LSTM forward
        output, _ = self.lstm(input_gen)
        
        # Project to output dimension
        out = self.fc(output)
        
        return self.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, c_dim, input_dim=3, hidden_dim=256):
        """
        cGAN Discriminator.
        Args:
            c_dim: Dimension of condition vector (Acc+Gyro = 6).
            input_dim: Input dimension (Mag noise = 3).
            hidden_dim: LSTM hidden dimension.
        """
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim + c_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Adversarial Head: Real (1) vs Fake (0)
        self.adv_head = nn.Linear(hidden_dim, 1)
        
        # Classification Head: Predicts condition or class (Optional, helpful for stability)
        # Here we try to reconstruct the mean condition features as a auxiliary task
        self.aux_head = nn.Linear(hidden_dim, c_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        # x: (Batch, Seq, 3) - Mag noise (Real or Fake)
        # c: (Batch, Seq, 6) - Condition (IMU)
        
        input_dis = torch.cat([x, c], dim=-1)
        
        # LSTM forward
        features, _ = self.lstm(input_dis)
        
        # Take the features from the LAST time step for sequence-level classification
        # Or we can classify every time step. 
        # For sequence generation, typically we discriminate the whole sequence or per-step.
        # User guide implies sequence level, but per-step is often better for time-series details.
        # Let's use per-step discrimination to force fine-grained realism.
        
        validity = self.sigmoid(self.adv_head(features)) # (Batch, Seq, 1)
        aux_pred = self.aux_head(features)               # (Batch, Seq, c_dim)
        
        return validity, aux_pred
