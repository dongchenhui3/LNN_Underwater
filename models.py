import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, z_dim, c_dim=9, hidden_dim=512, output_dim=3):
        """
        cGAN Generator.
        Args:
            z_dim: Dimension of random noise vector.
            c_dim: Dimension of condition vector (Acc+Gyro+Vel = 9).
            hidden_dim: LSTM hidden dimension.
            output_dim: Output dimension (Mag noise = 3).
        """
        super(Generator, self).__init__()
        # [修改] 容量增加 + BN + LeakyReLU
        # Layer 1: LSTM (Captures temporal dependencies)
        # Using 512 hidden units
        self.lstm = nn.LSTM(z_dim + c_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Layer 2: MLP with BN and LeakyReLU (Enhanced capacity)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024), # BN usually requires (Batch, C) or (Batch, C, Seq) -> Transpose needed
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, output_dim)
        )
        
        # Tanh limits output to [-1, 1], suitable for normalized data
        self.tanh = nn.Tanh()

    def forward(self, z, c):
        # z: (Batch, Seq, z_dim)
        # c: (Batch, Seq, c_dim)
        
        batch_size, seq_len, _ = z.size()
        
        # Concatenate noise and condition
        input_gen = torch.cat([z, c], dim=-1)
        
        # LSTM forward
        # output: (Batch, Seq, hidden_dim)
        output, _ = self.lstm(input_gen)
        
        # MLP processing (Need to flatten or iterate)
        # Reshape to (Batch * Seq, Hidden) for MLP
        out_flat = output.contiguous().view(batch_size * seq_len, -1)
        
        # MLP output: (Batch * Seq, Output_Dim)
        out_flat = self.mlp(out_flat)
        
        # Reshape back to sequence
        out = out_flat.view(batch_size, seq_len, -1)
        
        return self.tanh(out)

class Discriminator(nn.Module):
    def __init__(self, c_dim, input_dim=3, hidden_dim=256):
        """
        cGAN Discriminator.
        Args:
            c_dim: Dimension of condition vector.
            input_dim: Input dimension.
            hidden_dim: LSTM hidden dimension.
        """
        super(Discriminator, self).__init__()
        
        # [修改] 增加 Dropout, LeakyReLU
        self.lstm = nn.LSTM(input_dim + c_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        
        # Adversarial Head: Real (1) vs Fake (0)
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        # Auxiliary Head: Reconstruct Condition
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, c_dim)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, c):
        # x: (Batch, Seq, 3)
        # c: (Batch, Seq, 6)
        
        input_dis = torch.cat([x, c], dim=-1)
        
        # LSTM forward
        features, _ = self.lstm(input_dis)
        
        # Heads
        validity = self.sigmoid(self.adv_head(features)) # (Batch, Seq, 1)
        aux_pred = self.aux_head(features)               # (Batch, Seq, c_dim)
        
        return validity, aux_pred
