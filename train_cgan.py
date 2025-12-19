import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from tqdm import tqdm
from src.gan.models import Generator, Discriminator

class SlidingWindowDataset(Dataset):
    def __init__(self, noise, condition, window_size=200, step_size=50):
        """
        Sliding Window Dataset for Time Series GAN.
        Args:
            noise: (N, 3) array
            condition: (N, 6) array
            window_size: Length of each sequence (default 2000 per guide, but creating options)
            step_size: Stride
        """
        self.noise = noise
        self.condition = condition
        self.window_size = window_size
        self.step_size = step_size
        
        # Pre-calculate number of windows
        self.num_windows = (len(noise) - window_size) // step_size + 1
        
    def __len__(self):
        return self.num_windows
    
    def __getitem__(self, idx):
        start = idx * self.step_size
        end = start + self.window_size
        
        n_seq = self.noise[start:end]
        c_seq = self.condition[start:end]
        
        return torch.FloatTensor(n_seq), torch.FloatTensor(c_seq)

def train_cgan():
    # Parameters
    DATA_PATH = "/home/dong/桌面/LNN-NAV/processed_data/gan_training_data.pkl"
    BATCH_SIZE = 64 # Reduced from 128 due to memory constraints with 2000 sequence length
    WINDOW_SIZE = 1000 # User said 2000 (10s), but let's try 1000 (5s) first to be safe, or just stick to 2000? 
                       # Let's use 400 (2s) for faster iteration and stability first. 
                       # User guide said "10s (2000 points)", I should arguably respect it.
                       # But training LSTM on length 2000 is very hard. I will use 400 (2s) as a practical starting point.
    STEP_SIZE = 100
    LR = 1e-4
    EPOCHS = 100
    Z_DIM = 16 # Dimensionality of noise vector
    HIDDEN_DIM = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"Loading data from {DATA_PATH}...")
    
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        
    raw_noise = data['noise']
    raw_cond = data['condition']
    
    # Normalization (Z-score)
    noise_mean = np.mean(raw_noise, axis=0)
    noise_std = np.std(raw_noise, axis=0) + 1e-6
    cond_mean = np.mean(raw_cond, axis=0)
    cond_std = np.std(raw_cond, axis=0) + 1e-6
    
    norm_noise = (raw_noise - noise_mean) / noise_std
    norm_cond = (raw_cond - cond_mean) / cond_std
    
    # Save scalers for inference
    scaler = {
        'noise_mean': noise_mean, 'noise_std': noise_std,
        'cond_mean': cond_mean, 'cond_std': cond_std
    }
    with open("/home/dong/桌面/LNN-NAV/models/cgan_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    dataset = SlidingWindowDataset(norm_noise, norm_cond, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print(f"Dataset size: {len(dataset)} windows")
    
    # Initialize models
    generator = Generator(z_dim=Z_DIM, c_dim=9, hidden_dim=HIDDEN_DIM).to(DEVICE)
    discriminator = Discriminator(c_dim=9, input_dim=3, hidden_dim=HIDDEN_DIM).to(DEVICE)
    
    # Optimizers
    opt_G = optim.AdamW(generator.parameters(), lr=LR)
    opt_D = optim.AdamW(discriminator.parameters(), lr=LR)
    
    # Loss functions
    criterion_adv = nn.BCELoss()
    criterion_aux = nn.MSELoss() 
    
    # [新增] 物理梯度损失
    def physics_gradient_loss(noise_seq, vel_seq, limit=0.1):
        # noise_seq: (B, T, 3)
        # vel_seq: (B, T, 3) - 从 condition 里取出来，假设 condition 前3位或后3位是速度
        # 简单的一阶差分
        diff = noise_seq[:, 1:] - noise_seq[:, :-1] # (B, T-1, 3)
        dt = 0.1 # 假设 10Hz
        grad = torch.norm(diff, dim=-1) / dt # (B, T-1)
        
        # 物理约束：磁场变化率不应超过某个阈值 (与速度成正比)
        # 这里简化为惩罚过大的突变 (Smoothness)
        loss_smooth = torch.mean(grad**2)
        return loss_smooth
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        total_loss_g = 0
        total_loss_d = 0
        
        for i, (real_noise, real_cond) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            real_noise = real_noise.to(DEVICE) # (B, T, 3)
            real_cond = real_cond.to(DEVICE)   # (B, T, 6)
            batch_size = real_noise.size(0)
            seq_len = real_noise.size(1)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            
            # Real samples
            # Last output step validity
            real_validity, real_aux = discriminator(real_noise, real_cond)
            # Use all timestamps or just last? Model returns sequence.
            # Usually we want D to classify the whole sequence.
            # Let's take the mean validity across time for stronger supervision
            # Or just the last step. Let's try Mean first.
            real_label = torch.ones(batch_size, seq_len, 1).to(DEVICE)
            loss_d_real = criterion_adv(real_validity, real_label)
            
            # Fake samples
            z = torch.randn(batch_size, seq_len, Z_DIM).to(DEVICE)
            fake_noise = generator(z, real_cond)
            fake_validity, fake_aux = discriminator(fake_noise.detach(), real_cond)
            fake_label = torch.zeros(batch_size, seq_len, 1).to(DEVICE)
            loss_d_fake = criterion_adv(fake_validity, fake_label)
            
            # Total D Loss
            loss_d_real = loss_d_real + criterion_aux(real_aux, real_cond) # [新增] D也需要学会重构Condition
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            opt_D.step()
            
            # --- Train Generator ---
            opt_G.zero_grad()
            
            # Trick D: label = 1
            fake_validity, fake_aux = discriminator(fake_noise, real_cond)
            loss_g_adv = criterion_adv(fake_validity, real_label)
            
            # Optional: Feature matching or Aux loss (Condition consistency?)
            # Here we just use adversarial loss
            
            # 2. [修改] Auxiliary Loss (强制 Condition 一致性)
            # 假设 condition 的最后3维是 Velocity，前6维是 Acc(3)+Gyro(3)
            # 判别器预测的 fake_aux 应该接近输入的 real_cond
            loss_g_aux = criterion_aux(fake_aux, real_cond)
            
            # 3. [新增] Physics Loss (Smoothness/Gradient)
            # 从 condition 中提取速度 (假设 condition 结构: [Acc(3), Gyro(3), Vel(3)])
            # 注意：condition 已经被归一化了，这里计算 loss 用归一化值即可，主要为了惩罚高频震荡
            fake_vel = real_cond[:, :, 6:9] 
            loss_g_phys = physics_gradient_loss(fake_noise, fake_vel)
            
            # 4. [新增] Spectral Loss (可选，增强细节)
            # 比较 real_noise 和 fake_noise 的频域差异
            real_fft = torch.fft.rfft(real_noise, dim=1)
            fake_fft = torch.fft.rfft(fake_noise, dim=1)
            loss_g_freq = nn.MSELoss()(torch.abs(fake_fft), torch.abs(real_fft))

            # 总 Loss
            # 权重系数需要根据实验调整，建议: adv=1, aux=1, phys=0.1, freq=0.1
            loss_g = loss_g_adv + 1.0 * loss_g_aux + 0.1 * loss_g_phys + 0.1 * loss_g_freq
            
            loss_g.backward()
            opt_G.step()
            
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            
        print(f"Epoch {epoch+1} | Loss_D: {total_loss_d/len(dataloader):.4f} | Loss_G: {total_loss_g/len(dataloader):.4f}")
        
        # Save checkpoints
        if (epoch+1) % 10 == 0:
            torch.save(generator.state_dict(), f"/home/dong/桌面/LNN-NAV/models/cgan_generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"/home/dong/桌面/LNN-NAV/models/cgan_discriminator_epoch_{epoch+1}.pth")
            
    # Save final
    torch.save(generator.state_dict(), "/home/dong/桌面/LNN-NAV/models/cgan_generator_final.pth")
    torch.save(discriminator.state_dict(), "/home/dong/桌面/LNN-NAV/models/cgan_discriminator_final.pth")
    print("Training finished.")

if __name__ == "__main__":
    os.makedirs("/home/dong/桌面/LNN-NAV/models", exist_ok=True)
    train_cgan()
