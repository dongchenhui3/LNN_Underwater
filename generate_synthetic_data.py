import torch
import numpy as np
import pickle
import os
from scipy.spatial.transform import Rotation as R
from src.gan.models import Generator

def generate_synthetic_data(num_variations=5):
    # Paths
    MODEL_PATH = "/home/dong/桌面/LNN-NAV/models/cgan_generator_final.pth"
    SCALER_PATH = "/home/dong/桌面/LNN-NAV/models/cgan_scaler.pkl"
    DATA_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_train_full.pkl"
    OUTPUT_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_mag_augmented.pkl"
    
    DEVICE = torch.device('cpu') # Force CPU to avoid cuDNN non-contiguous errors during inference
    Z_DIM = 16
    HIDDEN_DIM = 128
    
    print(f"Loading scalers from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    noise_mean = torch.FloatTensor(scaler['noise_mean']).to(DEVICE)
    noise_std = torch.FloatTensor(scaler['noise_std']).to(DEVICE)
    cond_mean = torch.FloatTensor(scaler['cond_mean']).to(DEVICE)
    cond_std = torch.FloatTensor(scaler['cond_std']).to(DEVICE)
    
    # (Model loading moved to after data loading to ensure correct config)    
    # Load data from ALL sources
    TRAIN_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_train_full.pkl"
    TEST02_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_test_02.pkl"
    TEST07_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_test_07.pkl"
    
    X_total = []
    Y_quat_total = []
    Y_vel_body_total = []
    
    files = [TRAIN_PATH, TEST02_PATH, TEST07_PATH]
    for p in files:
        if not os.path.exists(p): continue
        with open(p, 'rb') as f:
            data = pickle.load(f)
            if 'X_list' in data:
                X_total.extend(data['X_list'])
                Y_quat_total.extend(data['Y_quat_list'])
                Y_vel_body_total.extend(data['Y_vel_body_list'])
            elif 'X' in data:
                X_total.append(data['X'])
                Y_quat_total.append(data['Y_quat'])
                # Handle VelB key variations
                if 'Y_velb' in data: vel = data['Y_velb']
                elif 'Y_vel_body' in data: vel = data['Y_vel_body']
                else: vel = np.zeros((len(data['X']), 3)) # Fallback
                Y_vel_body_total.append(vel)
    
    # Use re-estimated B_ideal from all sequences
    # We can just copy the value from prepare_gan_data output or load from gan_training_data.pkl
    # For robustnes, load from gan_training_data
    with open("/home/dong/桌面/LNN-NAV/processed_data/gan_training_data.pkl", "rb") as f:
        gan_data = pickle.load(f)
        B_ideal_vec = gan_data['B_ideal']
        
        # [新增] 读取校准参数
        calib_params = gan_data.get('calibration', None)
        if calib_params:
            b_hard = calib_params['b_hard']
            W_soft = calib_params['W_soft']
            print(f"Loaded calibration params: b_hard={b_hard}, W_soft diag={np.diag(W_soft)}")
        else:
            # 兼容旧数据
            b_hard = np.zeros(3)
            W_soft = np.eye(3)
            print("Warning: No calibration params found, using identity.")
        
    print(f"B_ideal loaded: {B_ideal_vec}")
    
    # ... (Generator instantiation with c_dim=9 is assumed from previous edits) ...
    # But we need to ensure we pass c_dim=9 here if we re-instantiate
    generator = Generator(z_dim=Z_DIM, c_dim=9, hidden_dim=HIDDEN_DIM).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print("Model file not found!")
        return
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()

    aug_X_list = []
    aug_Y_quat_list = []
    aug_Y_vel_body_list = []
    
    print(f"Generating {num_variations} variations per sequence...")
    
    with torch.no_grad():
        for i in range(len(X_total)):
            raw_X = X_total[i]
            quat_gt = Y_quat_total[i]
            vel_body = Y_vel_body_total[i]
            
            length = min(len(raw_X), len(quat_gt), len(vel_body))
            raw_X = raw_X[:length]
            quat_gt = quat_gt[:length]
            vel_body = vel_body[:length]
            
            # Extract Condition: Acc(3), Gyro(3), Vel(3)
            acc = raw_X[:, 0:3]
            gyro = raw_X[:, 3:6]
            # Vel is from Y_vel_body_total
            
            cond_np = np.concatenate([acc, gyro, vel_body], axis=1) # (T, 9)
            
            # Normalize Condition
            cond_tensor = torch.FloatTensor(cond_np).to(DEVICE)
            norm_cond = (cond_tensor - cond_mean) / cond_std
            norm_cond = norm_cond.unsqueeze(0) # (1, T, 9)
            
            # Calculate Base Ideal Mag (Body Frame)
            r = R.from_quat(quat_gt)
            B_body_ideal = r.inv().apply(B_ideal_vec) # (T, 3)
            
            # Repeat generation
            for v in range(num_variations):
                batch_size = 1
                seq_len = length
                z = torch.randn(batch_size, seq_len, Z_DIM).to(DEVICE)
                
                syn_noise_norm = generator(z, norm_cond)
                syn_noise_norm = syn_noise_norm.squeeze(0)
                
                syn_noise = (syn_noise_norm * noise_std + noise_mean).cpu().numpy()
                
                # [修改] 物理场重构
                # 1. 理想环境场 (World Frame) -> Body Frame
                r = R.from_quat(quat_gt)
                B_body_ideal = r.inv().apply(B_ideal_vec) # (T, 3)
                
                # 2. 加上 GAN 生成的环境动态噪声
                mag_clean_body = B_body_ideal + syn_noise
                
                # 3. [新增] 加上 AUV 自身的 Hard/Soft Iron (逆校准)
                # 模拟真实的传感器读数：m_raw = W^-1 * m_clean + b
                W_inv = np.linalg.inv(W_soft)
                mag_final_raw = (mag_clean_body @ W_inv.T) + b_hard
                
                X_aug = raw_X.copy()
                X_aug[:, 6:9] = mag_final_raw # 存入的是模拟的 Raw Data
                
                aug_X_list.append(X_aug)
                aug_Y_quat_list.append(quat_gt)
                aug_Y_vel_body_list.append(vel_body) # Same velocity for augmented data
                
    print(f"Original sequences: {len(X_total)}")
    print(f"Augmented sequences: {len(aug_X_list)}")
    
    output_data = {
        'X_list': X_total + aug_X_list,
        'Y_quat_list': Y_quat_total + aug_Y_quat_list,
        'Y_vel_body_list': Y_vel_body_total + aug_Y_vel_body_list,
        'description': f"All 7 Real + {num_variations}x cGAN Augmented Data (With Velocity)"
    }
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"Saved augmented dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()
