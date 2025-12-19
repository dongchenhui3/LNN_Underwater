import pickle
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# [新增] 简化的磁力计校准函数 (基于最小二乘法的椭球拟合)
def ellipsoidal_calibration(mag_data):
    # mag_data: (N, 3)
    # 这是一个简化的代数解法，用于估算 Hard Iron (bias)
    # 对于更精细的 Soft Iron (W_matrix)，建议使用专门的库如 Magpylib 或单独的校准脚本
    # 这里为了代码简洁，演示 Hard Iron 移除 + 标度归一化
    
    # 1. 构建矩阵 A x = b
    x, y, z = mag_data[:,0], mag_data[:,1], mag_data[:,2]
    # D = np.array([x**2, y**2, z**2, 2*y*z, 2*x*z, 2*x*y, 2*x, 2*y, 2*z, np.ones_like(x)]).T
    # ... (SVD解法较为复杂，这里提供一个工程常用的"最大最小值中心法"作为替代，
    #      如果您的数据覆盖了足够的旋转空间，这就足够了)
    
    min_xyz = np.min(mag_data, axis=0)
    max_xyz = np.max(mag_data, axis=0)
    
    b_hard = (max_xyz + min_xyz) / 2
    scale_xyz = (max_xyz - min_xyz) / 2
    avg_scale = np.mean(scale_xyz)
    
    # 简单的 Soft Iron 修正矩阵 (仅对角线缩放)
    W_soft = np.diag(avg_scale / scale_xyz)
    
    return b_hard, W_soft

# [新增] 逆校准函数 (用于生成数据时还原)
def inverse_calibration(mag_clean, b_hard, W_soft):
    # m_raw = (W^-1 * m_clean) + b
    W_inv = np.linalg.inv(W_soft)
    return (mag_clean @ W_inv.T) + b_hard

def prepare_gan_data(data_path, output_path):
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    # Data is a dict of lists: 'X_list' -> list of arrays (Seq, Features)
    X_list = data['X_list']
    Y_quat_list = data['Y_quat_list']
    
    # Feature indices from prepare_data_ins.py/EulerDataset
    # Acc: 0:3, Gyro: 3:6, Mag: 6:9, ...
    
    all_noise = []
    all_cond = []
    
    # 1. Estimate Global Magnetic Field (B_world_mean)
    # [修改] 1. 计算全局统一的 Hard/Soft Iron 参数
    # 将所有序列的磁力计数据拼起来做一次全局校准
    all_mag_raw = []
    for x in X_list:
        all_mag_raw.append(x[:, 6:9])
    all_mag_raw = np.concatenate(all_mag_raw, axis=0)
    
    print("Performing Hard/Soft Iron Calibration on full dataset...")
    b_hard, W_soft = ellipsoidal_calibration(all_mag_raw)
    print(f"Hard Iron Bias: {b_hard}")
    print(f"Soft Iron Scale: {np.diag(W_soft)}")

    # [修改] 2. 计算全局 B_earth_local (校准后的世界系均值)
    avg_mag_world = np.zeros(3)
    total_samples = 0
    
    for i in range(len(X_list)):
        X_seq = X_list[i]
        quat_gt = Y_quat_list[i]
        mag_raw = X_seq[:, 6:9]
        
        # 截断长度对齐
        min_len = min(len(mag_raw), len(quat_gt))
        mag_raw = mag_raw[:min_len]
        quat_gt = quat_gt[:min_len]
        
        # A. 应用校准：得到"去除船体干扰"的磁场
        # m_cal = W * (m_raw - b)
        mag_cal = (mag_raw - b_hard) @ W_soft.T
        
        # B. 转到世界系
        r = R.from_quat(quat_gt)
        mag_world = r.apply(mag_cal)
        
        avg_mag_world += np.sum(mag_world, axis=0)
        total_samples += min_len
        
    B_ideal = avg_mag_world / total_samples
    print(f"Global Earth Field (Clean): {B_ideal}")
    
    # 3. Compute Residual Noise (train data for GAN)
    print("Computing pure environmental residuals...")
    for i in range(len(X_list)):
        X_seq = X_list[i]
        quat_gt = Y_quat_list[i]
        
        acc = X_seq[:, 0:3]
        gyro = X_seq[:, 3:6]
        mag_raw = X_seq[:, 6:9]
        
        if len(mag_raw) != len(quat_gt):
            min_len = min(len(mag_raw), len(quat_gt))
            acc = acc[:min_len]
            gyro = gyro[:min_len]
            mag_raw = mag_raw[:min_len]
            quat_gt = quat_gt[:min_len]
        
        # 校准
        mag_cal = (mag_raw - b_hard) @ W_soft.T
        
        # 计算纯净残差： n_real = m_cal - R.T * B_ideal
        r = R.from_quat(quat_gt)
        B_body_ideal = r.inv().apply(B_ideal)
        
        noise = mag_cal - B_body_ideal # 这里的noise现在是纯粹的环境动态噪声
        
        # Condition = [Acc, Gyro]
        cond = np.concatenate([acc, gyro], axis=1)
        
        all_noise.append(noise)
        all_cond.append(cond)
        
    # Concatenate all sequences
    X_noise = np.concatenate(all_noise, axis=0)
    X_cond = np.concatenate(all_cond, axis=0)
    
    # Save normalization stats
    noise_mean = np.mean(X_noise, axis=0)
    noise_std = np.std(X_noise, axis=0)
    cond_mean = np.mean(X_cond, axis=0)
    cond_std = np.std(X_cond, axis=0)
    
    print(f"Total samples: {len(X_noise)}")
    print(f"Saving to {output_path}...")
    
    gan_dataset = {
        'noise': X_noise,
        'condition': X_cond,
        'B_ideal': B_ideal,
        'calibration': {'b_hard': b_hard, 'W_soft': W_soft}, # [新增] 保存校准参数
        'stats': {
            'noise_mean': noise_mean, 'noise_std': noise_std,
            'cond_mean': cond_mean, 'cond_std': cond_std
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(gan_dataset, f)
    print("Done.")

if __name__ == "__main__":
    # Define paths for ALL 7 sequences
    TRAIN_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_train_full.pkl"
    TEST02_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_test_02.pkl"
    TEST07_PATH = "/home/dong/桌面/LNN-NAV/processed_data/lnn_ins_test_07.pkl"
    OUTPUT_PATH = "/home/dong/桌面/LNN-NAV/processed_data/gan_training_data.pkl"
    
    # 1. Load lists from all files
    X_total = []
    Y_quat_total = []
    Y_vel_body_total = []
    
    files = [TRAIN_PATH, TEST02_PATH, TEST07_PATH]
    for p in files:
        if not os.path.exists(p):
            print(f"Warning: {p} not found, skipping...")
            continue
            
        print(f"Loading {p}...")
        with open(p, 'rb') as f:
            data = pickle.load(f)
            
            # Case 1: Training Set (Lists of sequences)
            if 'X_list' in data:
                X_total.extend(data['X_list'])
                Y_quat_total.extend(data['Y_quat_list'])
                if 'Y_vel_body_list' in data:
                    Y_vel_body_total.extend(data['Y_vel_body_list'])
                else:
                    print(f"Error: Y_vel_body_list missing in {p}")
                    exit(1)
            
            # Case 2: Test Set (Single sequence)
            elif 'X' in data:
                X_total.append(data['X'])
                Y_quat_total.append(data['Y_quat'])
                if 'Y_velb' in data:
                    Y_vel_body_total.append(data['Y_velb'])
                elif 'Y_vel_body' in data: # Try alt name
                    Y_vel_body_total.append(data['Y_vel_body'])
                else:
                    print(f"Error: Y_velb/Y_vel_body missing in {p}")
                    # Note: processed_data/lnn_ins_test_02.pkl normally has 'Y_velb' based on prepare_data_ins.py
                    exit(1)
            else:
                print(f"Unknown format in {p}, skipping.")

    print(f"Total Sequences Loaded: {len(X_total)} (Should be 7)")

    all_noise = []
    all_cond = []
    
    # 2. Estimate Global Magnetic Field (B_ideal) using ALL data
    # [新增] 全局 Hard/Soft Iron 校准
    print("Performing Hard/Soft Iron Calibration on ALL sequences...")
    all_mag_raw = []
    for x in X_total:
        all_mag_raw.append(x[:, 6:9])
    all_mag_raw = np.concatenate(all_mag_raw, axis=0)
    
    b_hard, W_soft = ellipsoidal_calibration(all_mag_raw)
    print(f"Hard Iron Bias: {b_hard}")
    print(f"Soft Iron Scale: {np.diag(W_soft)}")
    
    avg_mag_world = np.zeros(3)
    total_samples_mag = 0
    
    print("Estimating local magnetic field (B_ideal) from all sequences...")
    for i in range(len(X_total)):
        mag_raw = X_total[i][:, 6:9]
        quat_gt = Y_quat_total[i]
        
        limit = min(len(mag_raw), len(quat_gt))
        mag_raw = mag_raw[:limit]
        quat_gt = quat_gt[:limit]
        
        # 校准
        mag_cal = (mag_raw - b_hard) @ W_soft.T
        
        r = R.from_quat(quat_gt)
        mag_world = r.apply(mag_cal)
        avg_mag_world += np.sum(mag_world, axis=0)
        total_samples_mag += limit
        
    B_ideal = avg_mag_world / total_samples_mag
    print(f"Estimated B_ideal (All 7 Seqs, Clean): {B_ideal}")

    # 3. Compute Residual & Condition
    print("Computing residual noise and constructing condition [Acc, Gyro, Vel]...")
    for i in range(len(X_total)):
        X_seq = X_total[i]
        quat_gt = Y_quat_total[i]
        vel_body = Y_vel_body_total[i]
        
        # Features [Acc(0:3), Gyro(3:6), Mag(6:9)]
        acc = X_seq[:, 0:3]
        gyro = X_seq[:, 3:6]
        mag = X_seq[:, 6:9]
        
        # Align lengths
        limit = min(len(mag), len(quat_gt), len(vel_body))
        acc = acc[:limit]
        gyro = gyro[:limit]
        mag = mag[:limit]
        vel_body = vel_body[:limit]
        quat_gt = quat_gt[:limit]
        
        # Residual Noise
        # 校准
        mag_cal = (mag - b_hard) @ W_soft.T
        
        r = R.from_quat(quat_gt)
        B_body_ideal = r.inv().apply(B_ideal)
        noise = mag_cal - B_body_ideal
        
        # Condition: Acc(3) + Gyro(3) + Vel(3) = 9 dim
        cond = np.concatenate([acc, gyro, vel_body], axis=1)
        
        all_noise.append(noise)
        all_cond.append(cond)

    X_noise = np.concatenate(all_noise, axis=0)
    X_cond = np.concatenate(all_cond, axis=0)
    
    noise_mean = np.mean(X_noise, axis=0)
    noise_std = np.std(X_noise, axis=0)
    cond_mean = np.mean(X_cond, axis=0)
    cond_std = np.std(X_cond, axis=0)
    
    print(f"Total samples: {len(X_noise)}")
    print(f"Condition Dimension: {X_cond.shape[1]} (Acc+Gyro+Vel)")
    print(f"Saving to {OUTPUT_PATH}...")
    
    gan_dataset = {
        'noise': X_noise,
        'condition': X_cond,
        'B_ideal': B_ideal,
        'calibration': {'b_hard': b_hard, 'W_soft': W_soft},
        'stats': {
            'noise_mean': noise_mean, 'noise_std': noise_std,
            'cond_mean': cond_mean, 'cond_std': cond_std
        }
    }
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(gan_dataset, f)
    print("Done.")
