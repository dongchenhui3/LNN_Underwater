import pickle
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

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
    avg_mag_world = np.zeros(3)
    total_samples = 0
    
    print("Estimating local magnetic field (B_ideal)...")
    for i in range(len(X_list)):
        X_seq = X_list[i]
        quat_gt = Y_quat_list[i]
        
        mag = X_seq[:, 6:9]
        
        # Ensure quat matches length (it should)
        if len(mag) != len(quat_gt):
            min_len = min(len(mag), len(quat_gt))
            mag = mag[:min_len]
            quat_gt = quat_gt[:min_len]
            
        # Create Rotation object
        r = R.from_quat(quat_gt)
        
        # Rotate Mag to World Frame: B_world = R_WB * B_body
        mag_world = r.apply(mag)
        
        avg_mag_world += np.sum(mag_world, axis=0)
        total_samples += len(mag)
        
    B_ideal = avg_mag_world / total_samples
    print(f"Estimated B_ideal (World Frame): {B_ideal}")
    print(f"Magnitude: {np.linalg.norm(B_ideal)}")
    
    # 2. Compute Residual Noise (train data for GAN)
    print("Computing residual noise...")
    for i in range(len(X_list)):
        X_seq = X_list[i]
        quat_gt = Y_quat_list[i]
        
        acc = X_seq[:, 0:3]
        gyro = X_seq[:, 3:6]
        mag = X_seq[:, 6:9]
        
        if len(mag) != len(quat_gt):
            min_len = min(len(mag), len(quat_gt))
            acc = acc[:min_len]
            gyro = gyro[:min_len]
            mag = mag[:min_len]
            quat_gt = quat_gt[:min_len]
        
        r = R.from_quat(quat_gt)
        
        # Project Ideal World Field back to Body Frame
        # B_body_ideal = R_WB^T * B_ideal
        B_body_ideal = r.inv().apply(B_ideal)
        
        # Noise = Raw - Ideal
        noise = mag - B_body_ideal
        
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
    avg_mag_world = np.zeros(3)
    total_samples_mag = 0
    
    print("Estimating local magnetic field (B_ideal) from all sequences...")
    for i in range(len(X_total)):
        mag = X_total[i][:, 6:9]
        quat_gt = Y_quat_total[i]
        
        limit = min(len(mag), len(quat_gt))
        mag = mag[:limit]
        quat_gt = quat_gt[:limit]
        
        r = R.from_quat(quat_gt)
        mag_world = r.apply(mag)
        avg_mag_world += np.sum(mag_world, axis=0)
        total_samples_mag += limit
        
    B_ideal = avg_mag_world / total_samples_mag
    print(f"Estimated B_ideal (All 7 Seqs): {B_ideal}")

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
        r = R.from_quat(quat_gt)
        B_body_ideal = r.inv().apply(B_ideal)
        noise = mag - B_body_ideal
        
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
        'stats': {
            'noise_mean': noise_mean, 'noise_std': noise_std,
            'cond_mean': cond_mean, 'cond_std': cond_std
        }
    }
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(gan_dataset, f)
    print("Done.")
