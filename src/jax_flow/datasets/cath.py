import os
import requests
import json
import numpy as np
import gzip

# 1. Download Function
def download_cath_dataset(save_path="./data/chain_set.jsonl"):
    url = "http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl"
    
    if os.path.exists(save_path):
        print(f"Dataset already exists at {save_path}")
        return

    print(f"Downloading CATH dataset to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Stream download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def process_chain(entry, img_size=128):
    try:
        # 1. Safe Load
        coords_raw = entry['coords']['CA']
        if not coords_raw or len(coords_raw) < 10:
            return None
            
        coords = np.array(coords_raw, dtype=np.float32)
        
        # 2. Check for NaNs in raw coordinates
        if np.isnan(coords).any():
            return None
            
        L = coords.shape[0]
        
        # 3. Crop or Pad
        if L > img_size:
            start = np.random.randint(0, L - img_size)
            coords = coords[start : start+img_size]
        else:
            # Pad with "Far" value (100.0)
            # Do NOT use 0.0 (implies collision)
            # Do NOT use NaN
            padding = np.ones((img_size - L, 3), dtype=np.float32) * 100.0
            coords = np.concatenate([coords, padding], axis=0)
            
        # 4. Compute Distance Map
        # shape: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        diff = coords[:, None, :] - coords[None, :, :]
        # Square dist
        d2 = np.sum(diff**2, axis=-1)
        # Sqrt (Safe: clip min to 0 to avoid sqrt(-1e-8) numerical errors)
        d2 = np.maximum(d2, 0.0)
        dist_map = np.sqrt(d2)
        
        # 5. Normalize
        # Clip real distances to [0, 30] Angstroms
        dist_map = np.clip(dist_map, 0.0, 30.0)
        
        # Normalize to [-1, 1]
        # (0..30) -> (0..1) -> (-1..1)
        dist_map = (dist_map / 30.0) * 2.0 - 1.0
        
        # 6. FINAL PARANOID CHECK
        if np.isnan(dist_map).any() or np.isinf(dist_map).any():
            return None
            
        return dist_map
        
    except Exception as e:
        return None

def cath_generator(jsonl_path, batch_size=16, img_size=128):
    data = []
    
    while True:
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    dist_map = process_chain(entry, img_size)
                    
                    if dist_map is None:
                        continue
                        
                    data.append(dist_map[:, :, None]) # Add channel
                    
                    if len(data) == batch_size:
                        batch = np.array(data, dtype=np.float32)
                        
                        # Double Check Batch
                        if np.isnan(batch).any():
                            print("Warning: Skipped a NaN batch!")
                            data = []
                            continue
                            
                        yield batch
                        data = []
                        
                except ValueError:
                    continue