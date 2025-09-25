# encodec_service.py (Corrected)
import os
import torch
import torchaudio
from encodec import EncodecModel
import redis
import json
import traceback
import argparse  # <-- ADDED
import yaml      # <-- ADDED

from redis_utils import deserialize_tensor_from_redis

def load_config(path):
    """Loads the YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Encodec Audio Decoding Service")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    # --- Unpack config ---
    common_cfg = config['common']
    encodec_cfg = config['encodec_service']
    device = encodec_cfg['device']
    
    # --- Redis Connection (CRITICAL FIX) ---
    # We must NOT use decode_responses=True, as it will corrupt the raw tensor bytes.
    # redis_utils handles decoding the string fields manually.
    r = redis.Redis(host=common_cfg['redis_host'], port=common_cfg['redis_port'])
    
    INPUT_QUEUE = common_cfg['tensor_job_queue']
    OUTPUT_DIR = encodec_cfg['output']['audio_dir']
    
    print(f"Service configured. Listening for jobs on '{INPUT_QUEUE}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- MODEL LOADING (ONCE) ---
    print("Loading Encodec 24kHz model...")
    encodec_model = EncodecModel.encodec_model_24khz().to(device)
    print("--- Encodec model loaded successfully ---")

    # --- MAIN SERVICE LOOP ---
    while True:
        try:
            # blpop returns bytes, so we decode the job string manually
            _, job_data_str = r.blpop(INPUT_QUEUE)
            job_data = json.loads(job_data_str.decode('utf-8'))
            
            run_id = job_data["run_id"]
            redis_key = job_data["tensor_redis_key"]
            
            print(f"\n[{run_id}] Received decoding job for Redis key: {redis_key}")
            
            final_rollouts, metadata = deserialize_tensor_from_redis(r, redis_key)
            
            if final_rollouts is None:
                print(f"[ERROR] Tensor key '{redis_key}' not found or invalid. Skipping.")
                continue

            final_rollouts = final_rollouts.to(device)
            model_config = metadata['model_config']
            pad_id = model_config.get('pad_token_id', -1)
            num_samples = final_rollouts.shape[0]

            for i in range(num_samples):
                print(f"  - Decoding sample {i+1}/{num_samples}")
                token_tensor = final_rollouts[i]
                
                if pad_id != -1:
                    token_tensor = token_tensor[token_tensor != pad_id]

                codes = token_tensor.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    waveforms = encodec_model.decode([(codes, None)])
                
                output_filename = os.path.join(OUTPUT_DIR, f"{run_id}_sample_{i}.wav")
                torchaudio.save(output_filename, waveforms.squeeze(0).cpu(), sample_rate=encodec_model.sample_rate)
                print(f"  - Saved {output_filename}")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    main()