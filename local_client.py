# local_client.py (Corrected)
import redis
import json
import argparse
import yaml
import os
import torch
import requests
import time
from tqdm import tqdm

from redis_utils import deserialize_tensor_from_redis

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def connect_redis(config):
    common_cfg = config['common']
    try:
        # CRITICAL FIX: Do not use decode_responses=True.
        r = redis.Redis(host=common_cfg['redis_host'], port=common_cfg['redis_port'])
        r.ping()
        print(f"Successfully connected to Redis at {common_cfg['redis_host']}:{common_cfg['redis_port']}.")
        return r
    except redis.exceptions.ConnectionError:
        print("Could not connect to Redis.")
        print("Please ensure you have an active SSH tunnel: ssh -L 6379:localhost:6379 your_user@your_pi_ip")
        exit()

def submit_job(r, config, args):
    job = {
        "num_samples": args.samples,
        "num_iterations": args.iters,
        "seed": args.seed,
        "infill_prob": args.infill_prob
    }
    # We must encode the job JSON to bytes to send to Redis
    r.rpush(config['t5_service']['input_queue'], json.dumps(job).encode('utf-8'))
    print("Successfully submitted job:")
    print(json.dumps(job, indent=2))

def monitor_queues(r, config):
    t5_queue = config['t5_service']['input_queue']
    encodec_queue = config['common']['tensor_job_queue']
    
    t5_len = r.llen(t5_queue)
    encodec_len = r.llen(encodec_queue)
    # .keys() returns bytes, so we need to decode them
    tensors_stored_keys = r.keys(f"{config['common']['tensor_key_prefix']}:*")
    tensors_stored = len([key.decode('utf-8') for key in tensors_stored_keys])


    print("\n--- Service Mesh Status ---")
    print(f"Jobs waiting for T5 generation: {t5_len}")
    print(f"Tensors waiting for Encodec decoding: {encodec_len}")
    print(f"Tensors currently stored in Redis: {tensors_stored}")
    print("---------------------------\n")

def fetch_and_save(r, config):
    common_cfg = config['common']
    client_cfg = config['local_client']
    encodec_cfg = config['encodec_service']
    
    os.makedirs(client_cfg['output_dir'], exist_ok=True)
    
    p = r.pubsub(ignore_subscribe_messages=True)
    p.subscribe(common_cfg['job_completion_channel'])
    print(f"Listening on channel '{common_cfg['job_completion_channel']}' for completed jobs...")

    for message in p.listen():
        # Pub/Sub messages are bytes, decode them
        run_id = message['data'].decode('utf-8')
        print(f"\n--- Detected completed job: {run_id} ---")
        
        redis_key = f"{common_cfg['tensor_key_prefix']}:{run_id}"
        tensor, metadata = deserialize_tensor_from_redis(r, redis_key)

        if tensor is None:
            print(f"[ERROR] Could not find or deserialize data for key {redis_key}.")
            continue
        
        local_data_path = os.path.join(client_cfg['output_dir'], f"{run_id}_data.pt")
        torch.save({'final_rollout': tensor, 'metadata': metadata}, local_data_path)
        print(f"  -> Saved tensor and metadata to: {local_data_path}")
        
        # BUG FIX: Use 'metadata' variable, not 'data'
        num_samples = metadata['job_params']['num_samples']
        for i in range(num_samples):
            audio_filename = f"{run_id}_sample_{i}.wav"
            url = f"http://{client_cfg['remote_server_ip']}:{encodec_cfg['audio_server_port']}/{audio_filename}"
            local_audio_path = os.path.join(client_cfg['output_dir'], audio_filename)
            
            print(f"  -> Attempting to download audio from {url}...")
            
            for attempt in range(10):
                try:
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(local_audio_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"     Success! Saved to {local_audio_path}")
                        break
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(1)
            else:
                print(f"     [FAILED] Could not download audio after 10 attempts.")

def main():
    parser = argparse.ArgumentParser(description="Client for T5 Audio Generation Service Mesh.")
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_submit = subparsers.add_parser('submit', help='Submit a new generation job.')
    p_submit.add_argument('--seed', type=int, default=420)
    p_submit.add_argument('--samples', type=int, default=4)
    p_submit.add_argument('--iters', type=int, default=12)
    # BUG FIX: 'add_argument', not 'add_nargument'
    p_submit.add_argument('--infill_prob', type=float, default=0.0)

    subparsers.add_parser('monitor', help='Check the status of the job queues.')
    subparsers.add_parser('fetch', help='Listen for completed jobs and download results.')

    args = parser.parse_args()
    config = load_config()
    r = connect_redis(config)

    if args.command == 'submit':
        submit_job(r, config, args)
    elif args.command == 'monitor':
        monitor_queues(r, config)
    elif args.command == 'fetch':
        fetch_and_save(r, config)

if __name__ == '__main__':
    main()