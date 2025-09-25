import os
import torch
import torch.nn as nn
import pgptlformer
import argparse
from tqdm import tqdm
from prompt_utils import AudioPromptGenerator
from datetime import datetime
import random
import redis
import json
import traceback
import yaml

from redis_utils import serialize_tensor_to_redis

# --- All Model Inference and Generation Logic ---

def _t5_decoder_engine(model, encoder_input_ids, max_new, temp, top_k, model_config):
    # (This function is identical to the one you provided and is correct)
    pad_id = model_config['pad_token_id']
    eos_id = model_config['eos_token_id']
    mask_id = model_config['mask_token_start_id']
    batch_size = encoder_input_ids.shape[0]
    device = encoder_input_ids.device
    
    encoder_padding_mask = (encoder_input_ids != pad_id)
    with torch.no_grad():
        encoder_hidden_states = model.encode(encoder_input_ids, encoder_padding_mask)
    
    decoder_input_ids = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    has_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new):
        with torch.no_grad():
            logits = model.decode_step(decoder_input_ids, encoder_hidden_states, encoder_padding_mask)
        logits = logits[:, -1, :] / temp
        
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx_next[has_finished] = pad_id
        has_finished |= (idx_next.squeeze() == eos_id)
        decoder_input_ids = torch.cat((decoder_input_ids, idx_next), dim=1)
        if has_finished.all():
            break
            
    output_sequences = []
    for i in range(batch_size):
        seq = decoder_input_ids[i, 1:]
        eos_idx = (seq == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_idx) > 0:
            seq = seq[:eos_idx[0]]
        output_sequences.append(seq)
        
    return output_sequences

def t5_continue_step(model, context, max_new, temp, top_k, model_config):
    # (This function is identical to the one you provided and is correct)
    batch_size = context.shape[0]
    device = context.device
    mask_id = model_config['mask_token_start_id']
    mask_tokens = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    encoder_input_ids = torch.cat([context, mask_tokens], dim=1)
    return _t5_decoder_engine(model, encoder_input_ids, max_new, temp, top_k, model_config)

def t5_infill_step(model, context, max_new, temp, top_k, model_config):
    """ High-Level In-filling Wrapper """
    prefix_len = context.shape[1] // 2
    prefix_batch = context[:, :prefix_len]
    postfix_batch = context[:, -prefix_len:]
    
    batch_size = context.shape[0]
    device = context.device
    mask_id = model_config['mask_token_start_id']
    
    mask_tokens = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=device)
    encoder_input_ids = torch.cat([prefix_batch, mask_tokens, postfix_batch], dim=1)
    return _t5_decoder_engine(model, encoder_input_ids, max_new, temp, top_k, model_config)

def generate_long_form_audio(model, model_config, prompt_generator, job_params, device):
    """
    This function contains the core generation loop.
    It's called by the service for each new job.
    """
    # Set seed for this specific job
    torch.manual_seed(job_params['seed'])
    torch.cuda.manual_seed(job_params['seed'])

    # Get initial prompts
    initial_sequences = prompt_generator.get_prompts(
        num_prompts=job_params['num_samples'],
        prompt_length=job_params['prompt_length']
    ).to(device)

    current_sequences = initial_sequences
    pad_id = model_config['pad_token_id']
    batch_size = job_params['num_samples']

    # --- The Main Generation Loop ---
    with torch.no_grad():
        for i in tqdm(range(job_params['num_iterations']), desc="Rollout Progress"):
            context = current_sequences[:, -job_params['prompt_length']:]
            
            # Decide whether to infill or continue based on probability
            if random.random() < job_params['infill_prob']:
                print(f"\nStep {i+1}: Performing ADDITIVE in-fill...")
                
                infilled_tokens_list = t5_infill_step(
                    model, context, job_params['infill_length'], 
                    job_params['temperature'], job_params['top_k'], model_config
                )

                prefix_len = context.shape[1] // 2
                postfix_motif = context[:, -prefix_len:]

                next_sequences_list = []
                for j in range(batch_size):
                    part_to_append = torch.cat([infilled_tokens_list[j], postfix_motif[j]])
                    full_extended_sequence = torch.cat([current_sequences[j], part_to_append])
                    next_sequences_list.append(full_extended_sequence)
                
                current_sequences = torch.nn.utils.rnn.pad_sequence(
                    next_sequences_list, batch_first=True, padding_value=pad_id
                )

            else: # Continuation logic
                print(f"\nStep {i+1}: Performing continuation...")
                new_tokens_list = t5_continue_step(
                    model, context, job_params['tokens_per_step'], 
                    job_params['temperature'], job_params['top_k'], model_config
                )

                padded_new_tokens = torch.nn.utils.rnn.pad_sequence(
                    new_tokens_list, batch_first=True, padding_value=pad_id
                )
                current_sequences = torch.cat([current_sequences, padded_new_tokens], dim=1)

    return current_sequences


# --- Service Infrastructure ---

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_config_dict, device):
    out_dir = model_config_dict['out_dir']
    checkpoint_name = model_config_dict['checkpoint_name']
    
    print(f"Loading model from: {os.path.join(out_dir, checkpoint_name)}")
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_config = checkpoint['model_args']
    
    model = pgptlformer.PGPT_Lformer(model_config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
        
    print("--- Model loaded successfully ---")
    return model, model_config

def main_service_loop(config):
    """The main service loop, now fully configured by the YAML file."""
    
    common_cfg = config['common']
    t5_cfg = config['t5_service']
    device = t5_cfg['device']

    r = redis.Redis(
        host=common_cfg['redis_host'], port=common_cfg['redis_port'], decode_responses=True
    )
    INPUT_QUEUE = t5_cfg['input_queue']
    OUTPUT_QUEUE = common_cfg['tensor_job_queue']
    print(f"Service configured. Listening for jobs on '{INPUT_QUEUE}'...")

    model, model_config = load_model(t5_cfg['model'], device)
    
    prompt_generator = AudioPromptGenerator(
        npz_path=t5_cfg['data']['prompt_npz_path'], 
        parquet_path=t5_cfg['data']['prompt_parquet_path']
    )

    while True:
        try:
            _, job_data_str = r.blpop(INPUT_QUEUE)
            client_job = json.loads(job_data_str)
            
            run_id = datetime.now().strftime('%y%m%d%H%M%S')
            print(f"\n[{run_id}] Received job: {client_job}")

            # --- Merge client job with config defaults ---
            # Start with defaults, then update with client-provided values
            job_params = t5_cfg['generation_defaults'].copy()
            job_params.update(client_job)
            print(f"Running with combined parameters: {job_params}")

            # --- Call the dedicated generation function ---
            final_sequences = generate_long_form_audio(
                model, model_config, prompt_generator, job_params, device
            )

            # --- NEW: Serialize using the NumPy utility ---
            redis_key = f"{config['common']['tensor_key_prefix']}:{run_id}"
            metadata_to_save = {
                'model_config': model_config,
                'run_id': run_id,
                'job_params': job_params
            }

            serialize_tensor_to_redis(r, redis_key, final_sequences, metadata_to_save)
            print(f"Serialized tensor for run {run_id} to Redis key '{redis_key}'")
            
            decode_job = {"tensor_redis_key": redis_key, "run_id": run_id}
            r.rpush(config['common']['tensor_job_queue'], json.dumps(decode_job))
            print(f"Pushed decode job with Redis key to '{config['common']['tensor_job_queue']}'")

            r.publish(config['common']['job_completion_channel'], run_id)
            print(f"Published completion notice for run {run_id} to channel '{config['common']['job_completion_channel']}'")
        except Exception as e:
            print(f"[ERROR] An error occurred in the service loop: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T5 Audio Generation Service")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    config = load_config(args.config)
    main_service_loop(config)