#redis_utils.py
import redis
import torch
import numpy as np
import json

def serialize_tensor_to_redis(r: redis.Redis, key: str, tensor: torch.Tensor, metadata: dict):
    """
    Serializes a PyTorch tensor and its metadata to a Redis Hash using NumPy.

    Args:
        r: The Redis connection object.
        key: The Redis key where the hash will be stored.
        tensor: The PyTorch tensor to serialize.
        metadata: A dictionary of supplementary data (e.g., model_config, job_params).
    """
    # Ensure tensor is on CPU before converting to NumPy
    numpy_array = tensor.cpu().numpy()

    # Use a pipeline for atomic and efficient execution
    pipe = r.pipeline()
    
    # Store the core tensor data
    pipe.hset(key, 'tensor_data', numpy_array.tobytes())
    pipe.hset(key, 'tensor_shape', json.dumps(numpy_array.shape))
    pipe.hset(key, 'tensor_dtype', str(numpy_array.dtype))
    
    # Store the supplementary metadata as a JSON string
    pipe.hset(key, 'metadata', json.dumps(metadata))
    
    # Set an expiration to auto-clean old tensors
    pipe.expire(key, 3600) # 1 hour
    
    pipe.execute()


def deserialize_tensor_from_redis(r: redis.Redis, key: str) -> (torch.Tensor, dict):
    """
    Deserializes a PyTorch tensor and its metadata from a Redis Hash.

    Args:
        r: The Redis connection object.
        key: The Redis key of the hash.

    Returns:
        A tuple containing the deserialized (tensor, metadata_dict), or (None, None) if not found.
    """
    # Use a pipeline to fetch all data in one round trip
    pipe = r.pipeline()
    pipe.hget(key, 'tensor_data')
    pipe.hget(key, 'tensor_shape')
    pipe.hget(key, 'tensor_dtype')
    pipe.hget(key, 'metadata')
    results = pipe.execute()

    # Unpack results
    tensor_bytes, shape_json, dtype_str, metadata_json = results

    if not tensor_bytes:
        return None, None # Key not found

    # Decode and reconstruct the tensor
    shape = json.loads(shape_json)
    dtype = np.dtype(dtype_str.decode())
    numpy_array = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
    tensor = torch.from_numpy(numpy_array)

    # Decode the metadata
    metadata = json.loads(metadata_json)

    return tensor, metadata