# t5_utils.py
import numpy as np
import torch
class T5BatchProcessor:
    def __init__(self, mask_token_start_id, pad_token_id, eos_token_id, vocab_size):
        self.mask_token_start_id = mask_token_start_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        # avg_span_length and mask_prob are now passed to __call__
        print("T5BatchProcessor initialized.")

    def __call__(self, batch_x, avg_span_length: int, mask_prob: float):
        # traditionally in t5 paper, avg_span_length:3, mask_prob = 0.15
        # batch_x is a tensor of shape [B, T]
        B, T = batch_x.shape
        
        masked_inputs_list = []
        raw_targets_list = []

        for i in range(B):
            sequence = batch_x[i].tolist()
            input_seq, target_seq = self._create_masked_sequence(sequence, avg_span_length, mask_prob)
            masked_inputs_list.append(torch.tensor(input_seq, dtype=torch.long))
            raw_targets_list.append(torch.tensor(target_seq, dtype=torch.long))
        
        # 1. Pad the raw sequences.
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            masked_inputs_list, batch_first=True, padding_value=self.pad_token_id
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            raw_targets_list, batch_first=True, padding_value=self.pad_token_id
        )

        # 2. Create the decoder_inputs by shifting the clean padded_targets.
        decoder_inputs = torch.roll(padded_targets, shifts=1, dims=1)
        decoder_inputs[:, 0] = self.pad_token_id
        # 3. Create the final labels tensor for the loss function.
        #    Start with a copy and then replace all sentinel tokens with the pad_token_id.
        labels = padded_targets.clone()
        # A token is a sentinel if it's the EOS token OR if it's in the mask token range.
        # Assuming a maximum of 100 sentinel tokens for masking.
        is_mask_token = (labels >= self.mask_token_start_id) & (labels < self.mask_token_start_id + 100)
        is_eos_token = (labels == self.eos_token_id)
        # Create a combined mask for all tokens that should NOT contribute to the loss.
        sentinel_mask = is_mask_token | is_eos_token
        # Replace these tokens in the labels tensor with the pad_token_id.
        # in normie-llm training code, we can't pass a loss mask without in-channel 'ignore-these' special values... :(
        labels[sentinel_mask] = self.pad_token_id
        # 4. Create the attention masks based on the model inputs.
        encoder_padding_mask = (padded_inputs != self.pad_token_id)
        decoder_padding_mask = (decoder_inputs != self.pad_token_id)
        # 5. Return the final, cleaned labels.
        return padded_inputs, decoder_inputs, labels, encoder_padding_mask, decoder_padding_mask

        """# In your training loop:
        logits, _, _ = model(...)
        loss_fct = nn.CrossEntropyLoss(reduction='none') # IMPORTANT: get per-token loss
        loss = loss_fct(logits.view(-1, logits.size(-1)), padded_labels.view(-1))
        loss = loss.view_as(padded_labels) # Reshape back to [B, T]

        # Apply the mask and get the correct mean
        masked_loss = loss * loss_mask
        final_loss = masked_loss.sum() / loss_mask.sum() # Average by number of REAL tokens"""

    def create_curriculum_batch(self, batch_x: torch.Tensor, bucket_distribution: torch.Tensor, sampler_ref):
        """
        Constructs a heterogeneous batch according to the curriculum distribution.

        Args:
            batch_x: The raw input tensor of shape [B, T].
            bucket_distribution: The probability vector [num_buckets] from the sampler.
            sampler_ref: A reference to the curriculum sampler instance to get params from buckets.
        
        Returns:
            The usual batch tensors, PLUS a tensor of bucket indices for loss disaggregation.
        """
        B, T = batch_x.shape
        
        # 1. Assign each example in the batch to a bucket based on the distribution
        bucket_indices = torch.multinomial(bucket_distribution, num_samples=B, replacement=True)
        
        masked_inputs_list, raw_targets_list = [], []

        # 2. Process each example according to its assigned bucket
        for i in range(B):
            bucket_idx = bucket_indices[i].item()
            avg_span_length, mask_prob = sampler_ref.get_params_from_bucket(bucket_idx)
            
            sequence = batch_x[i].tolist()
            input_seq, target_seq = self._create_masked_sequence(sequence, avg_span_length, mask_prob)
            
            masked_inputs_list.append(torch.tensor(input_seq, dtype=torch.long))
            raw_targets_list.append(torch.tensor(target_seq, dtype=torch.long))

        # 3. Pad and prepare the final batch (logic is the same as before)
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            masked_inputs_list, batch_first=True, padding_value=self.pad_token_id
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            raw_targets_list, batch_first=True, padding_value=self.pad_token_id
        )
        # ... (the rest of the __call__ logic from the previous correct version) ...
        decoder_inputs = torch.roll(padded_targets, shifts=1, dims=1)
        decoder_inputs[:, 0] = self.pad_token_id
        labels = padded_targets.clone()
        is_mask_token = (labels >= self.mask_token_start_id) & (labels < self.mask_token_start_id + 100)
        is_eos_token = (labels == self.eos_token_id)
        sentinel_mask = is_mask_token | is_eos_token
        labels[sentinel_mask] = self.pad_token_id
        encoder_padding_mask = (padded_inputs != self.pad_token_id)
        decoder_padding_mask = (decoder_inputs != self.pad_token_id)

        return padded_inputs, decoder_inputs, labels, encoder_padding_mask, decoder_padding_mask, bucket_indices


    def _create_masked_sequence(self, tokens, avg_span_length, mask_prob):
        mask_indices = np.random.permutation(len(tokens))
        num_to_mask = int(len(tokens) * mask_prob)
        max_spans = (self.vocab_size - self.mask_token_start_id)
        
        # Use np.random.permutation for efficiency
        token_indices = np.random.permutation(len(tokens))
        
        masked_indices = set()
        i = 0
        while len(masked_indices) < num_to_mask and i < len(token_indices):
            idx = token_indices[i]
            if idx in masked_indices:
                i += 1
                continue
            
            # Sample a span length from a Poisson distribution
            span_len = max(1, np.random.poisson(avg_span_length))
            
            start_idx = idx
            end_idx = min(len(tokens), start_idx + span_len)
            
            for j in range(start_idx, end_idx):
                masked_indices.add(j)
            i += 1
        
        # Create the final input and target sequences
        input_tokens = []
        target_tokens = []
        mask_token_id_counter = 0
        
        i = 0
        while i < len(tokens):
            if i in masked_indices:
                # --- THIS IS THE CRITICAL GUARDRAIL ---
                if mask_token_id_counter >= max_spans:
                    # We've run out of sentinel tokens.
                    # Treat the rest of the tokens as unmasked.
                    input_tokens.append(tokens[i])
                    i += 1
                    continue
                # --- END OF GUARDRAIL ---
                start_of_span = i
                while i in masked_indices:
                    i += 1
                end_of_span = i
                
                mask_token = self.mask_token_start_id + mask_token_id_counter
                input_tokens.append(mask_token)
                
                target_tokens.append(mask_token)
                target_tokens.extend(tokens[start_of_span:end_of_span])
                
                mask_token_id_counter += 1
            else:
                input_tokens.append(tokens[i])
                i += 1

        target_tokens.append(self.eos_token_id) # The target sequence should end with EOS.
        
        return input_tokens, target_tokens

from scipy.optimize import root_scalar # uv pip install scipy

# ==============================================================================
# == NEW: Adaptive Curriculum Sampler - The culmination of our design.
# ==============================================================================
class AdaptiveCurriculumSampler:
    """
    Implements an online, adaptive curriculum for T5 denoising tasks.

    This sampler dynamically adjusts the distribution of task difficulties (span lengths)
    based on the model's real-time performance, following a set of principled constraints
    to ensure robust, generalized learning.
    """
    @torch.no_grad()
    def __init__(self,
                 num_buckets: int = 7,
                 alpha: float = 1.0,
                 epsilon: float = 0.1,
                 hardness_growth_rate: float = 0.01,
                 update_ratchet_every_n_steps: int = 300,
                 ema_beta: float = 0.99,
                 base_mask_prob: float = 0.15,
                 base_avg_span_len: int = 3,
                 target_prctile_hardness: float = 0.25):
        """
        Initializes the state of the curriculum sampler.

        Args:
            num_buckets: The number of task categories, bucket `i` covers spans of length `[2^i, 2^(i+1))`.
            alpha: Power-law exponent for the base sampler. `alpha=0` is uniform, `alpha=1` is linear preference for easy tasks.
            epsilon: The exploration factor for epsilon-greedy sampling to ensure all buckets are visited.
            hardness_growth_rate: How quickly the target difficulty increases as the model learns (e.g., 0.01 = 1% increase).
            update_ratchet_every_n_steps: How often to check if the model has improved and the difficulty ratchet should be increased.
            ema_beta: The beta for the Exponential Moving Average of bucket losses.
            base_mask_prob: The original mask probability for the base task.
            base_avg_span_len: The original average span length for the base task.
        """
        self.num_buckets = num_buckets
        self.alpha = alpha
        self.epsilon = epsilon
        self.hardness_growth_rate = hardness_growth_rate
        self.update_ratchet_every_n_steps = update_ratchet_every_n_steps
        self.ema_beta = ema_beta
        self.base_mask_prob = base_mask_prob
        self.base_avg_span_len = base_avg_span_len

        # Initialize state
        # Start with high loss to encourage initial exploration
        self.ema_losses = torch.full((self.num_buckets,), 5.0, dtype=torch.float32)
        self.ema_global_loss = torch.tensor(5.0, dtype=torch.float32)
        self.last_global_loss_for_ratchet = torch.tensor(5.0, dtype=torch.float32)
        
        # The NEW ratchet is a percentile. We start by targeting the 25th percentile easiest task.
        self.target_loss_percentile = target_prctile_hardness
        self.step_counter = 0
        print(f"AdaptiveCurriculumSampler initialized with {num_buckets} buckets.")

    def update(self, bucket_index: int, batch_loss: float):
        """Updates the sampler's internal state with the latest performance data."""
        # Update EMA for the specific bucket
        self.ema_losses[bucket_index] *= self.ema_beta
        self.ema_losses[bucket_index] += (1 - self.ema_beta) * batch_loss

        # Update global loss EMA
        self.ema_global_loss *= self.ema_beta
        self.ema_global_loss += (1 - self.ema_beta) * batch_loss
        
        self.step_counter += 1

        # Constraint 2: Update the difficulty ratchet periodically
        if self.step_counter % self.update_ratchet_every_n_steps == 0:
            if self.ema_global_loss < self.last_global_loss_for_ratchet:
                # Ratchet up the target percentile of difficulty
                self.target_loss_percentile += self.hardness_growth_rate
                # Clamp the percentile to a maximum (e.g., 90th percentile) to avoid instability
                self.target_loss_percentile = min(self.target_loss_percentile, 0.90)
            self.last_global_loss_for_ratchet = self.ema_global_loss.clone()
    
    @torch.no_grad()
    def _solve_for_lambda(self, p_base: torch.Tensor, target_loss: float) -> float:
        """Finds lambda to meet a target_loss constraint, using ema_losses as the hardness metric."""
        p_base_np = p_base.detach().cpu().numpy()
        # The empirical losses ARE the hardness values.
        hardness_values_np = self.ema_losses.detach().cpu().numpy()
        target_loss_np = target_loss

        def f(lam):
            # A positive lambda should penalize HIGH loss (hard) tasks, so we use +lambda
            exp_term = np.exp(lam * hardness_values_np)
            # We want to shift probability mass AWAY from P_base's low-loss preference
            # towards higher-loss tasks. A negative lambda will do this.
            # q_i = (1/Z) * p_i * exp(-lambda * H_i). If we want to upweight high H_i, lambda must be negative.
            exp_term = np.exp(-lam * hardness_values_np)
            z = np.sum(p_base_np * exp_term)
            if z == 0: return np.inf
            q = (1.0 / z) * p_base_np * exp_term
            expected_loss = np.sum(q * hardness_values_np)
            return expected_loss - target_loss_np

        current_expected_loss = np.sum(p_base_np * hardness_values_np)
        
        # If the greedy choice is already harder (higher loss) than the target, we don't need to push.
        if current_expected_loss >= target_loss_np:
             return 0.0

        try:
            # We are looking for a negative lambda to increase the expected loss.
            sol = root_scalar(f, bracket=[-10, 0], method='brentq')
            return sol.root
        except ValueError:
            return 0.0

    @torch.no_grad()
    def get_distribution(self, device='cpu') -> dict:
        """Calculates the final task distribution based on the full constraint system."""
        # Step 1: Greedy choice based on performance (same as before)
        learnability_scores = 1.0 / self.ema_losses
        weights = torch.pow(learnability_scores, self.alpha)
        p_base = weights / torch.sum(weights)

        # Step 2: Determine the target loss from the current difficulty percentile
        # Ensure losses are sorted for quantile calculation
        sorted_losses, _ = torch.sort(self.ema_losses)
        target_loss = torch.quantile(self.ema_losses, self.target_loss_percentile)
        
        # Step 3: Solve for the lambda that gets us from P_base's expected loss to the target_loss
        lambda_val = self._solve_for_lambda(p_base, target_loss.item())
        
        # Step 4: Calculate the final distribution
        # The hardness vector IS the loss vector
        hardness_vector = self.ema_losses
        exp_term = torch.exp(-lambda_val * hardness_vector.to(p_base.device))
        z = torch.sum(p_base * exp_term)
        p_compromise = (1.0 / z) * p_base * exp_term if z > 0 else p_base

        p_final = (1.0 - self.epsilon) * p_compromise
        p_final += self.epsilon * (1.0 / self.num_buckets)
        p_final /= p_final.sum()

        expected_loss = torch.sum(p_final.to(device) * hardness_vector.to(device))
        
        return {
            "p_final": p_final.to(device),
            "ema_losses": self.ema_losses.clone(),
            "target_loss": target_loss.clone(), # New name
            "expected_loss": expected_loss.to(device), # New name
            "lambda": lambda_val,
            "p_base": p_base.to(device)
        }

    # You can add a separate `sample` method if you ever need a single draw
    def sample(self) -> int:
        """Samples a single task bucket index from the current distribution."""
        p_final = self.get_distribution()
        return torch.multinomial(p_final, 1).item()
        
    def get_params_from_bucket(self, bucket_index: int) -> (int, float):
        """Converts a bucket index into a concrete span length and mask probability."""
        #base from span:2
        min_len = int(2**(bucket_index + 1))
        max_len = int(2**(bucket_index + 2))
        
        # Uniformly sample a span length from within the bucket's range
        avg_span_length = np.random.randint(min_len, max_len)
        # Adjust mask_prob to keep the total number of masked tokens roughly constant
        mask_prob = self.base_mask_prob

        return avg_span_length, mask_prob