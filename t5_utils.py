# t5_utils.py
import numpy as np
import torch


class T5BatchProcessor:
    """turns a continuous token stream into a span-denoising (input, target) pair.

    --- the EOS bug, and what replaced it --------------------------------------------------

    the "fix special token masking" commit built the loss labels like this:

        is_mask_token = (labels >= mask_token_start_id) & (labels < mask_token_start_id + 100)
        is_eos_token  = (labels == eos_token_id)
        labels[is_mask_token | is_eos_token] = pad_token_id     # -> skipped by ignore_index

    so EOS was deleted from every target the model was ever trained on. the model was
    therefore never once taught to emit EOS -- while sample_t5.py, sample_audio_t5.py,
    sample_audio_t5_skipdecode.py and t5_service.py ALL terminate generation on EOS. every
    one of them has only ever stopped by running out of max_new. a year of rollouts.

    EOS is now in the loss. it is the single token the samplers depend on.

    sentinels stay OUT of the loss by default, and that part was a deliberate and defensible
    choice, not an accident: the k-th sentinel in a target stream is always
    mask_token_start_id + k, a deterministic counter the model can produce with zero
    information about the audio or the text. training on them buys nothing and it deflates
    the reported loss by mixing in free tokens. set train_on_sentinels=True for the
    T5-canonical treatment (original T5 trains on the full target including sentinels).

    the range was also wrong. it hardcoded `mask_token_start_id + 100` -- "Assuming a maximum
    of 100 sentinel tokens for masking" -- while _create_masked_sequence's own guardrail
    allows `vocab_size - mask_token_start_id` of them, which is 1022 for the audio configs.
    sentinels 100..1021 fell through the exclusion and WERE trained on, so the actual
    behaviour was "sentinels are excluded, except the ones that aren't". the bound is now
    derived from the same expression the masker uses -- see self.max_spans -- so the two
    cannot drift apart again.
    """

    def __init__(self, mask_token_start_id, pad_token_id, eos_token_id, vocab_size,
                 train_on_sentinels=False):
        self.mask_token_start_id = mask_token_start_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.train_on_sentinels = train_on_sentinels
        # avg_span_length and mask_prob are now passed to __call__
        print("T5BatchProcessor initialized.")

    @property
    def max_spans(self):
        """how many distinct sentinel ids exist. ONE definition, used by both the masker
        (as its guardrail) and the loss-exclusion range."""
        return self.vocab_size - self.mask_token_start_id

    def build_labels(self, padded_targets):
        """decoder targets -> loss labels, with excluded positions set to pad_token_id.

        EOS survives. sentinels do not (unless train_on_sentinels). padding was already
        pad_token_id and stays that way.
        """
        labels = padded_targets.clone()
        if not self.train_on_sentinels:
            is_sentinel = ((labels >= self.mask_token_start_id)
                           & (labels < self.mask_token_start_id + self.max_spans))
            labels[is_sentinel] = self.pad_token_id
        return labels

    def build_decoder_padding_mask(self, decoder_inputs):
        """decoder inputs -> [B, T] visibility mask.

        column 0 of decoder_inputs is the start-of-sequence slot, and this codebase spells
        SOS as pad_token_id (there is no separate bos id). a bare
        `(decoder_inputs != pad_token_id)` therefore marked position 0 INVISIBLE: not just
        to itself -- with a causal mask, query row 0 could attend to nothing at all -- but
        to every later query as well, since the padding mask gates KEY visibility for the
        whole column. no decoder position could ever see the start of the sequence.

        measured, not assumed: this does NOT produce NaN on torch 2.5.1. sdpa's safe-softmax
        returns an all-zero row rather than 0/0, so the symptom was silent -- position 0's
        self-attention output was exactly zero and BOS carried no signal anywhere. that is
        the failure mode this repo specializes in: correct-looking numbers from a mechanism
        that is not running.

        column 0 is a real token. say so.
        """
        mask = (decoder_inputs != self.pad_token_id)
        mask[:, 0] = True
        return mask

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
        # in normie-llm training code, we can't pass a loss mask without in-channel
        # 'ignore-these' special values... :(
        labels = self.build_labels(padded_targets)
        # 4. Create the attention masks based on the model inputs.
        encoder_padding_mask = (padded_inputs != self.pad_token_id)
        decoder_padding_mask = self.build_decoder_padding_mask(decoder_inputs)
        # 5. Return the final, cleaned labels.
        return padded_inputs, decoder_inputs, labels, encoder_padding_mask, decoder_padding_mask

    # (an unreachable string literal after that return used to hold a "# In your training
    #  loop:" snippet. it was worse than dead -- it TAUGHT `logits, _, _ = model(...)`, the
    #  3-unpack that killed all three samplers, and it referenced a `loss_mask` this API has
    #  never returned. deleted. the live version is loader.main(), which unpacks by index.)

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

        # --- device-sync accounting ---
        # this function is host-side by nature: numpy span sampling, python list building.
        # it used to reach across the bus once PER SEQUENCE, twice over:
        #     bucket_idx = bucket_indices[i].item()     # B transfers
        #     sequence   = batch_x[i].tolist()          # B more, each of length T
        # with device_batch_size=32 that is 64 pipeline drains per micro-step just to
        # assemble a batch. one transfer each, up front, same values, same order.
        bucket_index_list = bucket_indices.tolist()
        batch_rows = batch_x.tolist()

        masked_inputs_list, raw_targets_list = [], []

        # 2. Process each example according to its assigned bucket
        for i in range(B):
            avg_span_length, mask_prob = sampler_ref.get_params_from_bucket(bucket_index_list[i])

            input_seq, target_seq = self._create_masked_sequence(batch_rows[i], avg_span_length, mask_prob)

            masked_inputs_list.append(torch.tensor(input_seq, dtype=torch.long))
            raw_targets_list.append(torch.tensor(target_seq, dtype=torch.long))

        # 3. Pad and prepare the final batch (logic is the same as before)
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            masked_inputs_list, batch_first=True, padding_value=self.pad_token_id
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            raw_targets_list, batch_first=True, padding_value=self.pad_token_id
        )
        # ... (the rest of the __call__ logic, and now it is literally the same code) ...
        decoder_inputs = torch.roll(padded_targets, shifts=1, dims=1)
        decoder_inputs[:, 0] = self.pad_token_id
        labels = self.build_labels(padded_targets)
        encoder_padding_mask = (padded_inputs != self.pad_token_id)
        decoder_padding_mask = self.build_decoder_padding_mask(decoder_inputs)

        # bucket_indices comes back as a HOST tensor built from the list we already
        # transferred, so the trainer's curriculum bookkeeping needs no further sync.
        return (padded_inputs, decoder_inputs, labels, encoder_padding_mask, decoder_padding_mask,
                torch.tensor(bucket_index_list, dtype=torch.long))


    def _create_masked_sequence(self, tokens, avg_span_length, mask_prob):
        num_to_mask = int(len(tokens) * mask_prob)
        max_spans = self.max_spans

        # Use np.random.permutation for efficiency
        # (there used to be a second, identical permutation on the line above num_to_mask,
        #  bound to `mask_indices` and immediately shadowed by this one. it consumed a draw
        #  from the global numpy rng and was never read.)
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

# scipy is imported inside _solve_for_lambda, not here. it is the single heaviest dependency
# in this file and it is only needed by one method of one class, so a module-scope import
# made T5BatchProcessor -- which needs nothing but numpy and torch -- untestable anywhere
# scipy is missing. (uv pip install scipy, if you want the curriculum.)

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
                 num_buckets: int = 8,
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
        """Finds lambda to meet a target_loss constraint, using ema_losses as the hardness metric.

        this is a HOST-SIDE CONTROL LOOP. brentq is a scalar root find with data-dependent
        iteration count; it cannot be made device-resident and there is no point pretending
        otherwise. everything it touches (ema_losses, p_base) is deliberately kept on the cpu
        for exactly this reason, so calling it costs no transfer at all. see loader.main().
        """
        from scipy.optimize import root_scalar # uv pip install scipy
        p_base_np = p_base.detach().cpu().numpy()
        # The empirical losses ARE the hardness values.
        hardness_values_np = self.ema_losses.detach().cpu().numpy()
        target_loss_np = target_loss

        def f(lam):
            # We want to shift probability mass AWAY from P_base's low-loss preference
            # towards higher-loss tasks. A negative lambda will do this.
            # q_i = (1/Z) * p_i * exp(-lambda * H_i). If we want to upweight high H_i, lambda must be negative.
            # (there was a first `exp_term = np.exp(+lam * H)` here, immediately overwritten
            #  by the line below. leftover from working out the sign. deleted.)
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
        # (a `sorted_losses, _ = torch.sort(...)` lived here with the comment "Ensure losses
        #  are sorted for quantile calculation". torch.quantile sorts internally; the result
        #  was never read.)
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
        # this used to be `p_final = self.get_distribution()` -- get_distribution returns a
        # DICT -- and then handed that dict straight to torch.multinomial. it has never been
        # called, which is the only reason it has never raised. now it works.
        return int(torch.multinomial(self.get_distribution()["p_final"], 1).item())

    def get_params_from_bucket(self, bucket_index: int) -> "tuple[int, float]":
        """Converts a bucket index into a concrete span length and mask probability.

        bucket `i` covers span lengths [2^i, 2^(i+1)); one is drawn uniformly from that
        range. mask_prob is held CONSTANT at base_mask_prob across every bucket.

        the docstring used to claim this line "adjust[s] mask_prob to keep the total number
        of masked tokens roughly constant". it does not, and it should not: mask_prob is
        already the fraction of tokens masked, so holding it fixed is precisely what keeps
        the masked-token budget constant while span length varies. the only thing that
        changes across buckets is how that fixed budget is CHUNKED -- many short spans vs
        few long ones -- which is the difficulty axis the curriculum is built on. the code
        was right and the comment was describing a correction it did not need.
        """
        min_len = int(2**bucket_index)
        max_len = int(2**(bucket_index + 1))

        # Uniformly sample a span length from within the bucket's range
        avg_span_length = int(np.random.randint(min_len, max_len))
        mask_prob = self.base_mask_prob

        return avg_span_length, mask_prob