# t5_utils.py
import numpy as np
import torch
class T5BatchProcessor:
    def __init__(self, mask_token_start_id, pad_token_id, eos_token_id, mask_prob=0.15, avg_span_length=3):
        self.mask_token_start_id = mask_token_start_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.mask_prob = mask_prob
        self.avg_span_length = avg_span_length
        print("T5BatchProcessor initialized.")

    def __call__(self, batch_x):
        # batch_x is a tensor of shape [B, T]
        B, T = batch_x.shape
        
        masked_inputs_list = []
        raw_targets_list = []

        for i in range(B):
            sequence = batch_x[i].tolist()
            input_seq, target_seq = self._create_masked_sequence(sequence)
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

    def _create_masked_sequence(self, tokens):
        mask_indices = np.random.permutation(len(tokens))
        num_to_mask = int(len(tokens) * self.mask_prob)
        
        masked_spans = []
        covered_indices = set()
        
        # Decide which tokens to mask
        for idx in mask_indices:
            if idx in covered_indices:
                continue
            if len(covered_indices) >= num_to_mask:
                break
                
            span_len = np.random.poisson(self.avg_span_length)
            if span_len == 0: continue
            
            start_idx = idx
            end_idx = min(len(tokens), start_idx + span_len)
            
            # Add all indices in this span to the set of covered indices
            span_indices = set(range(start_idx, end_idx))
            if not span_indices.isdisjoint(covered_indices):
                continue # Avoid overlapping spans for simplicity
            
            masked_spans.append(tokens[start_idx:end_idx])
            covered_indices.update(span_indices)

        # Sort spans by their start index to process the sequence linearly
        # This requires storing the start index along with the span
        # For simplicity in this example, we'll assume a more direct replacement logic
        # A full implementation would be more robust here.

        # Create the final input and target sequences
        input_tokens = []
        target_tokens = []
        mask_token_id_counter = 0

        i = 0
        while i < len(tokens):
            if i in covered_indices:
                # This is the start of a masked span
                span_len = 0
                while (i + span_len) in covered_indices:
                    span_len += 1
                
                # Append a single mask token to the input
                mask_token = self.mask_token_start_id + mask_token_id_counter
                input_tokens.append(mask_token)
                
                # Append the mask token + original tokens + next mask token to the target
                target_tokens.append(mask_token)
                target_tokens.extend(tokens[i : i + span_len])
                
                mask_token_id_counter += 1
                i += span_len
            else:
                # This token is not masked
                input_tokens.append(tokens[i])
                i += 1
        
        target_tokens.append(self.eos_token_id)
        
        return input_tokens, target_tokens