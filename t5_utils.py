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
        # batch_x is a tensor of shape [B, T] from our IntelligentAudioLoader
        B, T = batch_x.shape
        
        masked_inputs = []
        target_labels = []

        for i in range(B):
            sequence = batch_x[i].tolist()
            
            input_seq, target_seq = self._create_masked_sequence(sequence)
            
            masked_inputs.append(torch.tensor(input_seq, dtype=torch.long))
            target_labels.append(torch.tensor(target_seq, dtype=torch.long))
        
        # Pad the batches to be of the same length
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            masked_inputs, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            target_labels, batch_first=True, padding_value=self.pad_token_id # Pad with pad_id first
        )

        # --- NEW: Create masks BEFORE returning ---
        # The mask is simply where the tensor is NOT the pad token. Shape: [B, T]
        encoder_padding_mask = (padded_inputs != self.pad_token_id)
        decoder_padding_mask = (padded_labels != self.pad_token_id)

        # Prepare labels for loss function (the only place the -100 sentinel is now used)
        # dayumn don't do that actually.
        # padded_labels[padded_labels == self.pad_token_id] = -100

        # The decoder input is the shifted labels, starting with a pad token
        decoder_inputs = torch.roll(padded_labels, shifts=1, dims=1)
        decoder_inputs[:, 0] = self.pad_token_id
        # We must also update its mask to reflect this shift
        decoder_padding_mask = torch.roll(decoder_padding_mask, shifts=1, dims=1)
        decoder_padding_mask[:, 0] = self.pad_token_id # The first position is now padding
        return padded_inputs, decoder_inputs, padded_labels, encoder_padding_mask, decoder_padding_mask

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
        
        target_tokens.append(self.mask_token_start_id + mask_token_id_counter) # Final sentinel
        
        return input_tokens, target_tokens