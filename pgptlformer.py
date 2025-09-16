# Import necessary, revised, libraries
import torch
import torch.nn as nn
import torch.optim as optim

#dubious 
from torch.utils.data import DataLoader, TensorDataset

#hehe
import math

### note: modded_nanogpt.py is an more full container for a transformer block structure
### it should specify an encoder ("embedder") and decoder for autoregress on tinystories.
### 



### 2302.05442's qk-layernorm is layernorm without centering and biases omitted.
### this is not equivalent to applying rmsnorm to the lexical scope of layernorm,
### as rmsnorm (1910.07467) doesn't use the mean statistic to yield variance.
### profiling and benchmarking a p%-pvarnorm would be great further work!
###
### to reach the complete spec of 2302.05442, 
### noncentered nonbiased norms must be applied to projected q&k
###
### candidate default: cfig = 
### {"dim":768,"dim_head":128,"headcount":6,"ffmult":4,
### "lambda":False,"layerwisenorm":"layernorm","qknorm":"identitynorm"}
### candidate tinystories: 
### {"dim":256,"dim_head":32,"headcount":8,"ffmult":4,
### "lambda":True,"layerwisenorm":"rmsnorm","qknorm":"identitynorm"}
###
### 2401.14489 suggests GEneral Matrix Multiplication dim alignment.
### basically vocabsize%64:=0.
### and (emb_dim/heads)%2^k:=0 , for some integer k.
### and (batchsize*sequence_len)%2^k:=0 , for some integer k.
### this places the smallest possible seqlen at 64@bf16 and 128@fp8
###
### ...
### the swiglu returns to bite us. the presence of that doubled swiggy matrix does something!
### specifically. uh.
### actually because we cranked up the swiggy_dim by 2x, it follows all of our scaling rules
### lmao, lol, lol, lmao, etcetera.
class vit22_tformer(nn.Module):
    def __init__(self, config, is_decoder=False):
        super().__init__() 
        #query_dim = config["query_dim"] #don't even think about cross_attention
        self.dim = config["dim"]
        self.dim_head = config["dim_head"]
        self.heads = config["headcount"]
        self.weighted_skipnet = config["lambda"]
        self.denseproj_mul = config["ff_mult"]
        #self.naive_causal = config["is_causal_llm"]
        self.is_decoder = is_decoder
        #...
        #self.qknormalized_shape = [config["dim_head"],config["training_seqlen"],config["headcount"],config["dim_head"],]
        self.qknormalized_shape = [config["headcount"],config["dim_head"]]
        self.layerwisenorm = getnorm(config["layerwisenorm"],shape=self.dim)
        self.projnorm = getnorm(config["qknorm"],shape=self.qknormalized_shape)    

        attn_inner_dim = self.dim_head * self.heads
        self.denseproj_inner_dim = self.dim * self.denseproj_mul

        if "rotary_embedding_base" in config.keys():
            self.rotbase = config["rotary_embedding_base"]
        else:
            self.rotbase = 1000 # hehe

        self.attention_II = None
        if "attention_deux" in config.keys():
            self.attention_II = True

    
        self.rotary = rotarizer(self.dim_head, base=self.rotbase)
        self.learnedlambda = nn.Parameter(torch.tensor(1.0))    #my beloved
        self.fused_swiglu_dim = self.denseproj_inner_dim*2   #this is necessary so the swiglu's two projections can be applied as a single operation.
        self.scale = self.dim_head**-0.5 #this is the 's' in 's'dpa! #exposed for cosine attention reasons!
        self.l2normscale = None
        if config["qknorm"] == "l2norm":    #bootleg cosine attention by overloading the scale term in sdpa
            self.l2normscale = nn.Parameter(torch.log(torch.tensor(config["training_seqlen"]**2)-torch.tensor(config["training_seqlen"])))

        #...
        self.queryproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.keyproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.valueproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.attnoutproj = nn.Linear(in_features=attn_inner_dim, out_features=self.dim, bias=True)

        if self.attention_II:
            self.queryBproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
            self.keyBproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)

        # --- NEW: CROSS-ATTENTION layers (only if it's a decoder block) ---
        if self.is_decoder:
            self.cross_attn_norm = getnorm(config["layerwisenorm"], shape=self.dim)
            self.cross_q_proj = nn.Linear(self.dim, self.dim, bias=False)
            self.cross_kv_proj = nn.Linear(self.dim, self.dim * 2, bias=False) # Project K and V together
            self.cross_out_proj = nn.Linear(self.dim, self.dim, bias=True)

        #dense ('mlp', 'feedforward', 'fully connected', ...) unit
        self.fused_denseproj_in = nn.Linear(in_features=self.dim, out_features=self.fused_swiglu_dim, bias=True) #this is the vit22b part
        self.dense_swiggy = swiglu() #this is kind of superfluous but this is pedagogical programming!
        self.denseproj_out = nn.Linear(in_features=self.denseproj_inner_dim, out_features=self.dim, bias=True)

    #[x]
    def self_attn(self, x, attention_mask):
        #norm -> {qkvproj -> qknorm{?}
        #reshape_h_d -> attn -> reshape_d_h} -> attnoutproj
        #project
        bat_len, seq_len, emb_dim = x.size()
        query   = self.queryproj(x)
        key     = self.keyproj(x)
        value   = self.valueproj(x)

        if self.attention_II:
            biasquery   = self.queryBproj(x)
            biaskey     = self.keyBproj(x)

        
        #reshape to bundled up matmul formme
        #query   = reshape_heads_dim(self.heads, query)
        #key     = reshape_heads_dim(self.heads, key)
        #value   = reshape_heads_dim(self.heads, value)
        #alternate reshape for compatibility with modded-nanogpt roformer
        query   = query.view(bat_len, seq_len, self.heads, self.dim_head)
        key     = key.view(bat_len, seq_len, self.heads, self.dim_head)
        value   = value.view(bat_len, seq_len, self.heads, self.dim_head)

        if self.attention_II:
            biasquery   = biasquery.view(bat_len, seq_len, self.heads, self.dim_head)
            biaskey     = biaskey.view(bat_len, seq_len, self.heads, self.dim_head)

        #pos_emb suggested before qknorm re: kellerjordan re: @Grad62304977
        #but we get an error for the x.ndim assertion if we run this after reshaping. whoopsie!
        cos, sin = self.rotary(query)       #our rotary unit does the shape detection from states

        #qk*norm
        query   = self.projnorm(query)
        key     = self.projnorm(key)

        if self.attention_II:
            biasquery   = self.projnorm(biasquery)
            biaskey     = self.projnorm(biaskey)

        #rotary embed after qknorm as suggested etc.
        query   = apply_rotarizer_emb(query, cos, sin)
        key     = apply_rotarizer_emb(key, cos, sin)

        if self.attention_II:
            biasquery   = apply_rotarizer_emb(biasquery, cos, sin)
            biaskey     = apply_rotarizer_emb(biaskey, cos, sin)

        #laser-attn goes here
        #...
        
        #if we were here to explain attention instead of projections and norms,
        #we would have written this in jax or a language that compiles well!
        #instead, to benefit from flash attention 2, we want to use torch SDPA!
        if self.l2normscale is not None:
            y = self.l2normscale*nn.functional.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), scale=1, is_causal=True)
        else:
            y = nn.functional.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2),
            attn_mask=attention_mask[:, None, :, :], # SDPA expects [B, H, T, T]
            scale=self.scale, 
            #is_causal=True 
            #switch from is_causal notation to explicit attn mask
            )

        if self.attention_II:
            #REV1
            bias_mask = attention_mask.to(query.dtype)
            dud = torch.ones_like(value, dtype=query.dtype, device=query.device)
            y = y + scaled_dot_product_attn_bias(   #~~attempt to reuse whatever efficient kernels we have already~~ nvm
                biasquery.transpose(1,2) , biaskey.transpose(1,2) , dud.transpose(1,2),
                attn_mask=bias_mask, # Pass the float mask here
                scale=self.scale,
                #is_causal=True
                #switch from is_causal notation to explicit attn mask
                )

        #reshape scalars from folded position to unfolded position so the ribosome can read the messenger headrna
        #y = self.reshape_dim_heads(self.heads, y)
        #alternate reshape scalars
        y = y.transpose(1,2).contiguous().view_as(x) #thanks a bunch modded-nanogpt

        #laser-attn unscale goes here
        #...

        return self.attnoutproj(y)
    
    #[?] --- NEW: CROSS-ATTENTION METHOD ---
    def cross_attn(self, x, encoder_hidden_states, cross_attention_mask):
        B, T_decoder, C = x.shape
        T_encoder = encoder_hidden_states.shape[1]

        # Query from the decoder's state, K/V from the encoder's state
        q = self.cross_q_proj(x)
        kv = self.cross_kv_proj(encoder_hidden_states)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T_decoder, self.heads, self.dim_head)
        k = k.view(B, T_encoder, self.heads, self.dim_head)
        v = v.view(B, T_encoder, self.heads, self.dim_head)

        # No rotary embeddings on cross-attention
        q = self.projnorm(q)
        k = self.projnorm(k)

        # Perform attention
        y = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=cross_attention_mask[:, None, :, :], # Expand for heads
            scale=self.scale
        )

        y = y.transpose(1, 2).contiguous().view(B, T_decoder, C)
        return self.cross_out_proj(y)

    #[x]
    def feedfor(self,x):
        x = self.fused_denseproj_in(x)
        x = self.dense_swiggy(x)
        x = self.denseproj_out(x)
        return x

    #parallel forward from kingoflolz/mesh-transformer-jax/! check it out!!
    # "discovered by Wang et al + EleutherAI from GPT-J fame"
    def forward(self, h_states, attention_mask=None, cross_attention_mask=None, encoder_hidden_states=None):
        # in trad dialect: b->batch, n,i,j,k,l,m,f,a,o -> sequentiality dims,  h->heads, d->embedding dim
        # bat_len, seq_len, emb_dim = h_states.size()
        # ^ detritus from modded-nanogpt transpose implementation. profile later ig.
        # highly traditional pre layernorm
        inner_states = self.layerwisenorm(h_states)

        #crunchy parts
        attn_out = self.self_attn(inner_states, attention_mask)
                
        # 2. Cross-Attention (only if this is a decoder and encoder states are provided)
        if self.is_decoder and encoder_hidden_states is not None:
            #cross_attn_in = self.cross_attn_norm(h_states) # Norm before cross-attn
            #nice try gemini 2.5! i won't surrender so easily
            cross_attn_out = self.cross_attn(inner_states, encoder_hidden_states, cross_attention_mask)

        dense_out = self.feedfor(inner_states)
        if self.weighted_skipnet==True:
            skip_out = h_states*self.learnedlambda
        else:
            skip_out = h_states
        #output w/ unabstracted resnet
        if self.is_decoder and encoder_hidden_states is not None:
            #HYA! PARALLEL X-ATTN & S-ATTN! THE FORBIDDEN TECHNIQUE!
            return skip_out + dense_out + attn_out + cross_attn_out
        else:
            return skip_out + dense_out + attn_out

def getnorm(type, shape=None):
    if type == "layernorm":
        return nn.LayerNorm(shape, elementwise_affine=True, bias=True)
    elif type == "layernorm-nobias":
        return nn.LayerNorm(shape, elementwise_affine=True, bias=False) #???
    elif type == "rmsnorm":
        return nn.RMSNorm(shape, elementwise_affine=False)
    elif type == "dynamic_shape_rmsnorm":
        return dynamic_shape_rmsnorm()
    elif type == "dynamic_shape_layernorm":
        return dynamic_shape_layernorm()
    elif type == "l2norm":
        return l2norm() #un function
    elif type == "identitynorm":
        return identitynorm(shape)
    else:
        raise Exception("Not implemented")

class l2norm(nn.Module):    #haha
    def forward(self, inputter, **kwargs):
        inputter = nn.functional.normalize(inputter, p=2, dim=-1)
        return inputter

def identitynorm(row):
    return nn.Identity(row)

#from `questions/76067020/`` lol
class dynamic_shape_rmsnorm(nn.Module):
    def forward(self, inputter, **kwargs):
        inputter = inputter.transpose(1,2)  #rotate!
        #i am so sorry haha
        #normalized_shape seems to require adjacencies, i tried a few other things first.
        #wait the notation in the paper suggests... [3:].
        inner_shape = inputter.size()[3:]   

        inputter = nn.functional.rms_norm(inputter, normalized_shape=inner_shape, **kwargs)   
        inputter = inputter.transpose(1,2)                  #reverse rotate!
        return inputter

class dynamic_shape_layernorm(nn.Module):
    def forward(self, inputter, **kwargs):
        inputter = inputter.transpose(1,2)  #rotate!
        #i am so sorry haha
        #normalized_shape seems to require adjacencies, i tried a few other things first.
        #wait the notation in the paper suggests... [3:].
        inner_shape = inputter.size()[3:]   

        inputter = nn.functional.layer_norm(inputter, normalized_shape=inner_shape, **kwargs)   
        inputter = inputter.transpose(1,2)                  #reverse rotate!
        return inputter

#we too are hitting that mfing noam shazeer https://arxiv.org/pdf/2002.05202 
#if there was a self-gated ELU id want to use it instead though
class swiglu(nn.Module):
     def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return nn.functional.silu(gate) * x

#rippin this one from modded-nanogpt
class rotarizer(nn.Module):
    def __init__(self, dim, base=1000): #shhh don't tell anyone about the rotemb base
        super().__init__()
        inv_freq = (base ** (torch.arange(0,dim,2).float() / dim))**-1
        self.register_buffer("inv_freq", inv_freq, persistent=False) # persistent=False is recommended for non-stateful buffers

        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.size()[1]   #perform the surgical LENGTH,YOINKEMS,{}, b {n} h d
                                #using torch tensor.size()[idx] notation bc i think it is more explicit than shape[]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            reg_freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = reg_freqs.cos().bfloat16()
            self.sin_cached = reg_freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :] 
        #yeah slay em with the list comprehensions, cited author 😒

def apply_rotarizer_emb(x, cos, sin):
    #assert x.ndim == 4  # b n h d 
    d = x.size()[3]//2     # perform the superb DIVIDE,2,LENGTH,YOINKEMS,{}, b n h {d}
    x1 = x[..., :d]     #some kind of slicing mystery code
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1,y2], 3).type_as(x)

#alternate attention to retrieve a shift matrix instead of scale matrix.
#this will either break the first time it runs or make perfect sense whomstdve doubted it all along
#REVISION 1:
#expect to pass this a 'dud' matrix of 1s
#"""
def scaled_dot_product_attn_bias(query, key, value, attn_mask=None, dropout_p=0.0,
    is_causal=False, scale=None, enable_gqa=False):
    #make sure you compile this or it will be slow! haha! it will be slow otherwise! haha!
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale 
    if enable_gqa:  #who can say what this does
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_magnitude = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if attn_mask is not None:
        # The attn_mask is [B, T, T]. attn_magnitude is [B, H, T, T].
        # We unsqueeze the mask to make them broadcast-compatible.
        attn_magnitude *= attn_mask[:, None, :, :]
    #attn_magnitude = torch.softmax(attn_weight, dim=-1)   we dont want this lol
    attn_magnitude = torch.dropout(attn_magnitude, dropout_p, train=True)
    return attn_magnitude @ value

### states take format batch, sequence, embedding
### therefore 
### batch_size, sequence_length, embedding_dim = h_states.shape
def reshape_heads_dim(heads, tensor):
    bat_len, seq_len, emb_dim = tensor.size()
    head_len = heads
    # i think equivalent to traditional
    # "b n (h d) -> b h n d"
    tensor = tensor.reshape(bat_len , seq_len, head_len, emb_dim // head_len)
    tensor = tensor.permute(0, 2, 1, 3).reshape(bat_len*head_len, seq_len, emb_dim // head_len)
    return tensor

def reshape_dim_heads(heads, tensor):
    bat_len, seq_len, emb_dim = tensor.size()
    head_len = heads
    # i think equivalent to traditional
    # "b h n d -> b n (h d)"
    tensor = tensor.reshape(bat_len // head_len, head_len, seq_len, emb_dim)
    tensor = tensor.permute(0, 2, 1, 3).reshape(bat_len // head_len, seq_len, emb_dim*head_len)
    return tensor

def create_attention_mask(padding_mask, is_causal):
    """
    Creates a boolean attention mask from a pre-computed padding mask.
    - `padding_mask` is a [B, T] boolean tensor (True for real tokens).
    - `is_causal` controls the application of a causal mask.
    
    Returns a boolean mask of shape [B, T, T] where True means "attend".
    """
    B, T = padding_mask.shape
    
    # 1. Expand padding mask to [B, T, T] for key visibility.
    # A query at position `q` can attend to a key at position `k` if padding_mask[k] is True.
    attention_mask = padding_mask[:, None, :].expand(B, T, T)
    
    if is_causal:
        # 2. Get a causal mask of shape [T, T]
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=padding_mask.device))
        
        # 3. Combine them. The final value is True only if both masks are True.
        attention_mask = attention_mask & causal_mask[None, :, :]
        
    return attention_mask

###
### modelwise config:
### {"vocab_size":8000, "num_layers":4}
### 
class PGPT_Lformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.is_t5 = config.get("is_t5", False)
        
        self.what_the_embedder_doin = nn.Embedding(config["vocab_size"], config["dim"])

        if self.is_t5:
            # T5 uses two separate stacks
            self.encoder = nn.ModuleList([vit22_tformer(config, is_decoder=False) for _ in range(config["num_layers"])])
            self.decoder = nn.ModuleList([vit22_tformer(config, is_decoder=True) for _ in range(config["num_layers"])])
            # We need to modify vit22_tformer to accept encoder_hidden_states for cross-attention
        else:
            # Autoregressive blocks are not 'decoders' t. 'gemini2.5'
            self.lambdaformer = nn.ModuleDict(dict(
                blocks = nn.ModuleList([vit22_tformer(config, is_decoder=False) for _ in range(config["num_layers"])])
            ))
        self.tokenpicker_head = nn.Linear(in_features=config["dim"], out_features=config["vocab_size"], bias=False)
        self.tokenpicker_head.weight.data.zero_() #re: @Grad62304977
        """
        self.chatbot_tokenpicker_head = nn.Linear(in_features=config["dim"], out_features=config["vocab_size"], bias=False)
        self.chatbot_tokenpicker_head.weight.data.zero_() #re: @Grad62304977
        """
    def forward(self, *args, **kwargs):
        if self.is_t5:
            # T5-style forward pass
            return self.forward_t5(*args, **kwargs)
        else:
            # Original autoregressive forward pass
            return self.forward_arg(*args, **kwargs)

    def forward_arg(self, input_ids, targets=None, padding_mask=None, return_logits=True, return_zloss=False, chatbot=False):
        if padding_mask is None:
            # Assume no padding if mask isn't provided
            padding_mask = torch.ones_like(input_ids)
        # is_causal=True for autoregressive model
        attn_mask = create_attention_mask(padding_mask, is_causal=True)

        x = self.what_the_embedder_doin(input_ids) # get token embeddings
        x = nn.functional.rms_norm(x, (x.size(-1),)) #re: @Grad62304977
        for decoder in self.lambdaformer.blocks:
            x = decoder(x, attention_mask=attn_mask)
        x = nn.functional.rms_norm(x, (x.size(-1),)) #re: @Grad62304977
        pad_token_id = self.config.get("pad_token_id")
        if targets is not None:
            #grab some losses woooo
            logits  = self.tokenpicker_head(x)
            if return_zloss: #tracking https://arxiv.org/abs/2309.14322 
                z = torch.sum(torch.exp(logits)) #reduce: e^logit[j]
                z_loss = torch.log(z)**2 #log and square Z. make sure to set a coefficient in trainer!
            logits  = 30 * torch.tanh(logits / 30) # @Grad62304977
            logits  = logits.float() # use tf32/fp32 for logits
            #loss    = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id )
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
            loss_per_token = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_per_token = loss_per_token.view(logits.size(0), logits.size(1)) # Reshape to [B, T]

            # Calculate the mean loss for each sequence in the batch
            num_real_tokens = (targets != pad_token_id).sum(dim=1)
            loss_per_sequence = loss_per_token.sum(dim=1) / num_real_tokens.clamp(min=1) # Shape: [B]
            loss = loss_per_sequence.mean()
        else: 
            #kellerjordan optimi
            logits  = self.tokenpicker_head(x[:, [-1], :])   # re: kj: note: using list [-1] to preserve the time dim
            logits  = 30 * torch.tanh(logits / 30) # @Grad62304977 
            logits  = logits.float() # use tf32/fp32 for logits
            loss    = None
        
        #an appeal to performance is made:
        if not return_logits:
            logits = None
        if not return_zloss:
            z_loss = None
        
        return logits, loss, z_loss, loss_per_sequence

    def forward_t5(self, input_ids, decoder_input_ids, targets, encoder_padding_mask, decoder_padding_mask, return_logits=True, return_zloss=False):
        # 1. Create ENCODER mask (bidirectional + padding)
        # The model receives the padding_mask and determines causality internally.
        encoder_attn_mask = create_attention_mask(encoder_padding_mask, is_causal=False)
        encoder_hidden_states = self.what_the_embedder_doin(input_ids)
        encoder_hidden_states = nn.functional.rms_norm(encoder_hidden_states, (encoder_hidden_states.size(-1),)) #re: @Grad62304977
        for block in self.encoder:
            encoder_hidden_states = block(encoder_hidden_states,  attention_mask=encoder_attn_mask)
        encoder_hidden_states = nn.functional.rms_norm(encoder_hidden_states, (encoder_hidden_states.size(-1),)) #re: @Grad62304977

        # 2. Decode to reconstruct the original text
        # The decoder uses causal self-attention and cross-attends to the encoder output
        decoder_self_attn_mask = create_attention_mask(decoder_padding_mask, is_causal=True)
        # Create the non-square cross-attention mask
        # basically shape is [B, T_decoder, T_encoder] which is slightly different or something
        cross_attn_mask = encoder_padding_mask[:, None, :].expand(
            -1, decoder_input_ids.shape[1], -1
        )
        # Cross-attention mask is just the encoder's padding, but shaped for the decoder's queries
        decoder_hidden_states = self.what_the_embedder_doin(decoder_input_ids) #re: @Grad62304977
        for block in self.decoder:
            # Block must be modified to take encoder_hidden_states and perform cross-attention
            decoder_hidden_states = block(decoder_hidden_states,
            attention_mask=decoder_self_attn_mask,
            cross_attention_mask=cross_attn_mask,
            encoder_hidden_states=encoder_hidden_states)
        decoder_hidden_states = nn.functional.rms_norm(decoder_hidden_states, (decoder_hidden_states.size(-1),)) #re: @Grad62304977
        pad_token_id = self.config.get("pad_token_id")
        if targets is not None:
            logits = self.tokenpicker_head(decoder_hidden_states)
            if return_zloss: #tracking https://arxiv.org/abs/2309.14322 
                z = torch.sum(torch.exp(logits)) #reduce: e^logit[j]
                z_loss = torch.log(z)**2 #log and square Z. make sure to set a coefficient in trainer!
            logits  = 30 * torch.tanh(logits / 30) # @Grad62304977
            logits  = logits.float() # use tf32/fp32 for logits
            #loss    = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=pad_token_id )
            loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
            loss_per_token = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_per_token = loss_per_token.view(logits.size(0), logits.size(1)) # Reshape to [B, T]

            # Calculate the mean loss for each sequence in the batch
            num_real_tokens = (targets != pad_token_id).sum(dim=1)
            loss_per_sequence = loss_per_token.sum(dim=1) / num_real_tokens.clamp(min=1) # Shape: [B]
            loss = loss_per_sequence.mean()
        else:
            logits  = self.tokenpicker_head(x[:, [-1], :])  # re: kj:
            logits  = 30 * torch.tanh(logits / 30) # @Grad62304977 
            logits  = logits.float() # use tf32/fp32 for logits
            loss    = None
            loss_per_sequence = None
        return logits, loss, z_loss, loss_per_sequence

    # --- NEW: Method for efficient T5 encoding ---
    @torch.no_grad()
    def encode(self, input_ids, encoder_padding_mask):
        """
        Runs the encoder once to generate the memory for the decoder.
        """
        assert self.is_t5, "encode() is only for T5 models"
        encoder_attn_mask = create_attention_mask(encoder_padding_mask, is_causal=False)
        encoder_hidden_states = self.what_the_embedder_doin(input_ids)
        encoder_hidden_states = nn.functional.rms_norm(encoder_hidden_states, (encoder_hidden_states.size(-1),))
        for block in self.encoder:
            encoder_hidden_states = block(encoder_hidden_states, attention_mask=encoder_attn_mask)
        encoder_hidden_states = nn.functional.rms_norm(encoder_hidden_states, (encoder_hidden_states.size(-1),))
        return encoder_hidden_states

    # --- NEW: Method for a single T5 decoding step ---
    @torch.no_grad()
    def decode_step(self, decoder_input_ids, encoder_hidden_states, encoder_padding_mask):
        """
        Runs the decoder for a single step of autoregressive generation.
        """
        assert self.is_t5, "decode_step() is only for T5 models"
        
        # Create masks for this specific step
        decoder_padding_mask = (decoder_input_ids != self.config['pad_token_id'])
        decoder_self_attn_mask = create_attention_mask(decoder_padding_mask, is_causal=True)
        cross_attn_mask = encoder_padding_mask[:, None, :].expand(-1, decoder_input_ids.shape[1], -1)

        # Get embeddings for the current decoder sequence
        decoder_hidden_states = self.what_the_embedder_doin(decoder_input_ids)
        for block in self.decoder:
            decoder_hidden_states = block(
                decoder_hidden_states,
                attention_mask=decoder_self_attn_mask,
                cross_attention_mask=cross_attn_mask,
                encoder_hidden_states=encoder_hidden_states
            )
        decoder_hidden_states = nn.functional.rms_norm(decoder_hidden_states, (decoder_hidden_states.size(-1),))
        
        # Get logits for the *very last* token in the sequence
        logits = self.tokenpicker_head(decoder_hidden_states[:, [-1], :])
        return logits