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
    def __init__(self, config):
        super().__init__() 
        #query_dim = config["query_dim"] #don't even think about cross_attention
        self.dim = config["dim"]
        self.dim_head = config["dim_head"]
        self.heads = config["headcount"]
        self.weighted_skipnet = config["lambda"]
        self.denseproj_mul = config["ff_mult"]
        #self.naive_causal = config["is_causal_llm"]
        #...
        self.qknormalized_shape = [config["dim_head"],config["training_seqlen"],config["headcount"],config["dim_head"],]
        self.layerwisenorm = getnorm(config["layerwisenorm"],shape=self.dim)
        self.projnorm = getnorm(config["qknorm"],shape=self.qknormalized_shape)

        attn_inner_dim = self.dim_head * self.heads
        self.denseproj_inner_dim = self.dim * self.denseproj_mul

        self.rotary = rotarizer(self.dim_head)
        self.learnedlambda = nn.Parameter(torch.tensor(1.0))    #my beloved
        self.fused_swiglu_dim = self.denseproj_inner_dim*2   #this is necessary so the swiglu's two projections can be applied as a single operation.
        self.scale = self.dim_head**-0.5 #this is the 's' in 's'dpa! #exposed for cosine attention reasons!
        if config["qknorm"] == "l2norm":    #bootleg cosine attention by overloading the scale term in sdpa
            self.scale = nn.Parameter(torch.tensor(log(config["training_seqlen"]**2-config["training_seqlen"])))

        #...
        self.queryproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.keyproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.valueproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.attnoutproj = nn.Linear(in_features=attn_inner_dim, out_features=self.dim, bias=True)

        #dense ('mlp', 'feedforward', 'fully connected', ...) unit
        self.fused_denseproj_in = nn.Linear(in_features=self.dim, out_features=self.fused_swiglu_dim, bias=True) #this is the vit22b part
        self.dense_swiggy = swiglu() #this is kind of superfluous but this is pedagogical programming!
        self.denseproj_out = nn.Linear(in_features=self.denseproj_inner_dim, out_features=self.dim, bias=True)

    #[x]
    def self_attn(self, x, bat_len, seq_len):
        #norm -> {qkvproj -> qknorm{?}
        #reshape_h_d -> attn -> reshape_d_h} -> attnoutproj
        #project
        query   = self.queryproj(x)
        key     = self.keyproj(x)
        value   = self.valueproj(x)

        #reshape to bundled up matmul formme
        #query   = reshape_heads_dim(self.heads, query)
        #key     = reshape_heads_dim(self.heads, key)
        #value   = reshape_heads_dim(self.heads, value)
        #alternate reshape for compatibility with modded-nanogpt roformer
        query   = query.view(bat_len, seq_len, self.heads, self.dim_head)
        key     = key.view(bat_len, seq_len, self.heads, self.dim_head)
        value   = value.view(bat_len, seq_len, self.heads, self.dim_head)

        #pos_emb suggested before qknorm re: kellerjordan re: @Grad62304977
        #but we get an error for the x.ndim assertion if we run this after reshaping. whoopsie!
        cos, sin = self.rotary(query)       #our rotary unit does the shape detection from states

        #qk*norm
        query   = self.projnorm(query)
        key     = self.projnorm(key)
        #query   = self.projnorm(query, (query.size(-1)))    #something about functional rmsnorm requiring normalized shapes.
        #key     = self.projnorm(key, (key.size(-1)))           

        #rotary embed after qknorm as suggested etc.
        query   = apply_rotarizer_emb(query, cos, sin)
        key     = apply_rotarizer_emb(key, cos, sin)

        #laser-attn goes here
        #...
        
        #if we were here to explain attention instead of projections and norms,
        #we would have written this in jax or a language that compiles well!
        #instead, to benefit from flash attention 2, we want to use torch SDPA!
        y = nn.functional.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), scale=self.scale, is_causal=True)

        #reshape scalars from folded position to unfolded position so the ribosome can read the messenger headrna
        #y = self.reshape_dim_heads(self.heads, y)
        #alternate reshape scalars
        y = y.transpose(1,2).contiguous().view_as(x) #thanks a bunch modded-nanogpt

        #laser-attn unscale goes here
        #...

        return self.attnoutproj(y)

    #[x]
    def feedfor(self,x):
        x = self.fused_denseproj_in(x)
        x = self.dense_swiggy(x)
        x = self.denseproj_out(x)
        return x

    #parallel forward from kingoflolz/mesh-transformer-jax/! check it out!!
    # "discovered by Wang et al + EleutherAI from GPT-J fame"
    def forward(self, h_states):
        # in trad dialect: b->batch, n,i,j,k,l,m,f,a,o -> sequentiality dims,  h->heads, d->embedding dim
        bat_len, seq_len, emb_dim = h_states.size()
        # ^ detritus from modded-nanogpt transpose implementation. profile later ig.

        # highly traditional pre layernorm
        inner_states = self.layerwisenorm(h_states)

        #crunchy parts
        attn_out = self.self_attn(inner_states, bat_len, seq_len)
        dense_out = self.feedfor(inner_states)
        if self.weighted_skipnet==True:
            skip_out = h_states*self.learnedlambda
        else:
            skip_out = h_states
        #output w/ unabstracted resnet
        return skip_out + dense_out + attn_out

def getnorm(type, shape=None):
    if type == "layernorm":
        return nn.LayerNorm(shape, elementwise_affine=True, bias=True)
    elif type == "layernorm-nobias":
        return nn.LayerNorm(shape, elementwise_affine=True, bias=False) #???
    elif type == "rmsnorm":
        return nn.RMSNorm(shape, elementwise_affine=False)
    elif type == "l2norm":
        return l2norm(shape) #un function
    elif type == "identitynorm":
        return identitynorm(shape)
    else:
        raise Exception("Not implemented")

def l2norm(row):    #haha
    return nn.functional.normalize(row, p=2, dim=-1)

def identitynorm(row):
    return nn.Identity(row)

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
        self.inv_freq = (base ** (torch.arange(0,dim,2).float() / dim))**-1
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


###
### modelwise config:
### {"vocab_size":8000, "num_layers":4}
### 
class PGPT_Lformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.lambdaformer = nn.ModuleDict(dict(
            what_the_embedder_doin = nn.Embedding(config["vocab_size"], config["dim"]),
            blocks = nn.ModuleList([vit22_tformer(config) for _ in range(config["num_layers"])])
        ))
        self.tokenpicker_head = nn.Linear(in_features=config["dim"], out_features=config["vocab_size"], bias=False)
        self.tokenpicker_head.weight.data.zero_() #re: @Grad62304977

    def forward(self, index, targets=None, return_logits=True):
        x = self.lambdaformer.what_the_embedder_doin(index) # get token embeddings
        x = nn.functional.rms_norm(x, (x.size(-1),)) #re: @Grad62304977
        for decoder in self.lambdaformer.blocks:
            x = decoder(x)
        x = nn.functional.rms_norm(x, (x.size(-1),)) #re: @Grad62304977
        
        if targets is not None:
            #grab some losses woooo
            logits  = self.tokenpicker_head(x)
            logits  = 30 * torch.tanh(logits / 30) # @Grad62304977
            logits  = logits.float() # use tf32/fp32 for logits
            loss    = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else: 
            #kellerjordan optimi
            logits  = self.tokenpicker_head(x[:, [-1], :])   # re: kj: note: using list [-1] to preserve the time dim
            logits  = 30 * torch.tanh(logits / 30) # @Grad62304977 
            logits  = logits.float() # use tf32/fp32 for logits
            loss    = None
        
        #an appeal to performance is made:
        if not return_logits:
            logits = None
        
        return logits, loss