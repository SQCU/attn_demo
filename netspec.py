# Import necessary, revised, libraries
import torch
import torch.nn as nn
import torch.optim as optim

#dubious 
from torch.utils.data import DataLoader, TensorDataset

### note: netspec.py is an empty container for a transformer block structure
### it has no specification for an embedder or decoder to yield a trainable model.
### preserved for reference reasons, esp. wrt cross-attn and fused projection stuff.



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
        self.layerwisenorm = getnorm(config["norm"])
        self.projnorm = getnorm(config["qknorm"])
        self.denseproj_mul = config["ff_mult"]
        self.naive_causal = config["is_causal_llm"]
        #...

        attn_inner_dim = self.dim_head * self.heads
        denseproj_inner_dim = dim * denseproj_mul

        self.learnedlambda = nn.Parameter(torch.tensor(1.0))    #my beloved
        self.fused_swiglu_dim = self.dim*2   #this is necessary so the swiglu's two projections can be applied as a single operation.
        self.scale = dim_head**-0.5 #this is the 's' in 's'dpa! #exposed for cosine attention reasons!

        #...
        self.queryproj = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.keyproj = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.valueproj = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.attnoutproj = nn.Linear(in_features=attn_inner_dim, out_features=dim, bias=True)

        # okay don't implement fused projections to minimize complexity of reshapes.
        # if you do fuse all of the linear projections into a single operation, training is 15% faster btw.

        #dense ('mlp', 'feedforward', 'fully connected', ...) unit
        self.fused_denseproj_in = nn.Linear(in_features=dim, out_features=self.fused_swiglu_dim, bias=True) #this is the vit22b part
        self.dense_swiggy = swiglu() #this is kind of superfluous but this is pedagogical programming!
        self.denseproj_out = nn.Linear(in_features=denseproj_inner_dim, out_features=dim, bias=True)

        #def skiplambdaforgreatjustice(self, x)
        #nope. incorporate this into the forward.

        #[x]
    def self_attn(self, x, attn_bias=None):
        #norm -> {qkvproj -> qknorm{?}
        #reshape_h_d -> attn -> reshape_d_h} -> attnoutproj
        #project
        query   = self.queryproj(x)
        key     = self.keyproj(x)
        value   = self.valueproj(x)

        #reshape
        query   = reshape_heads_dim(self.heads, query)
        key     = reshape_heads_dim(self.heads, key)
        value   = reshape_heads_dim(self.heads, value)

        #qk*norm
        query   = self.projnorm(query)
        key     = self.projnorm(key)

        #laser-attn goes here
        #...
        
        #if we were here to explain attention instead of projections and norms,
        #we would have written this in jax or a language that compiles well!
        #instead, to benefit from flash attention 2, we want to use torch SDPA!
        if self.naive_causal:
            y = nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            y = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias)

        #reshape
        y = self.reshape_dim_heads(self.heads, y)

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
        #seq_len = h_states.size()[1]

        # highly traditional pre layernorm
        inner_states = self.layerwisenorm(h_states)

        #crunchy parts
        attn_out = self.self_attn(inner_states)
        dense_out = self.feedfor(inner_states)
        if self.weighted_skipnet==True:
            skip_out = h_states*learnedlambda
        else:
            skip_out = h_states
        #output w/ unabstracted resnet
        return skip_out + dense_out + attn_out

def getnorm(type, shape):
    if type == "layernorm":
        return nn.LayerNorm(shape, elementwise_affine=True, bias=True)
    elif type == "layernorm-nobias":
        return nn.LayerNorm(shape, elementwise_affine=True, bias=False) #???
    elif type == "rmsnorm":
        return nn.RMSNorm(shape, elementwise_affine=False)
    elif type == "l2norm":
        return l2norm #un function
    elif type == "identitynorm":
        return identitynorm
    else:
        raise Exception("Not implemented")


#we too are hitting that mfing noam shazeer https://arxiv.org/pdf/2002.05202 
#if there was a self-gated ELU id want to use it instead though
class swiglu(nn.module):
     def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

def l2norm(row):    #haha
    return nn.functional.normalize(row, p=2, dim=-1)

def identitynorm(row):
    return nn.Identity(row)

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