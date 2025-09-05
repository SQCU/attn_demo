class simple_shimformer(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.dim = config["dim"]
        self.dim_head = config["dim_head"]
        self.heads = config["headcount"]
        self.qknormalized_shape = [config["headcount"],config["dim_head"]]
        self.layerwisenorm = getnorm(config["layerwisenorm"],shape=self.dim)
        self.projnorm = getnorm(config["qknorm"],shape=self.qknormalized_shape)    

        attn_inner_dim = self.dim_head * self.heads
        self.denseproj_inner_dim = self.dim * self.denseproj_mul

        if "rotary_embedding_base" in config.keys():
            self.rotbase = config["rotary_embedding_base"]
        else:
            self.rotbase = 1000 # hehe

        self.scale = self.dim_head**-0.5 #this is the 's' in 's'dpa! #exposed for cosine attention reasons!
        self.l2normscale = None
        if config["qknorm"] == "l2norm":    #bootleg cosine attention by overloading the scale term in sdpa
            self.l2normscale = nn.Parameter(torch.log(torch.tensor(config["training_seqlen"]**2)-torch.tensor(config["training_seqlen"])))

        #...
        self.queryproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.keyproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.valueproj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=False)
        self.attnoutproj = nn.Linear(in_features=attn_inner_dim, out_features=self.dim, bias=True)

    def self_attn(self, x, bat_len, seq_len):
        #norm -> {qkvproj -> qknorm{?}
        #reshape_h_d -> attn -> reshape_d_h} -> attnoutproj
        #project
        query   = self.queryproj(x)
        key     = self.keyproj(x)
        value   = self.valueproj(x)
       
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

        #rotary embed after qknorm as suggested etc.
        query   = apply_rotarizer_emb(query, cos, sin)
        key     = apply_rotarizer_emb(key, cos, sin)

        if self.l2normscale is not None:
            y = self.l2normscale*nn.functional.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), scale=1, is_causal=True)
        else:
            y = nn.functional.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), scale=self.scale, is_causal=True)

        y = y.transpose(1,2).contiguous().view_as(x)

        return self.attnoutproj(y)

    def forward(self, h_states):
        # in trad dialect: b->batch, n,i,j,k,l,m,f,a,o -> sequentiality dims,  h->heads, d->embedding dim
        bat_len, seq_len, emb_dim = h_states.size()
        # ^ detritus from modded-nanogpt transpose implementation. profile later ig.

        # highly traditional pre layernorm
        inner_states = self.layerwisenorm(h_states)

        #crunchy parts
        attn_out = self.self_attn(inner_states, bat_len, seq_len)

        skip_out = h_states
        #output w/ unabstracted resnet
        return skip_out + attn_out


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

        nn.functional.rms_norm(inputter, normalized_shape=inner_shape, **kwargs)   
        inputter = inputter.transpose(1,2)                  #reverse rotate!
        return inputter

class dynamic_shape_layernorm(nn.Module):
    def forward(self, inputter, **kwargs):
        inputter = inputter.transpose(1,2)  #rotate!
        #i am so sorry haha
        #normalized_shape seems to require adjacencies, i tried a few other things first.
        #wait the notation in the paper suggests... [3:].
        inner_shape = inputter.size()[3:]   

        nn.functional.layer_norm(inputter, normalized_shape=inner_shape, **kwargs)   
        inputter = inputter.transpose(1,2)                  #reverse rotate!
        return inputter

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




#complex example from our cuda-optimized network design
"""
layer_prefab = {"dim":768,"dim_head":64,"headcount":12,"ff_mult":4, 
"lambda":True,"layerwisenorm":"rmsnorm","qknorm":"dynamic_shape_rmsnorm", 
"attention_deux":True, "training_seqlen":args.sequence_length}
global_prefab = {"vocab_size":50304, "num_layers":4}
config = {}
config.update(layer_prefab)
config.update(global_prefab)

"""
#simpler:
layer_prefab = {"dim":32,"dim_head":8,"headcount":4, 
"layerwisenorm":"rmsnorm","qknorm":"dynamic_shape_rmsnorm", 
"training_seqlen":12}
config = {}
config.update(layer_prefab)

sample_states = torch.randn((
    1,
    config["training_seqlen"],
    config["headcount"],
    config["dim"],))

