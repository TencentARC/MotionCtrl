import math
from inspect import isfunction

import torch
import torch as th
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    
from lvdm.common import (
    checkpoint,
    exists,
    uniq,
    default,
    max_neg_value,
    init_
)
from lvdm.basics import (
    conv_nd,
    zero_module,
    normalization
)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# ---------------------------------------------------------------------------------------------------
class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(th.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = th.arange(length_q, device=device)
        range_vec_k = th.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = th.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        # final_mat = th.LongTensor(final_mat).to(self.embeddings_table.device)
        # final_mat = th.tensor(final_mat, device=self.embeddings_table.device, dtype=torch.long)
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class TemporalCrossAttention(nn.Module):
    def __init__(self, 
        query_dim, 
        context_dim=None, 
        heads=8, 
        dim_head=64, 
        dropout=0.,
        temporal_length=None,           # For relative positional representation and image-video joint training.
        image_length=None,              # For image-video joint training.
        use_relative_position=False,    # whether use relative positional representation in temporal attention.
        img_video_joint_train=False,    # For image-video joint training.
        use_tempoal_causal_attn=False,
        bidirectional_causal_attn=False,
        tempoal_attn_type=None,
        joint_train_mode="same_batch",
        **kwargs,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.context_dim = context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.temporal_length = temporal_length
        self.use_relative_position = use_relative_position
        self.img_video_joint_train = img_video_joint_train
        self.bidirectional_causal_attn = bidirectional_causal_attn
        self.joint_train_mode = joint_train_mode
        assert(joint_train_mode in ["same_batch", "diff_batch"])
        self.tempoal_attn_type = tempoal_attn_type

        if bidirectional_causal_attn:
            assert use_tempoal_causal_attn
        if tempoal_attn_type:
            assert(tempoal_attn_type in ['sparse_causal', 'sparse_causal_first'])
            assert(not use_tempoal_causal_attn) 
            assert(not (img_video_joint_train and (self.joint_train_mode == "same_batch"))) 
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        assert(not (img_video_joint_train and (self.joint_train_mode == "same_batch") and use_tempoal_causal_attn))
        if img_video_joint_train:
            if self.joint_train_mode == "same_batch":
                mask = torch.ones([1, temporal_length+image_length, temporal_length+image_length])
                # mask[:, image_length:, :] = 0
                # mask[:, :, image_length:] = 0
                mask[:, temporal_length:, :] = 0
                mask[:, :, temporal_length:] = 0
                self.mask = mask
            else:
                self.mask = None
        elif use_tempoal_causal_attn:
            # normal causal attn
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))
        elif tempoal_attn_type == 'sparse_causal':
            # all frames interact with only the `prev` & self frame
            mask1 = torch.tril(torch.ones([1, temporal_length, temporal_length])).bool() # true indicates keeping
            mask2 = torch.zeros([1, temporal_length, temporal_length]) # initialize to same shape with mask1
            mask2[:,2:temporal_length, :temporal_length-2] = torch.tril(torch.ones([1,temporal_length-2, temporal_length-2]))
            mask2=(1-mask2).bool() # false indicates masking
            self.mask = mask1 & mask2
        elif tempoal_attn_type == 'sparse_causal_first':
            # all frames interact with only the `first` & self frame
            mask1 = torch.tril(torch.ones([1, temporal_length, temporal_length])).bool() # true indicates keeping
            mask2 = torch.zeros([1, temporal_length, temporal_length])
            mask2[:,2:temporal_length, 1:temporal_length-1] = torch.tril(torch.ones([1,temporal_length-2, temporal_length-2]))
            mask2=(1-mask2).bool() # false indicates masking
            self.mask = mask1 & mask2
        else:
            self.mask = None

        if use_relative_position:
            assert(temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        nn.init.constant_(self.to_q.weight, 0)
        nn.init.constant_(self.to_k.weight, 0)
        nn.init.constant_(self.to_v.weight, 0)
        nn.init.constant_(self.to_out[0].weight, 0)
        nn.init.constant_(self.to_out[0].bias, 0)

    def forward(self, x, context=None, mask=None):
        # if context is None:
        #     print(f'[Temp Attn] x={x.shape},context=None')
        # else:
        #     print(f'[Temp Attn] x={x.shape},context={context.shape}')

        nh = self.heads
        out = x
        q = self.to_q(out)
        # if context is not None:
        #     print(f'temporal context 1 ={context.shape}')
        # print(f'x={x.shape}')
        context = default(context, x)
        # print(f'temporal context 2 ={context.shape}')
        k = self.to_k(context)
        v = self.to_v(context)
        # print(f'q ={q.shape},k={k.shape}')

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=nh), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.use_relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale # TODO check 
            sim += sim2
        # print('mask',mask)
        if exists(self.mask):
            if mask is None:
                mask = self.mask.to(sim.device)
            else:
                mask = self.mask.to(sim.device).bool() & mask #.to(sim.device)
        else:
            mask = mask            
            # if self.img_video_joint_train:
            #     # process mask (make mask same shape with sim)
            #     c, h, w = mask.shape
            #     c, t, s = sim.shape
            #     # assert(h == w and t == s),f"mask={mask.shape}, sim={sim.shape}, h={h}, w={w}, t={t}, s={s}"
            
            #     if h > t:
            #         mask = mask[:, :t, :]
            #     elif h < t: # pad zeros to mask (no attention) only initial mask =1 area compute weights
            #         mask_ = torch.zeros([c,t,w]).to(mask.device)
            #         mask_[:, :h, :] = mask
            #         mask = mask_
            #     c, h, w = mask.shape
            #     if w > s:
            #         mask = mask[:, :, :s]
            #     elif w < s: # pad zeros to mask
            #         mask_ = torch.zeros([c,h,s]).to(mask.device)
            #         mask_[:, :, :w] = mask
            #         mask = mask_
            
            # max_neg_value = -torch.finfo(sim.dtype).max
            # sim = sim.float().masked_fill(mask == 0, max_neg_value)
        if mask is not None:
            max_neg_value = -1e9
            sim = sim + (1-mask.float()) * max_neg_value # 1=masking,0=no masking
            # print('sim after masking: ', sim)
            
            # if torch.isnan(sim).any() or torch.isinf(sim).any() or (not sim.any()):
            #     print(f'sim [after masking], isnan={torch.isnan(sim).any()}, isinf={torch.isinf(sim).any()}, allzero={not sim.any()}')

        attn = sim.softmax(dim=-1)
        # print('attn after softmax: ', attn)
        # if torch.isnan(attn).any() or torch.isinf(attn).any() or (not attn.any()):
        #     print(f'attn [after softmax], isnan={torch.isnan(attn).any()}, isinf={torch.isinf(attn).any()}, allzero={not attn.any()}')
        
        # attn = torch.where(torch.isnan(attn), torch.full_like(attn,0), attn)
        # if torch.isinf(attn.detach()).any():
        #     import pdb;pdb.set_trace()
        # if torch.isnan(attn.detach()).any():
        #     import pdb;pdb.set_trace()
        out = einsum('b i j, b j d -> b i d', attn, v)
        
        if self.bidirectional_causal_attn:
            mask_reverse = torch.triu(torch.ones([1, self.temporal_length, self.temporal_length], device=sim.device))
            sim_reverse = sim.float().masked_fill(mask_reverse == 0, max_neg_value)
            attn_reverse = sim_reverse.softmax(dim=-1)
            out_reverse = einsum('b i j, b j d -> b i d', attn_reverse, v)
            out += out_reverse
        
        if self.use_relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', attn, v2) # TODO check
            out += out2 # TODO check：先add还是先merge head？先计算rpr，on split head之后的数据，然后再merge。
        out = rearrange(out, '(b h) n d -> b n (h d)', h=nh) # merge head
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 sa_shared_kv=False, shared_type='only_first', **kwargs,):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.sa_shared_kv = sa_shared_kv
        assert(shared_type in ['only_first', 'all_frames', 'first_and_prev', 'only_prev', 'full', 'causal', 'full_qkv'])
        self.shared_type = shared_type

        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        if XFORMERS_IS_AVAILBLE:
            self.forward = self.efficient_forward

    def forward(self, x, context=None, mask=None):
        h = self.heads
        b = x.shape[0]
        
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        if self.sa_shared_kv:
            if self.shared_type == 'only_first':
                k,v = map(lambda xx: rearrange(xx[0].unsqueeze(0), 'b n c -> (b n) c').unsqueeze(0).repeat(b,1,1),
                          (k,v))
            else:
                raise NotImplementedError

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    
    def efficient_forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class VideoSpatialCrossAttention(CrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0):
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)
    def forward(self, x, context=None, mask=None):
        b, c, t, h, w = x.shape
        if context is not None:
            context = context.repeat(t, 1, 1)
        x = super.forward(spatial_attn_reshape(x), context=context) + x
        return spatial_attn_reshape_back(x,b,h)

class BasicTransformerBlockST(nn.Module):
    def __init__(self, 
        # Spatial Stuff
        dim, 
        n_heads, 
        d_head, 
        dropout=0., 
        context_dim=None, 
        gated_ff=True, 
        checkpoint=True,
        # Temporal Stuff
        temporal_length=None,   
        image_length=None,
        use_relative_position=True,
        img_video_joint_train=False,
        cross_attn_on_tempoal=False,
        temporal_crossattn_type="selfattn",
        order="stst",
        temporalcrossfirst=False,
        temporal_context_dim=None,
        split_stcontext=False,
        local_spatial_temporal_attn=False,
        window_size=2,
        **kwargs,
    ):
        super().__init__()
        # Self attention
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, **kwargs,)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        # cross attention if context is not None
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, **kwargs,)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.order = order
        assert(self.order in ["stst", "sstt", "st_parallel"])
        self.temporalcrossfirst = temporalcrossfirst
        self.split_stcontext = split_stcontext
        self.local_spatial_temporal_attn = local_spatial_temporal_attn
        if self.local_spatial_temporal_attn:
            assert(self.order == 'stst')
            assert(self.order == 'stst')
            self.window_size = window_size
        if not split_stcontext:
            temporal_context_dim = context_dim
        # Temporal attention
        assert(temporal_crossattn_type in ["selfattn", "crossattn", "skip"])
        self.temporal_crossattn_type = temporal_crossattn_type
        self.attn1_tmp = TemporalCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                                temporal_length=temporal_length,
                                                image_length=image_length,
                                                use_relative_position=use_relative_position,
                                                img_video_joint_train=img_video_joint_train,
                                                **kwargs,
        )
        self.attn2_tmp = TemporalCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                                # cross attn
                                                context_dim=temporal_context_dim if temporal_crossattn_type == "crossattn" else None,
                                                # temporal attn
                                                temporal_length=temporal_length,
                                                image_length=image_length,
                                                use_relative_position=use_relative_position,
                                                img_video_joint_train=img_video_joint_train,
                                                **kwargs,
        )
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)
        # self.norm1_tmp = nn.LayerNorm(dim)
        # self.norm2_tmp = nn.LayerNorm(dim)
        
    ##############################################################################################################################################
    def forward(self, x, context=None, temporal_context=None, no_temporal_attn=None, attn_mask=None, **kwargs):
        # print(f'no_temporal_attn={no_temporal_attn}')
        
        if not self.split_stcontext:
            # st cross attention use the same context vector
            temporal_context = context.detach().clone()
        
        if context is None and temporal_context is None:
            # self-attention models
            if no_temporal_attn:
                raise NotImplementedError
            return checkpoint(self._forward_nocontext, (x), self.parameters(), self.checkpoint)
        else:
            # cross-attention models
            if no_temporal_attn:
                forward_func = self._forward_no_temporal_attn
            else:
                forward_func = self._forward
            inputs = (x, context, temporal_context) if temporal_context is not None else (x, context)
            return checkpoint(forward_func, inputs, self.parameters(), self.checkpoint)
            # if attn_mask is not None:
            #     return checkpoint(self._forward, (x, context, temporal_context, attn_mask), self.parameters(), self.checkpoint)
            # return checkpoint(self._forward, (x, context, temporal_context), self.parameters(), self.checkpoint)
            
    def _forward(self, x, context=None, temporal_context=None, mask=None, no_temporal_attn=None, ):
        assert(x.dim() == 5), f"x shape = {x.shape}"
        b, c, t, h, w = x.shape
        
        if self.order in ["stst", "sstt"]:
            x = self._st_cross_attn(x, context, temporal_context=temporal_context, order=self.order, mask=mask,)#no_temporal_attn=no_temporal_attn,
        elif self.order == "st_parallel":
            x = self._st_cross_attn_parallel(x, context, temporal_context=temporal_context, order=self.order,)#no_temporal_attn=no_temporal_attn,
        else:
            raise NotImplementedError

        x = self.ff(self.norm3(x)) + x
        if (no_temporal_attn is None) or (not no_temporal_attn):
            x = rearrange(x, '(b h w) t c -> b c t h w', b=b,h=h,w=w) # 3d -> 5d
        elif no_temporal_attn:
            x = rearrange(x, '(b t) (h w) c -> b c t h w', b=b,h=h,w=w) # 3d -> 5d
        return x
    
    def _forward_no_temporal_attn(self, x, context=None, temporal_context=None, ):
        # temporary implementation :(
        # because checkpoint does not support non-tensor inputs currently.
        assert(x.dim() == 5), f"x shape = {x.shape}"
        b, c, t, h, w = x.shape
        
        if self.order in ["stst", "sstt"]:
            # x = self._st_cross_attn(x, context, temporal_context=temporal_context, order=self.order, no_temporal_attn=True,)
            # mask = torch.zeros([1, t, t], device=x.device).bool() if context is None else torch.zeros([1, context.shape[1], t], device=x.device).bool()
            mask = torch.zeros([1, t, t], device=x.device).bool()
            x = self._st_cross_attn(x, context, temporal_context=temporal_context, order=self.order, mask=mask,)
        elif self.order == "st_parallel":
            x = self._st_cross_attn_parallel(x, context, temporal_context=temporal_context, order=self.order, no_temporal_attn=True,)
        else:
            raise NotImplementedError

        x = self.ff(self.norm3(x)) + x
        x = rearrange(x, '(b h w) t c -> b c t h w', b=b,h=h,w=w) # 3d -> 5d
        # x = rearrange(x, '(b t) (h w) c -> b c t h w', b=b,h=h,w=w) # 3d -> 5d
        return x
    
    def _forward_nocontext(self, x, no_temporal_attn=None):
        assert(x.dim() == 5), f"x shape = {x.shape}"
        b, c, t, h, w = x.shape
        
        if self.order in ["stst", "sstt"]:
            x = self._st_cross_attn(x, order=self.order, no_temporal_attn=no_temporal_attn)
        elif self.order == "st_parallel":
            x = self._st_cross_attn_parallel(x, order=self.order, no_temporal_attn=no_temporal_attn)
        else:
            raise NotImplementedError
        
        x = self.ff(self.norm3(x)) + x
        x = rearrange(x, '(b h w) t c -> b c t h w', b=b,h=h,w=w) # 3d -> 5d

        return x
    ##############################################################################################################################################
    
    def _st_cross_attn(self, x, context=None, temporal_context=None, order="stst", mask=None): #no_temporal_attn=None, 
        b, c, t, h, w = x.shape
        # print(f'[_st_cross_attn input] x={x.shape}, context={context.shape}')
        
        if order == "stst":
            # spatial self attention
            x = rearrange(x, 'b c t h w -> (b t) (h w) c')
            x = self.attn1(self.norm1(x)) + x
            x = rearrange(x, '(b t) (h w) c -> b c t h w', b=b,h=h)
            
            # temporal self attention
            # if (no_temporal_attn is None) or (not no_temporal_attn):
            if self.local_spatial_temporal_attn:
                x = local_spatial_temporal_attn_reshape(x,window_size=self.window_size)
            else:
                x = rearrange(x, 'b c t h w -> (b h w) t c')
            x = self.attn1_tmp(self.norm4(x), mask=mask) + x
            
            if self.local_spatial_temporal_attn:
                x = local_spatial_temporal_attn_reshape_back(x, window_size=self.window_size, 
                    b=b, h=h, w=w, t=t)
            else:
                x = rearrange(x, '(b h w) t c -> b c t h w', b=b,h=h,w=w) # 3d -> 5d
            
            # spatial cross attention
            x = rearrange(x, 'b c t h w -> (b t) (h w) c')
            # context_ = context.repeat(t, 1, 1) if context is not None else None
            # print(f'[before spatial cross] context={context.shape}')
            if context is not None:
                if context.shape[0] == t:    # img captions no_temporal_attn or 
                    context_ = context
                else:  
                    context_ = []
                    for i in range(context.shape[0]):
                        context_.append(context[i].unsqueeze(0).repeat(t, 1, 1))
                    context_ = torch.cat(context_,dim=0)
            else:
                context_ = None
            # print(f'[before spatial cross] x={x.shape}, context_={context_.shape}')
            x = self.attn2(self.norm2(x), context=context_) + x

            # temporal cross attention
            # if (no_temporal_attn is None) or (not no_temporal_attn):
            x = rearrange(x, '(b t) (h w) c -> b c t h w', b=b,h=h)
            x = rearrange(x, 'b c t h w -> (b h w) t c')
            if self.temporal_crossattn_type == "crossattn":
                # tmporal cross attention
                if temporal_context is not None:
                    # print(f'STATTN context={context.shape}, temporal_context={temporal_context.shape}')
                    temporal_context = torch.cat([context, temporal_context], dim=1) # blc
                    # print(f'STATTN after concat temporal_context={temporal_context.shape}')
                    temporal_context = temporal_context.repeat(h*w, 1,1)
                    # print(f'after repeat temporal_context={temporal_context.shape}')
                else:
                    temporal_context = context[0:1,...].repeat(h*w, 1, 1)
                # print(f'STATTN after concat x={x.shape}')
                x = self.attn2_tmp(self.norm5(x), context=temporal_context, mask=mask) + x
            elif self.temporal_crossattn_type == "selfattn":
                # temporal self attention
                x = self.attn2_tmp(self.norm5(x), context=None, mask=mask) + x
            elif self.temporal_crossattn_type == "skip":
                # no temporal cross and self attention
                pass
            else:
                raise NotImplementedError

        elif order == "sstt":
            # spatial self attention
            x = rearrange(x, 'b c t h w -> (b t) (h w) c')
            x = self.attn1(self.norm1(x)) + x
            
            # spatial cross attention
            context_ = context.repeat(t, 1, 1) if context is not None else None
            x = self.attn2(self.norm2(x), context=context_) + x
            x = rearrange(x, '(b t) (h w) c -> b c t h w', b=b,h=h)
            
            if (no_temporal_attn is None) or (not no_temporal_attn):
                if self.temporalcrossfirst:
                    # temporal cross attention
                    if self.temporal_crossattn_type == "crossattn":
                        # if temporal_context is not None:
                        temporal_context = context.repeat(h*w, 1, 1)
                        x = self.attn2_tmp(self.norm5(x), context=temporal_context, mask=mask) + x
                    elif self.temporal_crossattn_type == "selfattn":
                        x = self.attn2_tmp(self.norm5(x), context=None, mask=mask) + x
                    elif self.temporal_crossattn_type == "skip":
                        pass
                    else:
                        raise NotImplementedError
                    # temporal self attention
                    x = rearrange(x, 'b c t h w -> (b h w) t c')
                    x = self.attn1_tmp(self.norm4(x), mask=mask) + x
                else:
                    # temporal self attention
                    x = rearrange(x, 'b c t h w -> (b h w) t c')
                    x = self.attn1_tmp(self.norm4(x), mask=mask) + x
                    # temporal cross attention
                    if self.temporal_crossattn_type == "crossattn":
                        if temporal_context is not None:
                            temporal_context = context.repeat(h*w, 1, 1)
                        x = self.attn2_tmp(self.norm5(x), context=temporal_context, mask=mask) + x
                    elif self.temporal_crossattn_type == "selfattn":
                        x = self.attn2_tmp(self.norm5(x), context=None, mask=mask) + x
                    elif self.temporal_crossattn_type == "skip":
                        pass
                    else:
                        raise NotImplementedError
        else:
            raise NotImplementedError

        return x
    
    def _st_cross_attn_parallel(self, x, context=None, temporal_context=None, order="sst", no_temporal_attn=None):
        """ order: x -> Self Attn -> Cross Attn -> attn_s
                   x -> Temp Self Attn -> attn_t
                   x' = x + attn_s + attn_t
        """
        if no_temporal_attn is not None:
            raise NotImplementedError

        B, C, T, H, W = x.shape
        # spatial self attention
        h = x
        h = rearrange(h, 'b c t h w -> (b t) (h w) c')
        h = self.attn1(self.norm1(h)) + h
        # spatial cross
        # context_ = context.repeat(T, 1, 1) if context is not None else None
        if context is not None:
            context_ = []
            for i in range(context.shape[0]):
                context_.append(context[i].unsqueeze(0).repeat(T, 1, 1))
            context_ = torch.cat(context_,dim=0)
        else:
            context_ = None

        h = self.attn2(self.norm2(h), context=context_) + h
        h = rearrange(h, '(b t) (h w) c -> b c t h w', b=B, h=H)
        
        # temporal self
        h2 = x
        h2 = rearrange(h2, 'b c t h w -> (b h w) t c')
        h2 = self.attn1_tmp(self.norm4(h2))# + h2
        h2 = rearrange(h2, '(b h w) t c -> b c t h w', b=B, h=H, w=W)
        out = h + h2
        return rearrange(out, 'b c t h w -> (b h w) t c')

    ##############################################################################################################################################

def spatial_attn_reshape(x):
    return rearrange(x, 'b c t h w -> (b t) (h w) c')
def spatial_attn_reshape_back(x,b,h):
    return rearrange(x, '(b t) (h w) c -> b c t h w', b=b,h=h)
def temporal_attn_reshape(x):
    return rearrange(x, 'b c t h w -> (b h w) t c')
def temporal_attn_reshape_back(x, b,h,w):
    return rearrange(x, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
def local_spatial_temporal_attn_reshape(x, window_size):
    B, C, T, H, W = x.shape
    NH = H // window_size
    NW = W // window_size
    # x = x.view(B, C, T, NH, window_size, NW, window_size)
    # tokens = x.permute(0, 1, 2, 3, 5, 4, 6).contiguous() 
    # tokens = tokens.view(-1, window_size, window_size, C) 
    x = rearrange(x, 'b c t (nh wh) (nw ww) -> b c t nh wh nw ww', nh=NH, nw=NW, wh=window_size, ww=window_size).contiguous() # # B, C, T, NH, NW, window_size, window_size
    x = rearrange(x, 'b c t nh wh nw ww -> (b nh nw) (t wh ww) c') # (B, NH, NW) (T, window_size, window_size) C 
    return x
def local_spatial_temporal_attn_reshape_back(x, window_size, b, h, w, t):
    B, L, C = x.shape
    NH = h // window_size
    NW = w // window_size
    x = rearrange(x, '(b nh nw) (t wh ww) c -> b c t nh wh nw ww', b=b, nh=NH, nw=NW, t=t, wh=window_size, ww=window_size)
    x = rearrange(x, 'b c t nh wh nw ww -> b c t (nh wh) (nw ww)')
    return x


class SpatialTemporalTransformer(nn.Module):
    """
    Transformer block for video-like data (5D tensor).
    First, project the input (aka embedding) with NO reshape.
    Then apply standard transformer action.
    The 5D -> 3D reshape operation will be done in the specific attention module.
    """
    def __init__(
        self,
        in_channels, n_heads, d_head,
        depth=1, dropout=0.,
        context_dim=None,
        # Temporal stuff
        temporal_length=None,
        image_length=None,
        use_relative_position=True,
        img_video_joint_train=False,
        cross_attn_on_tempoal=False,
        temporal_crossattn_type=False,
        order="stst",
        temporalcrossfirst=False,
        split_stcontext=False,
        temporal_context_dim=None,
        **kwargs,
        ):
        super().__init__()

        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv3d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlockST(
                inner_dim, n_heads, d_head, dropout=dropout,
                # cross attn
                context_dim=context_dim,
                # temporal attn
                temporal_length=temporal_length,   
                image_length=image_length,
                use_relative_position=use_relative_position,
                img_video_joint_train=img_video_joint_train,
                temporal_crossattn_type=temporal_crossattn_type,
                order=order,
                temporalcrossfirst=temporalcrossfirst,
                split_stcontext=split_stcontext,
                temporal_context_dim=temporal_context_dim,
                **kwargs
                ) for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv3d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        
    def forward(self, x, context=None, temporal_context=None, **kwargs):
        # note: if no context is given, cross-attention defaults to self-attention
        assert(x.dim() == 5), f"x shape = {x.shape}"
        b, c, t, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        for block in self.transformer_blocks:
            x = block(x, context=context, temporal_context=temporal_context, **kwargs)
        
        x = self.proj_out(x)
        return x + x_in

# ---------------------------------------------------------------------------------------------------

class STAttentionBlock2(nn.Module):
    def __init__(
        self, 
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,           # not used, only used in ResBlock
        use_new_attention_order=False,  # QKVAttention or QKVAttentionLegacy
        temporal_length=16,             # used in relative positional representation.
        image_length=8,                 # used for image-video joint training.
        use_relative_position=False,     # whether use relative positional representation in temporal attention.
        img_video_joint_train=False,
        # norm_type="groupnorm",
        attn_norm_type="group",
        use_tempoal_causal_attn=False,
    ):
        """ 
        version 1: guided_diffusion implemented version 
        version 2: remove args input argument 
        """
        super().__init__()

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint

        self.temporal_length = temporal_length
        self.image_length = image_length
        self.use_relative_position = use_relative_position
        self.img_video_joint_train = img_video_joint_train
        self.attn_norm_type = attn_norm_type
        assert(self.attn_norm_type in ["group", "no_norm"])
        self.use_tempoal_causal_attn = use_tempoal_causal_attn

        if self.attn_norm_type == "group":
            self.norm_s = normalization(channels)
            self.norm_t = normalization(channels)

        self.qkv_s = conv_nd(1, channels, channels * 3, 1)
        self.qkv_t = conv_nd(1, channels, channels * 3, 1)
        
        if self.img_video_joint_train:
            mask = th.ones([1, temporal_length+image_length, temporal_length+image_length])
            mask[:, temporal_length:, :] = 0
            mask[:, :, temporal_length:] = 0
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        if use_new_attention_order:
            # split qkv before split heads
            self.attention_s = QKVAttention(self.num_heads)
            self.attention_t = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention_s = QKVAttentionLegacy(self.num_heads)
            self.attention_t = QKVAttentionLegacy(self.num_heads)
        
        if use_relative_position:
            self.relative_position_k = RelativePosition(num_units=channels // self.num_heads, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=channels // self.num_heads, max_relative_position=temporal_length)

        self.proj_out_s = zero_module(conv_nd(1, channels, channels, 1)) # conv_dim, in_channels, out_channels, kernel_size
        self.proj_out_t = zero_module(conv_nd(1, channels, channels, 1)) # conv_dim, in_channels, out_channels, kernel_size

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape
        
        # spatial
        out = rearrange(x, 'b c t h w -> (b t) c (h w)')
        if self.attn_norm_type == "no_norm":
            qkv = self.qkv_s(out)
        else:
            qkv = self.qkv_s(self.norm_s(out))
        out = self.attention_s(qkv)
        out = self.proj_out_s(out)
        out = rearrange(out, '(b t) c (h w) -> b c t h w', b=b,h=h)
        x += out

        # temporal
        out = rearrange(x, 'b c t h w -> (b h w) c t')
        if self.attn_norm_type == "no_norm":
            qkv = self.qkv_t(out)
        else:
            qkv = self.qkv_t(self.norm_t(out))
        
        # relative positional embedding
        if self.use_relative_position:
            len_q = qkv.size()[-1]
            len_k, len_v = len_q, len_q
            k_rp = self.relative_position_k(len_q, len_k)
            v_rp = self.relative_position_v(len_q, len_v) #[T,T,head_dim]
            out = self.attention_t(qkv, rp=(k_rp, v_rp), mask=self.mask, use_tempoal_causal_attn=self.use_tempoal_causal_attn)
        else:
            out = self.attention_t(qkv, rp=None, mask=self.mask, use_tempoal_causal_attn=self.use_tempoal_causal_attn)

        out = self.proj_out_t(out)
        out = rearrange(out, '(b h w) c t -> b c t h w', b=b,h=h,w=w)
        
        return (x + out)

# ---------------------------------------------------------------------------------------------------------------

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, rp=None, mask=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        if rp is not None or mask is not None:
            raise NotImplementedError
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

# ---------------------------------------------------------------------------------------------------------------

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, rp=None, mask=None, use_tempoal_causal_attn=False):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        # print('qkv', qkv.size())
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        # print('bs, self.n_heads, ch, length', bs, self.n_heads, ch, length)

        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        # weight:[b,t,s] b=bs*n_heads*T

        if rp is not None:
            k_rp, v_rp = rp # [length, length, head_dim] [8, 8, 48]
            weight2 = th.einsum(
                'bct,tsc->bst',
                (q * scale).view(bs * self.n_heads, ch, length),
                k_rp
            )
            weight += weight2
        
        if use_tempoal_causal_attn:
            # weight = torch.tril(weight)
            assert(mask is None), f'Not implemented for merging two masks!'
            mask = torch.tril(torch.ones(weight.shape))
        else:
            if mask is not None: # only keep upper-left matrix
                # process mask
                c, t, _ = weight.shape
                
                if mask.shape[-1] > t:
                    mask = mask[:, :t, :t]
                elif mask.shape[-1] < t: # pad ones
                    mask_ = th.zeros([c,t,t]).to(mask.device)
                    t_ = mask.shape[-1]
                    mask_[:, :t_, :t_] = mask
                    mask = mask_
                else:
                    assert(weight.shape[-1] == mask.shape[-1]), f'weight={weight.shape}, mask={mask.shape}'
        
        if mask is not None:
            INF = -1e8 #float('-inf')
            weight = weight.float().masked_fill(mask == 0, INF)

        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype) #[256, 8, 8] [b, t, t] b=bs*n_heads*h*w,t=nframes
        # weight = F.softmax(weight, dim=-1)#[256, 8, 8] [b, t, t] b=bs*n_heads*h*w,t=nframes
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)) #[256, 48, 8] [b, head_dim, t]
        
        if rp is not None:
            a2 = th.einsum(
                "bts,tsc->btc",
                weight,
                v_rp
            ).transpose(1,2) # btc->bct
            a += a2
        
        return a.reshape(bs, -1, length)

# ---------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------
