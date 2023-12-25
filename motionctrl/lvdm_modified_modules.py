import logging

import torch
from einops import rearrange, repeat

from lvdm.models.utils_diffusion import timestep_embedding

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

mainlogger = logging.getLogger('mainlogger')



def TemporalTransformer_forward(self, x, context=None, is_imgbatch=False):
    b, c, t, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
    if not self.use_linear:
        x = self.proj_in(x)
    x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
    if self.use_linear:
        x = self.proj_in(x)

    temp_mask = None
    if self.causal_attention:
        temp_mask = torch.tril(torch.ones([1, t, t]))
    if is_imgbatch:
        temp_mask = torch.eye(t).unsqueeze(0)
    if temp_mask is not None:
        mask = temp_mask.to(x.device)
        mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
    else:
        mask = None

    if self.only_self_att:
        ## note: if no context is given, cross-attention defaults to self-attention
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, mask=mask)
        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
    else:
        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
        for i, block in enumerate(self.transformer_blocks):
            # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
            for j in range(b):
                unit_context = context[j][0:1]
                context_j = repeat(unit_context, 't l con -> (t r) l con', r=(h * w)).contiguous()
                ## note: causal mask will not applied in cross-attention case
                x[j] = block(x[j], context=context_j)
    
    if self.use_linear:
        x = self.proj_out(x)
        x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
    if not self.use_linear:
        x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
        x = self.proj_out(x)
        x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

    if self.use_image_dataset:
        x = 0.0 * x + x_in
    else:
        x = x + x_in
    return x

def selfattn_forward_unet(self, x, timesteps, context=None, y=None, features_adapter=None, is_imgbatch=False, T=None,  **kwargs):
        b,_,t,_,_ = x.shape
    
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        if self.micro_condition and y is not None:
            micro_emb = timestep_embedding(y, self.model_channels, repeat_only=False)
            emb = emb + self.micro_embed(micro_emb)

        

        # pose_emb = pose_emb.reshape(-1, pose_emb.shape[-1])
        ## repeat t times for context [(b t) 77 768] & time embedding
        if not is_imgbatch:
            context = context.repeat_interleave(repeats=t, dim=0)

        if 'pose_emb' in kwargs:
            pose_emb = kwargs.pop('pose_emb')
            context = { 'context': context, 'pose_emb': pose_emb }

        emb = emb.repeat_interleave(repeats=t, dim=0)

        ## always in shape (b t) c h w, except for temporal layer
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        if features_adapter is not None:
            features_adapter = [rearrange(feature, 'b c t h w -> (b t) c h w') for feature in features_adapter]

        h = x.type(self.dtype)
        adapter_idx = 0
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b,is_imgbatch=is_imgbatch)
            if id ==0 and self.addition_attention:
                h = self.init_attn(h, emb, context=context, batch_size=b,is_imgbatch=is_imgbatch)
            ## plug-in adapter features
            if ((id+1)%3 == 0) and features_adapter is not None:
                # if adapter_idx == 0 or adapter_idx == 1 or adapter_idx == 2:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1
            hs.append(h)
        if features_adapter is not None:
            assert len(features_adapter)==adapter_idx, 'Wrong features_adapter'

        h = self.middle_block(h, emb, context=context, batch_size=b, is_imgbatch=is_imgbatch)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b, is_imgbatch=is_imgbatch)
        h = h.type(x.dtype)
        y = self.out(h)
        
        # reshape back to (b c t h w)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        return y

def spatial_forward_BasicTransformerBlock(self, x, context=None, mask=None):
    if isinstance(context, dict):
        context = context['context']
    x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
    x = self.attn2(self.norm2(x), context=context, mask=mask) + x
    x = self.ff(self.norm3(x)) + x
    return x

def temporal_selfattn_forward_BasicTransformerBlock(self, x, context=None, mask=None):
    if isinstance(context, dict) and 'pose_emb' in context:
        pose_emb = context['pose_emb'] # {channel_num: [B, video_length, pose_dim, pose_embedding_dim]}
        context = None
    else:
        pose_emb = None
        context = None

    x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x

    # Add camera pose
    if pose_emb is not None:
        B, t, _, _ = pose_emb.shape # [B, video_length, pose_dim, pose_embedding_dim]
        hw = x.shape[0] // B
        pose_emb = pose_emb.reshape(B, t, -1)
        pose_emb = pose_emb.repeat_interleave(repeats=hw, dim=0)
        x = self.cc_projection(torch.cat([x, pose_emb], dim=-1))

    x = self.attn2(self.norm2(x), context=context, mask=mask) + x
    x = self.ff(self.norm3(x)) + x
    return x
