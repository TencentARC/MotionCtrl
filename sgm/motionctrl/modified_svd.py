from functools import partial
from typing import List, Optional, Union

import torch
from einops import rearrange, repeat

from sgm.modules.attention import checkpoint, exists
from sgm.modules.diffusionmodules.util import timestep_embedding


### VideoUnet #####
def forward_VideoUnet(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        num_video_frames: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        RT: Optional[torch.Tensor] = None
    ):
        if RT is not None:
             context = {'RT': RT, 'context': context}

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        ## tbd: check the role of "image_only_indicator"
        num_video_frames = self.num_frames
        image_only_indicator = torch.zeros(
                    x.shape[0]//num_video_frames, num_video_frames
                ).to(x.device) if image_only_indicator is None else image_only_indicator

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames
            )
            hs.append(h)
        h = self.middle_block(
            h,
            emb,
            context=context,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_video_frames=num_video_frames
        )
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(
                h,
                emb,
                context=context,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_video_frames=num_video_frames
            )
        h = h.type(x.dtype)
        return self.out(h)

### VideoTransformerBlock #####

def forward_VideoTransformerBlock(self, x, context, timesteps):
    if self.checkpoint:
        return checkpoint(self._forward, x, context, timesteps)
    else:
        return self._forward(x, context, timesteps=timesteps)


def _forward_VideoTransformerBlock_attan2(self, x, context=None, timesteps=None):
    assert self.timesteps or timesteps
    assert not (self.timesteps and timesteps) or self.timesteps == timesteps
    timesteps = self.timesteps or timesteps
    B, S, C = x.shape
    x = rearrange(x, "(b t) s c -> (b s) t c", t=timesteps)

    if isinstance(context, dict):
        RT = context['RT'] # (b, t, 12)
        context = context['context']
    else:
        RT = None

    if self.ff_in:
        x_skip = x
        x = self.ff_in(self.norm_in(x))
        if self.is_res:
            x += x_skip

    if self.disable_self_attn:
        x = self.attn1(self.norm1(x), context=context) + x
    else:
        x = self.attn1(self.norm1(x)) + x

    if RT is not None:
        # import pdb; pdb.set_trace()
        RT = RT.repeat_interleave(repeats=S, dim=0) # (b*s, t, 12)
        x = torch.cat([x, RT], dim=-1)
            
        x = self.cc_projection(x)

    if self.attn2 is not None:
        if self.switch_temporal_ca_to_sa:
            x = self.attn2(self.norm2(x)) + x
        else:
            x = self.attn2(self.norm2(x), context=context) + x
    x_skip = x
    x = self.ff(self.norm3(x))
    if self.is_res:
        x += x_skip

    x = rearrange(
        x, "(b s) t c -> (b t) s c", s=S, b=B // timesteps, c=C, t=timesteps
    )
    return x


#### BasicTransformerBlock #####
def _forward_BasicTransformerBlock(
    self, x, context=None, additional_tokens=None, n_times_crossframe_attn_in_self=0
):
    if isinstance(context, dict):
        context = context['context']
    x = (
        self.attn1(
            self.norm1(x),
            context=context if self.disable_self_attn else None,
            additional_tokens=additional_tokens,
            n_times_crossframe_attn_in_self=n_times_crossframe_attn_in_self
            if not self.disable_self_attn
            else 0,
        )
        + x
    )
    x = (
        self.attn2(
            self.norm2(x), context=context, additional_tokens=additional_tokens
        )
        + x
    )
    x = self.ff(self.norm3(x)) + x
    return x
    

#### SpatialVideoTransformer #####
def forward_SpatialVideoTransformer(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_context: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, h, w = x.shape
        x_in = x

        if isinstance(context, dict):
            RT = context['RT']
            context = context['context']
        else:
            RT = None

        spatial_context = None
        if exists(context):
            spatial_context = context

        if self.use_spatial_context:
            assert (
                context.ndim == 3
            ), f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = repeat(
                time_context_first_timestep, "b ... -> (b n) ...", n=h * w
            )
        elif time_context is not None and not self.use_spatial_context:
            time_context = repeat(time_context, "b ... -> (b n) ...", n=h * w)
            if time_context.ndim == 2:
                time_context = rearrange(time_context, "b c -> b 1 c")

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = torch.arange(timesteps, device=x.device)
        num_frames = repeat(num_frames, "t -> b t", b=x.shape[0] // timesteps)
        num_frames = rearrange(num_frames, "b t -> (b t)")
        t_emb = timestep_embedding(
            num_frames,
            self.in_channels,
            repeat_only=False,
            max_period=self.max_time_embed_period,
        )
        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(
            zip(self.transformer_blocks, self.time_stack)
        ):
            x = block(
                x,
                context=spatial_context,
            )

            x_mix = x
            x_mix = x_mix + emb

            if RT is not None:
                x_mix = mix_block(x_mix, context={'context': time_context, 'RT': RT}, timesteps=timesteps)
            else:
                x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
                image_only_indicator=image_only_indicator,
            )
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out