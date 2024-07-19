from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from diffusers.utils import BaseOutput, logging


@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# cmcm
def Adapted_TemporalTransformerBlock_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, **kwargs):
        # import pdb; pdb.set_trace()
        for attention_block, norm in zip(self.attention_blocks[:-1], self.norms[:-1]):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
            ) + hidden_states
        
        if 'RT' in kwargs and kwargs['RT'] is not None:
            RT = kwargs['RT']
            # hidden_states.shape = [batch * video_length, height * width, channel]
            # RT.shape = [batch, video_length, 12]
            B, t, _ = RT.shape
            RT = RT.reshape(B*t, 1, -1)
            _, hw, _ = hidden_states.shape
            RT = RT.repeat(1, hw, 1)
            hidden_states = torch.cat([hidden_states, RT], dim=-1)
            hidden_states = self.cc_projection(hidden_states)
        
        norm_hidden_states = self.norms[-1](hidden_states)
        hidden_states = self.attention_blocks[-1](
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.attention_blocks[-1].is_cross_attention else None,
                video_length=video_length,
        ) + hidden_states

            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output

# omcm
def Adapted_CrossAttnDownBlock3D_forward(self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, **kwargs):
        output_states = ()

        if 'traj_features' in kwargs:
            traj_features = kwargs['traj_features']
            kwargs.pop('traj_features')
        else:
            traj_features = None

        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(motion_module), hidden_states.requires_grad_(), temb, encoder_hidden_states)
                
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
                
                # add motion module
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states, **kwargs) if motion_module is not None else hidden_states

            output_states += (hidden_states,)

        # omcm
        if traj_features is not None:
            hidden_states = hidden_states + traj_features[self.traj_fea_idx]
            output_states = output_states[:-1] + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

def Adapted_DownBlock3D_forward(self, hidden_states, temb=None, encoder_hidden_states=None, **kwargs):
        output_states = ()

        if 'traj_features' in kwargs:
            traj_features = kwargs['traj_features']
            kwargs.pop('traj_features')
        else:
            traj_features = None

        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(motion_module, **kwargs), hidden_states.requires_grad_(), temb, encoder_hidden_states)
            else:
                hidden_states = resnet(hidden_states, temb)

                # add motion module
                hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states, **kwargs) if motion_module is not None else hidden_states

            output_states += (hidden_states,)

        # omcm
        if traj_features is not None:
            hidden_states = hidden_states + traj_features[self.traj_fea_idx]
            output_states = output_states[:-1] + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


# unet forward
def unet3d_forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # RT: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # cmcm
        if 'RT' in kwargs:
            RT = kwargs['RT']
            kwargs.pop('RT')
        else:
            RT = None

        if 'traj_features' in kwargs:
            traj_features = kwargs['traj_features']
            kwargs.pop('traj_features')
        else:
            traj_features = None

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    RT=RT,
                    traj_features=traj_features,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states, RT=RT)

            down_block_res_samples += res_samples

        # mid
        sample = self.mid_block(
            sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, RT=RT
        )

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    RT=RT
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, 
                    temb=emb, 
                    res_hidden_states_tuple=res_samples, 
                    upsample_size=upsample_size, 
                    encoder_hidden_states=encoder_hidden_states, 
                    RT=RT
                )

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)