
import torch.nn as nn

from sgm.models.diffusion import DiffusionEngine
from sgm.motionctrl.modified_svd import (
                                         _forward_VideoTransformerBlock_attan2,
                                         forward_SpatialVideoTransformer,
                                         forward_VideoTransformerBlock,
                                         forward_VideoUnet)

class CameraMotionControl(DiffusionEngine):
    def __init__(self,
                 pose_embedding_dim = 1,
                 pose_dim = 12,
                 *args, **kwargs):
        
        if 'ckpt_path' in kwargs:
            ckpt_path = kwargs.pop('ckpt_path')
        else:
            ckpt_path = None

        self.use_checkpoint = kwargs['network_config']['params']['use_checkpoint']

        super().__init__(*args, **kwargs)

        bound_method = forward_VideoUnet.__get__(
                self.model.diffusion_model, 
                self.model.diffusion_model.__class__)
        setattr(self.model.diffusion_model, 'forward', bound_method)

        self.train_module_names = []
        for _name, _module in self.model.diffusion_model.named_modules():
            if _module.__class__.__name__ == 'VideoTransformerBlock':
                bound_method = forward_VideoTransformerBlock.__get__(
                    _module, _module.__class__)
                setattr(_module, 'forward', bound_method)

                
                bound_method = _forward_VideoTransformerBlock_attan2.__get__(
                    _module, _module.__class__)
                setattr(_module, '_forward', bound_method)
                
                cc_projection = nn.Linear(_module.attn2.to_q.in_features + pose_embedding_dim*pose_dim, _module.attn2.to_q.in_features) # 1024
                nn.init.eye_(list(cc_projection.parameters())[0][:_module.attn2.to_q.in_features, :_module.attn2.to_q.in_features])
                nn.init.zeros_(list(cc_projection.parameters())[1])
            
                cc_projection.requires_grad_(True)

                _module.add_module('cc_projection', cc_projection)

                self.train_module_names.append(f'{_name}.cc_projection')
                
                self.train_module_names.append(f'{_name}.attn2')
                self.train_module_names.append(f'{_name}.norm2')


            if _module.__class__.__name__ == 'SpatialVideoTransformer':
                bound_method = forward_SpatialVideoTransformer.__get__(
                    _module, _module.__class__)
                setattr(_module, 'forward', bound_method)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)