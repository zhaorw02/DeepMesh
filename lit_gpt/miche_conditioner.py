import torch
from torch import nn
from beartype import beartype
from miche.encode import load_model

# helper functions

def exists(val):
    return val is not None

def default(*values):
    for value in values:
        if exists(value):
            return value
    return None


# point-cloud encoder from Michelangelo
@beartype
class PointConditioner(torch.nn.Module):
    def __init__(
        self,
        *,
        dim_latent = None,
        model_name = 'miche-256-feature',
        cond_dim = 768,
        freeze = True,
    ):
        super().__init__()

        # open-source version of miche
        if model_name == 'miche-256-feature':
            ckpt_path = None
            config_path = 'miche/shapevae-256.yaml'

            self.feature_dim = 1024    # embedding dimension
            self.cond_length = 257     # length of embedding
            self.point_encoder = load_model(ckpt_path=ckpt_path, config_path=config_path)
            print("loading miche/shapevae-256.yaml")
            # additional layers to connect miche and GPT
            self.cond_head_proj = nn.Linear(cond_dim, self.feature_dim)
            self.cond_proj = nn.Linear(cond_dim, self.feature_dim)
            
        else:
            raise NotImplementedError

        # whether to finetuen point-cloud encoder
        if freeze:
            for parameter in self.point_encoder.parameters():
                parameter.requires_grad = False

        self.freeze = freeze
        self.model_name = model_name
        self.dim_latent = default(dim_latent, self.feature_dim)
        
        self.register_buffer('_device_param', torch.tensor(0.), persistent = False)


    @property
    def device(self):
        return next(self.buffers()).device


    def embed_pc(self, pc_normal):
        # encode point cloud to embeddings
        if self.model_name == 'miche-256-feature':
            point_feature = self.point_encoder.encode_latents(pc_normal)
            pc_embed_head = self.cond_head_proj(point_feature[:, 0:1])
            pc_embed = self.cond_proj(point_feature[:, 1:])
            pc_embed = torch.cat([pc_embed_head, pc_embed], dim=1)

        return pc_embed


    def forward(
        self,
        pc = None,
        pc_embeds = None,
    ):
        if pc_embeds is None:
            pc_embeds = self.embed_pc(pc.to(next(self.buffers()).dtype))
            
        assert not torch.any(torch.isnan(pc_embeds)), 'NAN values in pc embedings'
        
        return pc_embeds
    
