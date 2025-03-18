# -*- coding: utf-8 -*-

import torch.nn as nn
from typing import Tuple, List, Optional

# Base class for output of Point to Mesh transformation
class Point2MeshOutput(object):
    def __init__(self):
        self.mesh_v = None  # Vertices of the mesh
        self.mesh_f = None  # Faces of the mesh
        self.center = None  # Center of the mesh
        self.pc = None  # Point cloud data


# Base class for output of Latent to Mesh transformation
class Latent2MeshOutput(object):
    def __init__(self):
        self.mesh_v = None  # Vertices of the mesh
        self.mesh_f = None  # Faces of the mesh


# Base class for output of Aligned Mesh transformation
class AlignedMeshOutput(object):
    def __init__(self):
        self.mesh_v = None  # Vertices of the mesh
        self.mesh_f = None  # Faces of the mesh
        self.surface = None  # Surface data
        self.image = None  # Aligned image data
        self.text: Optional[str] = None  # Aligned text data
        self.shape_text_similarity: Optional[float] = None  # Similarity between shape and text
        self.shape_image_similarity: Optional[float] = None  # Similarity between shape and image


# Base class for Shape as Latent with Point to Mesh transformation module
class ShapeAsLatentPLModule(nn.Module):
    latent_shape: Tuple[int]  # Shape of the latent space
    
    def encode(self, surface, *args, **kwargs):
        raise NotImplementedError

    def decode(self, z_q, *args, **kwargs):
        raise NotImplementedError

    def latent2mesh(self, latents, *args, **kwargs) -> List[Latent2MeshOutput]:
        raise NotImplementedError

    def point2mesh(self, *args, **kwargs) -> List[Point2MeshOutput]:
        raise NotImplementedError


# Base class for Shape as Latent module
class ShapeAsLatentModule(nn.Module):
    latent_shape: Tuple[int, int]  # Shape of the latent space

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError


# Base class for Aligned Shape as Latent with Point to Mesh transformation module
class AlignedShapeAsLatentPLModule(nn.Module):
    latent_shape: Tuple[int]  # Shape of the latent space

    def set_shape_model_only(self):
        raise NotImplementedError

    def encode(self, surface, *args, **kwargs):
        raise NotImplementedError

    def decode(self, z_q, *args, **kwargs):
        raise NotImplementedError

    def latent2mesh(self, latents, *args, **kwargs) -> List[Latent2MeshOutput]:
        raise NotImplementedError

    def point2mesh(self, *args, **kwargs) -> List[Point2MeshOutput]:
        raise NotImplementedError


# Base class for Aligned Shape as Latent module
class AlignedShapeAsLatentModule(nn.Module):
    shape_model: ShapeAsLatentModule  # Shape model module
    latent_shape: Tuple[int, int]  # Shape of the latent space


    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_shape_model_only(self):
        raise NotImplementedError

    def encode_image_embed(self, *args, **kwargs):
        raise NotImplementedError

    def encode_text_embed(self, *args, **kwargs):
        raise NotImplementedError

    def encode_shape_embed(self, *args, **kwargs):
        raise NotImplementedError

# Base class for Textured Shape as Latent module
class TexturedShapeAsLatentModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError

    def query_color(self, *args, **kwargs):
        raise NotImplementedError
