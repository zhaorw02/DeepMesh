"""Mesh data utilities."""
import random
import networkx as nx
import numpy as np
# import pyrr
from six.moves import range
import trimesh
from scipy.spatial.transform import Rotation


def to_mesh(vertices, faces, transpose=True, post_process=False):
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
        
    if faces.min() == 1:
        faces = (np.array(faces) - 1).tolist()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if post_process:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    return mesh


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    # vert_center = np.mean(vertices, axis=0)
    return vertices - vert_center


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def block_index(vertex, block_size=32):
    return (vertex[2] // block_size, vertex[1] // block_size, vertex[0] // block_size)

def block_id(block_index, num_blocks=4):
    return block_index[0] * num_blocks**2 + block_index[1] * num_blocks + block_index[2]


def normalize_vertices_scale(vertices, scale=0.95):
    """Scale the vertices so that the long axis of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = (vert_max - vert_min).max()
    return 2.0 * scale * vertices / (extents + 1e-6)


def quantize_process_mesh(vertices, faces, quantization_bits=8, block_first_order=True, block_size=32, num_blocks=4):
    """Quantize vertices, remove resulting duplicates and reindex faces."""
    vertices = discretize(vertices, num_discrete=2**quantization_bits)
    vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

    if block_first_order:
        block_indices = np.array([block_index(v, block_size) for v in vertices])
        block_ids = np.array([block_id(b, num_blocks) for b in block_indices])
        sort_inds = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2], block_ids))
    else:
        # Sort vertices by z then y then x.
        sort_inds = np.lexsort(vertices.T)
    
    vertices = vertices[sort_inds]
    faces = [np.argsort(sort_inds)[inv[f]] for f in faces]

    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            if c_length > 2:
                d = np.argmin(f)
                sub_faces.append([f[(d + i) % c_length] for i in range(c_length)])

    faces = sub_faces

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))
    num_verts = vertices.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]
    ).any(axis=-1)
    vertices = vertices[vert_connected]

    # Re-index faces to re-ordered vertices.
    vert_indices = np.arange(num_verts) - np.cumsum(1 - vert_connected.astype("int"))
    faces = [vert_indices[f].tolist() for f in faces]

    return vertices, faces


def process_mesh(vertices, faces, quantization_bits=8, augment=True, augment_dict=None):
    """Process mesh vertices and faces."""

    # Transpose so that z-axis is vertical.
    vertices = vertices[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    if augment:
        vertices = augment_mesh(vertices, **augment_dict)

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    vertices = normalize_vertices_scale(vertices)

    # Quantize and sort vertices, remove resulting duplicates, sort and reindex
    # faces.
    vertices, faces = quantize_process_mesh(
        vertices, faces, quantization_bits=quantization_bits
    )
    vertices = undiscretize(vertices, num_discrete=2**quantization_bits)


    # Discard degenerate meshes without faces.
    return {
        "vertices": vertices,
        "faces": faces,
    }


def load_process_mesh(mesh_obj_path, quantization_bits=8, augment=False, augment_dict=None):
    """Load obj file and process."""
    # Load mesh
    mesh = trimesh.load(mesh_obj_path, force='mesh', process=False)
    return process_mesh(mesh.vertices, mesh.faces, quantization_bits, augment=augment, augment_dict=augment_dict)


def augment_mesh(vertices, scale_min=0.95, scale_max=1.05, rotation=0., jitter_strength=0.):
    '''scale vertices by a factor in [0.75, 1.25]'''
    
    # # vertices [nv, 3]
    # for i in range(3):
    #     # Generate a random scale factor
    #     scale = random.uniform(scale_min, scale_max)    
    #     vertices[:, i] *= scale
    
    # if rotation != 0.:
    #     axis = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    #     radian = np.pi / 180 * rotation
    #     rotation = Rotation.from_rotvec(radian * np.array(axis))
    #     vertices =rotation.apply(vertices)
    
    if rotation != 0:
        # angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330] 
        angles = [0, 90, 180, 270]
        axes = [
            [1, 0, 0], 
            # [0, 1, 0],  
            # [0, 0, 1]   
        ]

        rotation_angle = random.choice(angles)  # 从角度列表中选择一个角度
        rotation_axis = random.choice(axes)    # 从旋转轴列表中选择一个轴

        radian = np.pi / 180 * rotation_angle
        
        rotation = Rotation.from_rotvec(radian * np.array(rotation_axis))

        vertices = rotation.apply(vertices)
        
        # if jitter_strength != 0.:
        #     jitter_amount = np.random.uniform(-jitter_strength, jitter_strength)
        #     vertices += jitter_amount
    
    return vertices


def discretize(
    t,
    continuous_range = (-1, 1),
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo
    # print(t.max())
    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().astype(np.int32).clip(min = 0, max = num_discrete - 1)


def undiscretize(
    t,
    continuous_range = (-1, 1),
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo

    t = t.astype(np.float32)

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

