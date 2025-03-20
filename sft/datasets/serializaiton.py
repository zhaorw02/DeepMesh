'''
 * Adapted from BPT (https://github.com/whaohan/bpt)
'''
import trimesh
import numpy as np
from .data_utils import discretize, undiscretize


def patchified_mesh(mesh: trimesh.Trimesh, special_token = -2, fix_orient=True):
    sequence = []
    unvisited = np.full(len(mesh.faces), True)
    degrees = mesh.vertex_degree.copy()

    # with fix_orient=True, the normal would be correct.
    # but this may increase the difficulty for learning.
    if fix_orient:
        face_orient = {}
        for ind, face in enumerate(mesh.faces):
            v0, v1, v2 = face[0], face[1], face[2]
            face_orient['{}-{}-{}'.format(v0, v1, v2)] = True
            face_orient['{}-{}-{}'.format(v1, v2, v0)] = True
            face_orient['{}-{}-{}'.format(v2, v0, v1)] = True
            face_orient['{}-{}-{}'.format(v2, v1, v0)] = False
            face_orient['{}-{}-{}'.format(v1, v0, v2)] = False
            face_orient['{}-{}-{}'.format(v0, v2, v1)] = False

    while sum(unvisited):
        unvisited_faces = mesh.faces[unvisited]
        
        # select the patch center
        cur_face = unvisited_faces[0]
        max_deg_vertex_id = np.argmax(degrees[cur_face])
        max_deg_vertex = cur_face[max_deg_vertex_id]
        
        # find all connected faces
        selected_faces = []
        for face_idx in mesh.vertex_faces[max_deg_vertex]:
            if face_idx != -1 and unvisited[face_idx]:
                face = mesh.faces[face_idx]
                u, v = sorted([vertex for vertex in face if vertex != max_deg_vertex])
                selected_faces.append([u, v, face_idx])
                
        face_patch = set()
        selected_faces = sorted(selected_faces)
        
        # select the start vertex, select it if it only appears once (the start or end), 
        # else select the lowest index
        cnt = {}
        for u, v, _ in selected_faces:
            cnt[u] = cnt.get(u, 0) + 1
            cnt[v] = cnt.get(v, 0) + 1
        starts = []
        for vertex, num in cnt.items():
            if num == 1:
                starts.append(vertex)
        start_idx = min(starts) if len(starts) else selected_faces[0][0]
        
        res = [start_idx]
        while len(res) <= len(selected_faces):
            vertex = res[-1]
            for u_i, v_i, face_idx_i in selected_faces:
                if face_idx_i not in face_patch and vertex in (u_i, v_i):
                    u_i, v_i = (u_i, v_i) if vertex == u_i else (v_i, u_i)
                    res.append(v_i)
                    face_patch.add(face_idx_i)
                    break
            
            if res[-1] == vertex:
                break
            
        if fix_orient and len(res) >= 2 and not face_orient['{}-{}-{}'.format(max_deg_vertex, res[0], res[1])]:
            res = res[::-1]
        
        # reduce the degree of related vertices and mark the visited faces
        degrees[max_deg_vertex] = len(selected_faces) - len(res) + 1
        for pos_idx, vertex in enumerate(res):
            if pos_idx in [0, len(res) - 1]:
                degrees[vertex] -= 1
            else:
                degrees[vertex] -= 2
        for face_idx in face_patch:
            unvisited[face_idx] = False 
        sequence.extend(
            [mesh.vertices[max_deg_vertex]] + 
            [mesh.vertices[vertex_idx] for vertex_idx in res] + 
            [[special_token] * 3]
        )
        
    assert sum(degrees) == 0, 'All degrees should be zero'

    return np.array(sequence)
    
def get_block_representation(
        sequence, 
        patch_size=4,
        block_size=8,
        offset_size=16, 
        block_compressed=True, 
        special_token=-2, 
        use_special_block=True
    ):
    '''
    convert coordinates from Cartesian system to block indexes.
    '''
    special_block_base = block_size**3 + offset_size**3 + patch_size**3
    # prepare coordinates
    sp_mask = sequence != special_token
    sp_mask = np.all(sp_mask, axis=1) #(492,)
    coords = sequence[sp_mask].reshape(-1, 3) # continuous array
    coords = discretize(coords, num_discrete=512) # array of int32

    # convert [x, y, z] to [patch_id, block_id, offset_id]
    patch_id = coords // (block_size * offset_size)  # Determine the patch location
    coords_within_patch = coords % (block_size * offset_size)  # Coordinates within the patch
    
    block_id = coords_within_patch // offset_size  # Determine block location within the patch
    offset_id = coords_within_patch % offset_size  # Determine offset within the block
    
    # flatten to 1D
    patch_id = patch_id[:, 0] * patch_size**2 + patch_id[:, 1] * patch_size + patch_id[:, 2]
    block_id = block_id[:, 0] * block_size**2 + block_id[:, 1] * block_size + block_id[:, 2]
    offset_id = offset_id[:, 0] * offset_size**2 + offset_id[:, 1] * offset_size + offset_id[:, 2]
    block_id += patch_size**3
    offset_id += block_size**3+patch_size**3  # Offset adjustment
    
    block_coords = np.concatenate([patch_id[..., None], block_id[..., None], offset_id[..., None]], axis=-1).astype(np.int64)
    sequence[:, :3][sp_mask] = block_coords  # update sequence with new coords
    sequence = sequence[:, :3]  # keep only the first three columns
    
    # convert to codes
    codes = []
    cur_patch_id = sequence[0, 0]
    cur_block_id = sequence[0, 1]
    
    codes.append(cur_patch_id)
    codes.append(cur_block_id)
    
    for i in range(len(sequence)):
        if sequence[i, 0] == special_token:
            if not use_special_block:
                codes.append(special_token)
            cur_patch_id = special_token
            cur_block_id = special_token
            
        elif sequence[i, 0] == cur_patch_id and sequence[i, 1] == cur_block_id:
            if block_compressed:
                codes.append(sequence[i, 2])
            else:
                codes.extend([sequence[i, 0], sequence[i, 1], sequence[i, 2]])
        elif sequence[i, 0] == cur_patch_id and sequence[i, 1] != cur_block_id:
            # If patch_id is the same but block_id is different
            # We need to handle this case as a transition to a new block
            codes.extend([sequence[i, 1], sequence[i, 2]])  # Add block_id, and offset_id
            cur_block_id = sequence[i, 1]
                
        else:
            if use_special_block and cur_patch_id == special_token:
                patch_id = sequence[i, 0] + special_block_base #center
            else:
                patch_id = sequence[i, 0]
            codes.extend([patch_id, sequence[i, 1], sequence[i, 2]])
            cur_patch_id = patch_id
            cur_block_id = sequence[i, 1]
    
    codes = np.array(codes).astype(np.int64)
    sequence = codes  # block-wise indexes
    
    return sequence.flatten()
    


def serialize(mesh: trimesh.Trimesh):
    # 1. patchify faces into patches
    sequence = patchified_mesh(mesh, special_token=-2)
    
    # 2. convert coordinates to block-wise indexes
    codes = get_block_representation(
        sequence, patch_size=4, block_size=8, offset_size=16, 
        block_compressed=True, special_token=-2, use_special_block=True
    )
    return codes    


def decode_block(sequence, compressed=True, patch_size=4, block_size=8, offset_size=16):
    
    # decode from compressed representation
    if compressed:
        res = []
        res_patch = 0
        res_block = 0
        for token_id in range(len(sequence)):                
            if block_size**3 + offset_size**3 + patch_size**3 > sequence[token_id] >= block_size**3 + patch_size**3: #offset
                res.append([res_patch, res_block, sequence[token_id]])
            elif patch_size**3 + block_size**3 > sequence[token_id] >= patch_size**3: #block
                res_block = sequence[token_id]
            elif patch_size**3 > sequence[token_id] >= 0: # patch
                res_patch = sequence[token_id]
            else:
                print('[Warning] too large offset idx!', token_id, sequence[token_id])
        sequence = np.array(res)
    
    patch_id, block_id, offset_id = np.array_split(sequence, 3, axis=-1)
    
    # from hash representation to xyz 
    coords = []
    block_id = block_id - patch_size**3
    offset_id = offset_id - block_size**3 - patch_size**3
    for i in [2, 1, 0]:
        # Decode patch coordinates
        patch_axis = patch_id // patch_size**i
        patch_id %= patch_size**i
        
        # Decode block coordinates
        block_axis = block_id // block_size**i
        block_id %= block_size**i
        
        # Decode offset coordinates
        offset_axis = offset_id // offset_size**i
        offset_id %= offset_size**i
        
        # Combine patch, block, and offset coordinates for final position
        axis = patch_axis * (block_size * offset_size) + block_axis * offset_size + offset_axis
        coords.append(axis)
    
    coords = np.concatenate(coords, axis=-1) # (nf 3)
    
    # back to continuous space
    coords = undiscretize(coords, num_discrete=512)

    return coords


def deserialize(sequence, patch_size=4, block_size=8, offset_size=16, compressed=True, special_token=-2, use_special_block=True):
    # decode codes back to coordinates
   
    special_block_base = block_size**3 + offset_size**3 + patch_size**3
    start_idx = 0
    vertices = []
    for i in range(len(sequence)):
        sub_seq = []
        if not use_special_block and (sequence[i] == special_token or i == len(sequence) - 1):
            sub_seq = sequence[start_idx:i]
            sub_seq = decode_block(sub_seq, compressed=compressed, patch_size=patch_size, block_size=block_size, offset_size=offset_size)
            start_idx = i + 1

        elif use_special_block and \
            (special_block_base <= sequence[i] < special_block_base + patch_size**3 or i == len(sequence)-1): #center
            if i != 0:
                sub_seq = sequence[start_idx:i] if i != len(sequence) - 1 else sequence[start_idx: i+1]
                if special_block_base <= sub_seq[0] < special_block_base + patch_size**3:
                    sub_seq[0] -= special_block_base
                sub_seq = decode_block(sub_seq, compressed=compressed, patch_size=patch_size, block_size=block_size, offset_size=offset_size)
                start_idx = i

        if len(sub_seq):
            center, sub_seq = sub_seq[0], sub_seq[1:]
            for j in range(len(sub_seq) - 1):
                vertices.extend([center.reshape(1, 3), sub_seq[j].reshape(1, 3), sub_seq[j+1].reshape(1, 3)])

    # (nf, 3)
    return np.concatenate(vertices, axis=0) 
