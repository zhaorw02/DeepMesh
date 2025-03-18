import argparse
import os
import torch
from tqdm import tqdm
from lit_gpt.model_cache import GPTCache, Config
from safetensors.torch import load_file
#from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from sft.datasets.dataset import Sample_Dataset
import os
from tqdm import tqdm
import trimesh
from sft.datasets.serializaiton import BPT_deserialize
from sft.datasets.data_utils import to_mesh
import numpy as np
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup_distributed_mode(rank, world_size, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed_mode():
    dist.destroy_process_group()

def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def codes2mesh(codes,end_ui,output_path,local_rank,i):
    code = codes[0][1:]
    
    index = (code >= 4737).nonzero()

    if index.numel() > 0:
        code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
    else:
        code = code.cpu()
    
    code_ = copy.deepcopy(code)
    end = min( end_ui*1000,len(code_) )
    vertices = BPT_deserialize(code_[:end])
    if len(vertices) == 0:
        1/0
    vertices = vertices[..., [2, 1, 0]]
    faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
    mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
    num_faces = len(mesh.faces)
    mesh.export(f'{output_path}/{local_rank}_{i}_mesh_{end}.obj')

@ torch.no_grad()
def ar_sample_kvcache(gpt, prompt, pc, temperature=0.5, \
                        context_length=90000, window_size=9000,device='cuda',\
                        output_path=None,local_rank=None,i=None):
    gpt.eval()
    N        = prompt.shape[0]
    end_list = [0 for _ in range(N)]
    with tqdm(total=context_length-1, desc="Processing") as pbar:
        for cur_pos in range(prompt.shape[1], context_length):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if cur_pos >= 9001 and (cur_pos - 9001)%4500 == 0:
                    start = 4500 + ((cur_pos - 9001)//4500)*4500
                else:
                    start = cur_pos-1
                input_pos    = torch.arange(cur_pos, dtype=torch.long, device=device)
                prompt_input = prompt[:, start:cur_pos]
                logits = gpt(prompt_input, pc=pc,start = start,window_size=window_size, input_pos=input_pos)[:, -1]

            logits_with_noise = add_gumbel_noise(logits, temperature)
            next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)

            prompt = torch.cat([prompt, next_token], dim=-1)

            pbar.set_description(f"with start:{start},cur_pos:{cur_pos},length:{prompt_input.size(1)}")
            pbar.update(1)

            # if cur_pos%2000 == 0:
            #     codes2mesh(prompt,cur_pos,output_path,local_rank,i)
            for u in range(N):
                if end_list[u] == 0:
                    if next_token[u] == torch.tensor([4737], device=device):
                        end_list[u] = 1
            if sum(end_list) == N:
                break
    return prompt, cur_pos
def first(it):
    return it[0]

def custom_collate(data, pad_id):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output

def build_dataloader_func(bs, dataset, local_rank, world_size):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=0,
        drop_last = False,
        collate_fn = partial(custom_collate, pad_id = 4737)
    )
    return dataloader

@torch.inference_mode()
def get_model_answers(
    local_rank,
    world_size
):
    model_path  = args.model_path
    model_id    = args.model_id
    output_path = args.output_path,
    steps       = args.steps
    temperature = args.temperature
    path        = args.input_path
    output_path = args.output_path
    point_num   = args.point_num
    uid_list    = args.uid_list.split(",")
    repeat_num  = args.repeat_num

    setup_distributed_mode(local_rank, world_size)
    model_name = f"Diff_LLaMA_{model_id}M"
    config = Config.from_name(model_name)
    print(config)
    config.padded_vocab_size=(2*4**3)+(8**3)+(16**3) +1 +1  #4736+2
    config.block_size = 270000
    model = GPTCache(config).to('cuda')
    if model_path.split(".")[-1]=="safetensors":
        loaded_state = load_file(model_path)
    elif model_path.split(".")[-1]=="bin":
        loaded_state = torch.load(model_path, map_location='cpu',weights_only=False)
    else:
        loaded_state = get_fp32_state_dict_from_zero_checkpoint(model_path)
    model.load_state_dict(loaded_state, strict=False)
    model       = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
    train_dataset    = Sample_Dataset(point_num = point_num,uid_list = uid_list,path=path)
    train_dataloader = build_dataloader_func(1,train_dataset, local_rank, world_size)
    while True:
        for i, test_batch in tqdm(enumerate(train_dataloader)):
            cond_pc = test_batch['pc_normal'].to('cuda')
            
            points = cond_pc[0].cpu().numpy()
            point_cloud = trimesh.points.PointCloud(points[..., 0:3])
            point_cloud.export(f'{output_path}/{local_rank}_{i}_pc.ply')
            
            output_ids, _ = ar_sample_kvcache(model,
                                    prompt = torch.tensor([[4736]]).to('cuda').repeat(repeat_num,1),
                                    pc = cond_pc.repeat(repeat_num,1,1),
                                    window_size=9000,
                                    temperature=temperature,
                                    context_length=steps,
                                    device='cuda',
                                    output_path=output_path,local_rank=local_rank,i=i)
            for u in range(repeat_num):
                code = output_ids[u][1:]
                index = (code >= 4737).nonzero()
                if index.numel() > 0:
                    code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
                else:
                    code = code.cpu()
                vertices = BPT_deserialize(code)
                if len(vertices) == 0:
                    print("you got:",len(vertices))
                    continue
                vertices = vertices[..., [2, 1, 0]]
                
                faces = torch.arange(1, len(vertices) + 1).view(-1, 3)
                mesh = to_mesh(vertices, faces, transpose=False, post_process=True)
                mesh.export(f'{output_path}/{local_rank}_{i}_{u}_mesh.obj') 
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model_id", type=str, default="551", help="A custom name for the model."
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True, 
        help="sampling steps.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./output_pc_aug'
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=""
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=4
    )
    parser.add_argument(
        "--point_num",
        type=int,
        default=16384
    )
    parser.add_argument(
        "--uid_list",
        type=str,
        default=''
    )
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    get_model_answers(
                local_rank=local_rank,
                world_size=world_size
    )
