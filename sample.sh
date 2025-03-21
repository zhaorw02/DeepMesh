CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=0 --master_port=12345 sample.py \
    --model_path "your_model_path" \
    --steps 120000 \
    --input_path examples \
    --output_path mesh_output \
    --repeat_num 4 \
    --uid_list "" \
    --temperature 0.5 \
