CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master_port=12345 sample.py \
    --model_path "your_model_path/pytorch_model.bin" \
    --steps 90000 \
    --input_path examples \
    --output_path mesh_output \
    --repeat_num 4 \
    --uid_list "" \
    --temperature 0.5 \
