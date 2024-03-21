cuda_device=7
# set environment
export CUDA_VISIBLE_DEVICES=$cuda_device

# nohup python finetune_lm.py \
#     --model_name_or_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b' \
#     --config_name "baffo32/decapoda-research-llama-7B-hf" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 2048 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir './saved_weights' \
#     > test2.log 2>&1 &


# nohup python finetune_lm.py \
#     --model_name_or_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int4' \
#     --config_name "baffo32/decapoda-research-llama-7B-hf" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 2048 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir './saved_weights-int4' \
#     > llama-int4-finetune.log 2>&1 &

# nohup python finetune_lm.py \
#     --model_name_or_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int3-2' \
#     --config_name "baffo32/decapoda-research-llama-7B-hf" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 2048 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir './saved_weights-int3' \
#     > llama-int3-finetune.log 2>&1 &

# nohup python finetune_lm.py \
#     --model_name_or_path '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int2-2' \
#     --config_name "baffo32/decapoda-research-llama-7B-hf" \
#     --dataset_name c4 \
#     --num_train_epochs 1 \
#     --block_size 2048 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 8 \
#     --do_train \
#     --do_eval \
#     --max_train_samples 30000 \
#     --max_eval_samples 128 \
#     --learning_rate 1e-4 \
#     --overwrite_output_dir \
#     --output_dir './saved_weights-int2' \
#     > llama-int2-finetune.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python evaluate_ppl.py \
#     --model '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int3-2' \
#     --ctx_length 400 \
#     --lora_weights './saved_weights-int3' \
#     > llama-finetune.log 2>&1 &

# nohup python evaluate_ppl.py \
#     --model '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int3-2' \
#     --lora_weights './saved_weights-int3' \
#     > llama-finetune-int3-eval.log 2>&1 &


# nohup python evaluate_ppl.py \
#     --model '/mnt/nvme0/wangzining/smooth2/examples/llama-7b' \
#     --ctx_length 320 \
#     --lora_weights './saved_weights' \
#     > llama-finetune-llama7b.log 2>&1 &

# nohup python evaluate_ppl.py \
#     --model '/mnt/nvme0/wangzining/smooth2/examples/llama-7b-int2-2' \
#     --lora_weights './saved_weights-int2' \
#     > llama-finetune-int2-eval.log 2>&1 &

nohup python finetune_lm.py \
    --model_name_or_path '/mnt/nvme0/wangzining/smooth2/examples/chatglm-6b' \
    --config_name "THUDM/chatglm3-6b-base" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 2048 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 30000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir './saved_weights' \
    > chatglm.log 2>&1 &