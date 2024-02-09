
deepspeed --num_gpus=8 --master_port 12345 main.py \
    --gradient_checkpointing \
    --deepspeed config/ds_config_zero2.json \
	--base_model hf_models/vicuna-7b-1.3 \
	--data_path data/mix \
	--output_dir output/mix/themis \
	--num_epochs 5 \
	--batch_size 64 \
    --learning_rate 1e-5 \
	--cutoff_len 512 \
	--micro_batch_size=1 \
	--lr_scheduler_type cosine \
    --add_eos_token \
    --lm \
    --ranking \
    --ranking_way 'last' \
    --invoke_tool \
    --weight_decay 0.1 \
    --add_special_tokens
