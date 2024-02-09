deepspeed --num_gpus=8 --master_port 12345 main.py \
    --deepspeed config/ds_config_zero2.json \
    --gradient_checkpointing \
	--base_model lmsys/vicuna-7b-1.3 \
	--data_path data/mix \
	--output_dir output/mix/rm_vicuna7b \
	--num_epochs 5 \
	--batch_size 64 \
    --learning_rate 1e-5 \
	--cutoff_len 2048 \
	--micro_batch_size=1 \
	--lr_scheduler_type cosine \
    --add_eos_token \
    --ranking \
    --ranking_way 'last' \
    --weight_decay 0.1
    
