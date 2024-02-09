
deepspeed --num_gpus=8 --master_port 12345 baselines/run_bert.py \
	--deepspeed config/ds_config_zero2.json \
    --gradient_checkpointing \
	--base_model hf_models/bert-large-uncased \
	--data_path data/mix \
	--output_dir output/mix/rm_bert_large \
	--prompt_template_name instruction_template \
	--num_epochs 8 \
	--batch_size 128 \
    --learning_rate 1e-5 \
	--cutoff_len 512 \
	--micro_batch_size=8 \
    --reward_type 'linear'