
CHECKPOINT=''   # the path of checkpoints
OUTPUT=''       # output path

deepspeed --num_gpus=8 --master_port 28005 generate_themis.py \
        --data_path data/mix \
        --model_path lmsys/vicuna-7b-1.3 \
        --checkpoint_path $CHECKPOINT \
        --output_path $OUTPUT\
        --max_iteractions 3 \
        --cutoff_len 2048 \
        --max_new_token 1024 \
        --eval_batch_size 8 \
        --device auto \
        --ranking \
        --ranking_way last \
        --invoke_tool \
        --add_special_tokens
