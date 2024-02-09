CHECKPOINT=''   # the path of checkpoints
OUTPUT=''       # output path

CUDA_VISIBLE_DEVICES=0 python3 generate_rm.py \
    --data_path data/mix \
    --model_path lmsys/vicuna-7b-1.3 \
    --checkpoint_path $CHECKPOINT \
    --output_path $OUTPUT \
    --eval_batch_size 32 \
    --ranking \
    --ranking_way last 