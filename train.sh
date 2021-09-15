GPU="4"
EPOCH=5.0
MODEL_NAME=hfl/chinese-bert-wwm-ext
SEED=1234
FORMATION_LR=4e-4
LR=5e-5
BSZ=32
DATA_DIR="data/v7"
TRAIN_FILE="train.tsv"
DEV_FILE="dev.tsv"
TEST_FILE="test.tsv"
CUDA_VISIBLE_DEVICES=$GPU python run.py --seed $SEED --formation_lr $FORMATION_LR \
  --model_name_or_path  $MODEL_NAME \
  --do_eval --do_train --do_predict --learning_rate $LR  --fp16 \
  --train_file $DATA_DIR/$TRAIN_FILE \
  --validation_file $DATA_DIR/$DEV_FILE --save_total_limit 10 \
  --test_file $DATA_DIR/$TEST_FILE \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ  --per_device_eval_batch_size $BSZ  \
  --logging_steps 500 --evaluation_strategy steps --metric_for_best_model "f1"  --load_best_model_at_end \
  --learning_rate $LR --formation_lr $FORMATION_LR \
  --num_train_epochs $EPOCH \
  --output_dir results/formbert_outputs
