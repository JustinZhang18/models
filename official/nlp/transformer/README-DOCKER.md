python3 transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --batch_size=4096 --train_steps=2000 \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET



python /home/ubuntu/repos/models/official/nlp/transformer/transformer_main.py \
  --model_dir=$MODEL_DIR \
  --data_dir=$DATA_DIR \
  --vocab_file=$DATA_DIR/vocab.ende.32768 \
  --bleu_source=$DATA_DIR/newstest2014.en \
  --bleu_ref=$DATA_DIR/newstest2014.end \
  --batch_size=2048 \
  --train_steps=200 \
  --param_set=$PARAM_SET 
  



      docker run --rm \
             -v /home/ubuntu/repos/models:/models \
             -v /home/ubuntu/repos/models/official/nlp/transformer/model_big:/model_big \
             -v /home/ubuntu/repos/models/official/nlp/transformer/data:/data_dir \
             --gpus all \
  huaifeng/tensorflow:models_2.4 \
  python3 /models/official/nlp/transformer/transformer_main.py \
  --model_dir=/model_big \
  --data_dir=/data_dir \
  --batch_size=4096 \
  --train_steps=2000 \
  --vocab_file= /data_dir/vocab.ende.32768 \
  --param_set=big 


  python3 /home/ubuntu/repos/cimplifier/bare-metal/code/slim.py tensorflow/serving:2.4.0-gpu ldconfig_slimmed  935c6b97f09c 6124  0509.log  > debloating.log   


  PARAM_SET=big
  DATA_DIR=/home/ubuntu/repos/models/official/nlp/transformer/data
  MODEL_DIR=/home/ubuntu/repos/models/official/nlp/transformer/model_big
  VOCAB_FILE=$DATA_DIR/vocab.ende.32768



# Train the model for 100000 steps and evaluate every 5000 steps on a single GPU.
# Each train step, takes 4096 tokens as a batch budget with 64 as sequence
# maximal length.
python3 /home/ubuntu/repos/models/official/nlp/transformer/transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --train_steps=2000 --steps_between_evals=300 \
    --batch_size=4096 --max_length=64 \
    --bleu_source=$DATA_DIR/newstest2014.en \
    --bleu_ref=$DATA_DIR/newstest2014.de \
    --num_gpus=1 \
    --enable_time_history=false > train.log 2>&1


predict test

python3 /home/ubuntu/repos/models/official/nlp/transformer/transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET \
    --mode=predict  \
    --max_length=64 \
    --bleu_source=$DATA_DIR/newstest2014.en.pred \
    --bleu_ref=$DATA_DIR/newstest2014.de.pred \
    --num_gpus=0 \
    --enable_time_history=false > predict.log 2>&1




    