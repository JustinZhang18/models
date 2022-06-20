## Set up
DATA_DIR=/home/ubuntu/repos/models/official/nlp/transformer/data
PARAM_SET=big
MODEL_DIR=/home/ubuntu/repos/models/official/nlp/transformer/model_big
VOCAB_FILE=$DATA_DIR/vocab.ende.32768
 export PYTHONPATH=$PYTHONPATH:/home/ubuntu/repos/models
conda activate tf_models_env 



request server
python3 /home/ubuntu/repos/models/official/nlp/transformer/transformer_request_server.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET  \
    --max_length=64 \
    --bleu_source=$DATA_DIR/newstest2014.en.pred \
    --bleu_ref=$DATA_DIR/newstest2014.de.pred \
    --num_gpus=0  --enable_time_history=false >req_inference_server.log 2>&1


demo train:
python3 /home/ubuntu/repos/models/official/nlp/transformer/transformer_main.py --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \          
    --vocab_file=$VOCAB_FILE --param_set=$PARAM_SET  \
    --train_steps=10  \                                                                                       
    --batch_size=4096 --max_length=64 \           
    --num_gpus=1 \                             
    --enable_time_history=false > train.log 2>&1