1. generate train dataset


export TASK_NAME=MNLI
export GLUE_DIR=/home/ubuntu/mnt/data/bert/glue
export BERT_DIR=/home/ubuntu/mnt/data/bert/model_dir/uncased_L-24_H-1024_A-16
export OUTPUT_DIR=/home/ubuntu/mnt/data/bert/data_dir

python ../data/create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --vocab_file=${BERT_DIR}/vocab.txt \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME}


 2. train

export TASK=MNLI
export BERT_DIR=/home/ubuntu/mnt/data/bert/model_dir/uncased_L-24_H-1024_A-16

 python run_classifier.py \
  --mode='train_and_eval' \
  --input_meta_data_path=/home/ubuntu/mnt/data/bert/data_dir/MNLI_meta_data \
  --train_data_path=/home/ubuntu/mnt/data/bert/data_dir/${TASK}_train.tf_record \
  --eval_data_path=/home/ubuntu/mnt/data/bert/data_dir/${TASK}_eval.tf_record \
  --bert_config_file=${BERT_DIR}/bert_config.json \
  --init_checkpoint=${BERT_DIR}/bert_model.ckpt \
  --train_batch_size=4 \
  --eval_batch_size=4 \
  --steps_per_loop=1 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --model_export_path=/home/ubuntu/mnt/data/bert/fined_model_dir/saved_model/1/ \
  --model_dir=/home/ubuntu/mnt/data/bert/fined_model_dir

  3. predict

   python run_classifier.py \
  --mode='predict' \       
  --input_meta_data_path=/home/ubuntu/mnt/data/bert/data_dir/MNLI_meta_data.short \
  --eval_data_path=/home/ubuntu/mnt/data/bert/data_dir/${TASK}_eval.tf_record \
  --bert_config_file=${BERT_DIR}/bert_config.json \

  --eval_batch_size=4 \ 
  --model_dir=/home/ubuntu/mnt/data/bert/fined_model_dir 

  4. request server
     python bert_request_server.py \
  --input_meta_data_path=/home/ubuntu/mnt/data/bert/data_dir/MNLI_meta_data.short \
  --eval_data_path=/home/ubuntu/mnt/data/bert/data_dir/${TASK}_eval.tf_record \
  --eval_batch_size=64 