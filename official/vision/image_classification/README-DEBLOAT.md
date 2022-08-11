python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=/home/ubuntu/mnt/data/resnet/mode_dir \
  --data_dir=/home/ubuntu/mnt/data/resnet/data_dir \
  --config_file=/home/ubuntu/repos/models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml \
  


docker run --gpus all -v 

huaifeng/tensorflow:models_2.4_new \


python3 classifier_trainer.py \
  --mode=export_only \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=/home/ubuntu/mnt/data/resnet/mode_dir \
  --data_dir=/home/ubuntu/mnt/data/resnet/data_dir \
  --config_file=/home/ubuntu/repos/models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml 


  request server:
   python3 resnet_request_server.py \
  --model_type=resnet \
  --dataset=imagenet \
  --data_dir=/home/ubuntu/mnt/data/resnet/data_dir \
  --config_file=/home/ubuntu/repos/models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml
  