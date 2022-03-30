1. `docker build -t {image_name} .`
2. 
trainning in a local machine:
```
python3 official/vision/image_classification/mnist_main.py \
  --model_dir=/home/ubuntu/repos/models/model_dir \
  --data_dir=/home/ubuntu/repos/models/data_dir \
  --train_epochs=10 \
  --distribution_strategy=one_device \
  --num_gpus=1 \
  --download
```

training in a docker container:
```
  docker run --rm \
             -v /home/ubuntu/repos/models:/models \
             -v /home/ubuntu/repos/models/model_dir:/model_dir \
             -v /home/ubuntu/repos/models/data_dir:/data_dir \
             --gpus all \
  {image_name} \
  python3 /models/official/vision/image_classification/mnist_main.py \
  --model_dir=/model_dir \
  --data_dir=/data_dir \
  --train_epochs=10 \
  --distribution_strategy=one_device \
  --num_gpus=1 \
  --download

```
trainning in a local machine:
```
python3 official/vision/image_classification/mnist_inference.py \
  --model_dir=/home/ubuntu/repos/models/model_dir \
  --data_dir=/home/ubuntu/repos/models/data_dir \
  --num_gpus=1 
```

inference in a docker container:
