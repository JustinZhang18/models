## Building a model training image
`docker build -t {image_name} .`
E.g. `docker build -t huaifeng/tensorflow:models_2.4 .`
 
## Training a model
### Trainning in a local machine
```
python3 official/vision/image_classification/mnist_main.py \
  --model_dir=/home/ubuntu/repos/models/model_dir \
  --data_dir=/home/ubuntu/repos/models/data_dir \
  --train_epochs=10 \
  --distribution_strategy=one_device \
  --num_gpus=1 \
  --download
```

### Training in a docker container
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
E.g.
```
 docker run --rm \
             -v /home/ubuntu/repos/models:/models \
             -v /home/ubuntu/repos/models/model_dir:/model_dir \
             -v /home/ubuntu/repos/models/data_dir:/data_dir \
             --gpus all \
  huaifeng/tensorflow:models_2.4 \
  python3 /models/official/vision/image_classification/mnist_main.py \
  --model_dir=/model_dir \
  --data_dir=/data_dir \
  --train_epochs=10 \
  --distribution_strategy=one_device \
  --num_gpus=1 \
  --download

```

## Evaluating
### Evalating in a local machine
```
python3 official/vision/image_classification/mnist_inference.py \
  --model_dir=/home/ubuntu/repos/models/model_dir \
  --data_dir=/home/ubuntu/repos/models/data_dir \
  --num_gpus=1 
```

### Evaluating in a docker container
to be done.


## Serving
## Servering in a docker container
```
 docker run  --gpus all -t --rm -p 8501:8501 \
    -v "/home/ubuntu/repos/models/model_dir/saved_model/:/models/mnist/" \
    -e MODEL_NAME=mnist \
    tensorflow/serving:2.4.0-gpu >server.log  2>&1
```
## Request a server
```
 python3 official/vision/image_classification/model_request_server.py --data_dir=/home/ubuntu/repos/models/data_dir --url=http://localhost:8501/v1/models/mnist:predict
```

```
python3 /home/ubuntu/repos/models/official/nlp/transformer/transformer_request_server.py \
    --vocab_file=$VOCAB_FILE   \
    --bleu_source=$DATA_DIR/newstest2014.en.pred \
    --bleu_ref=$DATA_DIR/newstest2014.de.pred \
     >req_inference_server.log 2>&1
```