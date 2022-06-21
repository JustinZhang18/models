# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""BERT classification or regression finetuning runner in TF 2.x."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import math
import os
import requests
import datetime
import numpy as np
from sklearn.metrics import accuracy_score

# Import libraries
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from official.common import distribute_utils
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.utils.misc import keras_utils

flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only', 'predict'],
    'One of {"train_and_eval", "export_only", "predict"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`. `predict`: takes a checkpoint and '
    'restores the model to output predictions on the test set.')
flags.DEFINE_string('train_data_path', None,
                    'Path to training data for BERT classifier.')
flags.DEFINE_string('eval_data_path', None,
                    'Path to evaluation data for BERT classifier.')
flags.DEFINE_string(
    'input_meta_data_path', None,
    'Path to file that contains meta data about input '
    'to be used for training and evaluation.')
flags.DEFINE_integer('train_data_size', None, 'Number of training samples '
                     'to use. If None, uses the full train data. '
                     '(default: None).')
flags.DEFINE_string('predict_checkpoint_path', None,
                    'Path to the checkpoint for predictions.')
flags.DEFINE_integer(
    'num_eval_per_epoch', 1,
    'Number of evaluations per epoch. The purpose of this flag is to provide '
    'more granular evaluation scores and checkpoints. For example, if original '
    'data has N samples and num_eval_per_epoch is n, then each epoch will be '
    'evaluated every N/n samples.')
flags.DEFINE_integer('train_batch_size', 32, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 32, 'Batch size for evaluation.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS

LABEL_TYPES_MAP = {'int': tf.int64, 'float': tf.float32}


def get_dataset_fn(input_file_pattern,
                   max_seq_length,
                   global_batch_size,
                   is_training,
                   label_type=tf.int64,
                   include_sample_weights=False,
                   num_samples=None):
    """Gets a closure to create a dataset."""

    def _dataset_fn(ctx=None):
        """Returns tf.data.Dataset for distributed BERT pretraining."""
        batch_size = ctx.get_per_replica_batch_size(
            global_batch_size) if ctx else global_batch_size
        dataset = input_pipeline.create_classifier_dataset(
            tf.io.gfile.glob(input_file_pattern),
            max_seq_length,
            batch_size,
            is_training=is_training,
            input_pipeline_context=ctx,
            label_type=label_type,
            include_sample_weights=include_sample_weights,
            num_samples=num_samples)
        return dataset

    return _dataset_fn


def custom_main(custom_callbacks=None, custom_metrics=None):
    """Run classification or regression.

    Args:
      custom_callbacks: list of tf.keras.Callbacks passed to training loop.
      custom_metrics: list of metrics passed to the training loop.
    """
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
        input_meta_data = json.loads(reader.read().decode('utf-8'))
    label_type = LABEL_TYPES_MAP[input_meta_data.get('label_type', 'int')]
    include_sample_weights = input_meta_data.get('has_sample_weights', False)

    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=FLAGS.distribution_strategy,
        num_gpus=FLAGS.num_gpus,
        tpu_address=FLAGS.tpu)
    eval_input_fn = get_dataset_fn(
        FLAGS.eval_data_path,
        input_meta_data['max_seq_length'],
        FLAGS.eval_batch_size,
        is_training=False,
        label_type=label_type,
        include_sample_weights=include_sample_weights)
    test_iter = iter(strategy.distribute_datasets_from_function(eval_input_fn))
    y_true = []
    y_pred = []
    stats = {}
    total_elapsed_time = 0

    for i in test_iter:
        for k, v in i[0].items():
            i[0][k] = v.numpy().tolist()
        data = json.dumps({
            "signature_name": "serving_default",
            "inputs": i[0]
        })
        headers = {"content-type": "application/json"}
        start = datetime.datetime.now()
        json_response = requests.post(
            "http://localhost:8501/v1/models/bert:predict", data=data, headers=headers)
        end = datetime.datetime.now()
        total_elapsed_time += (end-start).total_seconds() * 1000
        predictions = __resp_to_nparray(json_response)
        y_true.extend(i[1].numpy().tolist())
        y_pred.extend(predictions.tolist())

    stats['elapsed_time'] = total_elapsed_time
    acc = accuracy_score(y_true, y_pred)
    stats['accuracy'] = acc
    print(stats)


def __resp_to_nparray(resp):
    prob = json.loads(resp.text)['outputs']
    prob = np.array(prob)
    pred = np.argmax(prob, axis=1)

    return pred


def main(_):
    custom_main(custom_callbacks=None, custom_metrics=None)


if __name__ == '__main__':
    flags.mark_flag_as_required('input_meta_data_path')
    flags.mark_flag_as_required('eval_data_path')
    flags.mark_flag_as_required('eval_batch_size')
    app.run(main)
