# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train and evaluate the Transformer model.

See README for description of setting the training schedule and evaluating the
BLEU score.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np

# Import libraries
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.common import distribute_utils
from official.modeling import performance
from official.nlp.transformer import compute_bleu
from official.nlp.transformer import data_pipeline
from official.nlp.transformer import metrics
from official.nlp.transformer import misc
from official.nlp.transformer import optimizer
from official.nlp.transformer import transformer
from official.nlp.transformer import translate
from official.nlp.transformer.utils import tokenizer
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils

INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1


def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref,
                               distribution_strategy=None):
  """Translate file and report the cased and uncased bleu scores.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      params,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False,
      distribution_strategy=distribution_strategy)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score


def evaluate_and_log_bleu(model,
                          params,
                          bleu_source,
                          bleu_ref,
                          vocab_file,
                          distribution_strategy=None):
  """Calculate and record the BLEU score.

  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.

  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, params, subtokenizer, bleu_source, bleu_ref, distribution_strategy)

  logging.info("Bleu score (uncased): %s", uncased_score)
  logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score


class TransformerTask(object):
  """Main entry of Transformer model."""

  def __init__(self, flags_obj):
    """Init function of TransformerMain.

    Args:
      flags_obj: Object containing parsed flag values, i.e., FLAGS.

    Raises:
      ValueError: if not using static batch for input data on TPU.
    """
    self.flags_obj = flags_obj
    self.predict_model = None

    # Add flag-defined parameters to params object
    num_gpus = flags_core.get_num_gpus(flags_obj)
    self.params = params = misc.get_model_params(flags_obj.param_set, num_gpus)

    params["num_gpus"] = num_gpus
    params["use_ctl"] = flags_obj.use_ctl
    params["data_dir"] = flags_obj.data_dir
    params["model_dir"] = flags_obj.model_dir
    params["static_batch"] = flags_obj.static_batch
    params["max_length"] = flags_obj.max_length
    params["decode_batch_size"] = flags_obj.decode_batch_size
    params["decode_max_length"] = flags_obj.decode_max_length
    params["padded_decode"] = flags_obj.padded_decode
    params["max_io_parallelism"] = (
        flags_obj.num_parallel_calls or tf.data.experimental.AUTOTUNE)

    params["use_synthetic_data"] = flags_obj.use_synthetic_data
    params["batch_size"] = flags_obj.batch_size or params["default_batch_size"]
    params["repeat_dataset"] = None
    params["dtype"] = flags_core.get_tf_dtype(flags_obj)
    params["enable_tensorboard"] = flags_obj.enable_tensorboard
    params["enable_metrics_in_training"] = flags_obj.enable_metrics_in_training
    params["steps_between_evals"] = flags_obj.steps_between_evals
    params["enable_checkpointing"] = flags_obj.enable_checkpointing
    params["save_weights_only"] = flags_obj.save_weights_only

    self.distribution_strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=flags_obj.distribution_strategy,
        num_gpus=num_gpus,
        all_reduce_alg=flags_obj.all_reduce_alg,
        num_packs=flags_obj.num_packs,
        tpu_address=flags_obj.tpu or "")
    if self.use_tpu:
      params["num_replicas"] = self.distribution_strategy.num_replicas_in_sync
    else:
      logging.info("Running transformer with num_gpus = %d", num_gpus)

    if self.distribution_strategy:
      logging.info("For training, using distribution strategy: %s",
                   self.distribution_strategy)
    else:
      logging.info("Not using any distribution strategy.")

    performance.set_mixed_precision_policy(
        params["dtype"],
        flags_core.get_loss_scale(flags_obj, default_for_fp16="dynamic"))

  @property
  def use_tpu(self):
    if self.distribution_strategy:
      return isinstance(self.distribution_strategy,
                        tf.distribute.experimental.TPUStrategy)
    return False


  def eval(self):
    """Evaluates the model."""


    distribution_strategy = self.distribution_strategy if self.use_tpu else None

    # We only want to create the model under DS scope for TPU case.
    # When 'distribution_strategy' is None, a no-op DummyContextManager will
    # be used.
    with distribute_utils.get_strategy_scope(distribution_strategy):
      
      model = transformer.create_model(self.params, False)
      self._load_weights_if_possible(
          model,
          tf.train.latest_checkpoint(self.flags_obj.model_dir))
      model.summary()


    params=self.params
    batch_size = params["decode_batch_size"]
    # Read and sort inputs by length. Keep dictionary (original index-->new index
    # in sorted list) to write translations in the original order.
    input_file=self.flags_obj.bleu_source
    sorted_inputs, sorted_keys =translate._get_sorted_inputs(input_file)
    total_samples = len(sorted_inputs)
    num_decode_batches = (total_samples - 1) // batch_size + 1

    vocab_file=self.flags_obj.vocab_file
    subtokenizer = tokenizer.Subtokenizer(vocab_file)
    def input_generator():
      """Yield encoded strings from sorted_inputs."""
      for i in range(num_decode_batches):
        lines = [
            sorted_inputs[j + i * batch_size]
            for j in range(batch_size)
            if j + i * batch_size < total_samples
        ]
        
        lines = [translate._encode_and_add_eos(l, subtokenizer) for l in lines]
        if distribution_strategy:
          for j in range(batch_size - len(lines)):
            lines.append([tokenizer.EOS_ID])
        batch = tf.keras.preprocessing.sequence.pad_sequences(
            lines,
            maxlen=params["decode_max_length"],
            dtype="int32",
            padding="post")
        logging.info("Decoding batch %d out of %d.", i, num_decode_batches)
        yield batch

    @tf.function
    def predict_step(inputs):
      """Decoding step function for TPU runs."""

      def _step_fn(inputs):
        """Per replica step function."""
        tag = inputs[0]
        val_inputs = inputs[1]
        val_outputs, _ = model([val_inputs], training=False)
        return tag, val_outputs

      return distribution_strategy.run(_step_fn, args=(inputs,))

    translations = []
    if distribution_strategy:
      num_replicas = distribution_strategy.num_replicas_in_sync
      local_batch_size = params["decode_batch_size"] // num_replicas
    for i, text in enumerate(input_generator()):
      if distribution_strategy:
        text = np.reshape(text, [num_replicas, local_batch_size, -1])
        # Add tag to the input of each replica with the reordering logic after
        # outputs, to ensure the output order matches the input order.
        text = tf.constant(text)

        @tf.function
        def text_as_per_replica():
          replica_context = tf.distribute.get_replica_context()
          replica_id = replica_context.replica_id_in_sync_group
          return replica_id, text[replica_id]

        text = distribution_strategy.run(text_as_per_replica)
        outputs = distribution_strategy.experimental_local_results(
            predict_step(text))
        tags, unordered_val_outputs = outputs[0]
        tags = [tag.numpy() for tag in tags._values]
        unordered_val_outputs = [
            val_output.numpy() for val_output in unordered_val_outputs._values]
        # pylint: enable=protected-access
        val_outputs = [None] * len(tags)
        for k in range(len(tags)):
          val_outputs[tags[k]] = unordered_val_outputs[k]
        val_outputs = np.reshape(val_outputs, [params["decode_batch_size"], -1])
      else:
        val_outputs, _ = model.predict(text)

      length = len(val_outputs)
      for j in range(length):
        if j + i * batch_size < total_samples:
          translation = translate._trim_and_decode(val_outputs[j], subtokenizer)
          translations.append(translation)
    
          logging.info("Translating:\n\tInput: %s\n\tOutput: %s",
                      sorted_inputs[j + i * batch_size], translation)

    # Write translations in the order they appeared in the original file.
    output_file="server_output.txt"
    if output_file is not None:
      if tf.io.gfile.isdir(output_file):
        raise ValueError("File output is a directory, will not save outputs to "
                        "file.")
      logging.info("Writing to file %s", output_file)
      with tf.io.gfile.GFile(output_file, "w") as f:
        for i in sorted_keys:
          f.write("%s\n" % translations[i])

    uncased_score = compute_bleu.bleu_wrapper(input_file, output_file, False)
    cased_score = compute_bleu.bleu_wrapper(input_file, output_file, True)
    print("Bleu score (uncased): %s", uncased_score)
    print("Bleu score (cased): %s", cased_score)






  def _load_weights_if_possible(self, model, init_weight_path=None):
    """Loads model weights when it is provided."""
    if init_weight_path:
      logging.info("Load weights: {}".format(init_weight_path))
      if self.use_tpu:
        checkpoint = tf.train.Checkpoint(
            model=model, optimizer=self._create_optimizer())
        checkpoint.restore(init_weight_path)
      else:
        model.load_weights(init_weight_path)
    else:
      logging.info("Weights not loaded from path:{}".format(init_weight_path))

  def _create_optimizer(self):
    """Creates optimizer."""
    params = self.params
    lr_schedule = optimizer.LearningRateSchedule(
        params["learning_rate"], params["hidden_size"],
        params["learning_rate_warmup_steps"])
    opt = tf.keras.optimizers.Adam(
        lr_schedule,
        params["optimizer_adam_beta1"],
        params["optimizer_adam_beta2"],
        epsilon=params["optimizer_adam_epsilon"])

    opt = performance.configure_optimizer(
        opt,
        use_float16=params["dtype"] == tf.float16,
        use_graph_rewrite=self.flags_obj.fp16_implementation == "graph_rewrite",
        loss_scale=flags_core.get_loss_scale(
            self.flags_obj, default_for_fp16="dynamic"))

    return opt


def main(_):
  flags_obj = flags.FLAGS
  if flags_obj.enable_mlir_bridge:
    tf.config.experimental.enable_mlir_bridge()
  task = TransformerTask(flags_obj)

  # Execute flag override logic for better model performance
  if flags_obj.tf_gpu_thread_mode:
    keras_utils.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)

  
  task.eval()
  


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  misc.define_transformer_flags()
  app.run(main)
