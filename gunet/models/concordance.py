# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A learned fovea model with extra convolutions to reduce upscale noise.

This model is like fovea_core, except:
1) The defovea operations are done with overlap to minimize the screen door
effect.
2) Foveation and defoveation operations are followed by in-scale
model_util.modules,
to make scale changes more gradual and further minimize the screen door
effect.
3) The network is substantially taller, including a taller top tower.
The hope is that by adding depth, the predicted pixels will be more
concordant, thus the network name.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict

import tensorflow as tf
import numpy as np

# pylint: disable=g-bad-import-order
import tensorcheck
import util
from models import model_util
import unet_util
from unet_layers import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax,
                            cross_entropy, dense_block, conv2d_a, deconv2d_a)

logging = tf.logging
lt = tf.contrib.labeled_tensor
slim = tf.contrib.slim

# The standard convolution sizes for in-scale Minception model_util.modules.
IN_SCALE_EXPANSION_SIZE = 3
IN_SCALE_REDUCTION_SIZE = 1

# We project to a dimension of this size before learning layers with
# nonlinearities.
INITIAL_PROJECTION_DIMENSION = 32

@tensorcheck.well_defined()
def core(
    base_depth: int,
    is_train: bool,
    input_op: tf.Tensor,
    name: str = None,
) -> tf.Tensor:
  """A learned fovea model with extra convolutions to reduce upscale noise.

  Args:
    base_depth: The depth of a 1x1 layer.
      Used as a multiplier when computing layer depths from size.
    is_train: Whether we're training.
    input_op: The input.
    name: Optional op name.

  Returns:
    The output of the core model as an embedding layer.
    Network heads should take this layer as input.
  """
  keep_prob = 0.2
  layers=3
  filter_size=3
  pool_size=2
  dense_depth_down = [2, 4, 8]
  dense_depth_bottom = 8
  dense_depth_up = [1, 2, 4]
  growth_r = 16
    
    
  with tf.name_scope(name, 'concordance_core', [input_op]) as scope:
    # Ensure the input data is in the range [0.0, 1.0].
    input_op = tensorcheck.bounds_unlabeled(0.0, 1.0, input_op)

    input_op = slim.conv2d(
        input_op, INITIAL_PROJECTION_DIMENSION, [1, 1], activation_fn=None)

  print ("input is: ", input_op)

  x = util.crop_center_unlabeled(128, input_op)
  [_, _, _, channels] = x.shape.as_list()
  conv1 = conv2d_a(x, channels, filter_size, 1, keep_prob)
  x = conv2d_a(conv1, channels, filter_size, 1, keep_prob)
  print ("x is: ", x)


  x_reduce = util.crop_center_unlabeled(256, input_op)
  conv1 = conv2d_a(x_reduce, channels, filter_size, 1, keep_prob)
  conv2 = conv2d_a(conv1, channels, filter_size, 1, keep_prob)
  x_pool = max_pool(conv2, pool_size)
  conv3 = conv2d_a(x_pool, channels, filter_size, 1, keep_prob)
  x_reduce = conv2d_a(conv3, channels, filter_size, 1, keep_prob)
  print ("x_reduce is: ", x_reduce)

  x_expand = util.crop_center_unlabeled(64, input_op)
  conv1 = conv2d_a(x_expand, channels, filter_size, 1, keep_prob)
  conv2 = conv2d_a(conv1, channels, filter_size, 1, keep_prob)
  x_expand = deconv2d_a(conv2, channels, filter_size,  pool_size, keep_prob)
  print ("x_expand is: ", x_expand)

  x = tf.concat([x, x_reduce, x_expand], 3)
  print ("The concated x is ", x)


  x = slim.conv2d(
        x, INITIAL_PROJECTION_DIMENSION, [1, 1], activation_fn=None)
  print("The slim x is ", x)


  """
  Creates a new convolutional unet for the given parametrization.
  :param x: input tensor, shape [?,nx,ny,channels]
  :param keep_prob: dropout probability tensor
  :param channels: number of channels in the input image
  :param n_class: number of output labels
  :param layers: number of layers in the net
  :param features_root: number of features in the first layer
  :param filter_size: size of the convolution filter
  :param pool_size: size of the max pooling operation
  :param summaries: Flag if summaries should be created

  """

  # Placeholder for the input image
  with tf.name_scope("nominate"):
      in_node = x
      batch_size = tf.shape(x)[0]

  
  dw_h_convs = OrderedDict()

  # down layers
  for layer in range(0, layers):
      with tf.name_scope("down_conv_{}".format(str(layer))):
          dense = dense_block(in_node, growth_r, dense_depth_down[layer], filter_size, keep_prob)
          [_, _, _, dense_channels] = dense.shape.as_list()
          conv = conv2d_a(dense, dense_channels, 1, 1, keep_prob) 
          dw_h_convs[layer] = conv
          pools= max_pool(dw_h_convs[layer], pool_size)
          in_node = pools
          print("The output of each uplayers is : ", in_node)

  with tf.name_scope("down_conv"):
    dense = dense_block(in_node, growth_r, dense_depth_bottom, filter_size, keep_prob)
    [_, _, _, dense_channels] = dense.shape.as_list()
    conv = conv2d_a(dense, dense_channels, 1, 1, keep_prob)
    in_node = conv

  print("The output of the bottom layers is: ", in_node)

  # up layers
  for layer in range(layers - 1, -1, -1):
      with tf.name_scope("up_conv_{}".format(str(layer))):
          [_, _, _, deconv_channels] = in_node.shape.as_list()
          h_deconv = deconv2d_a(in_node, deconv_channels//2, filter_size,  pool_size, keep_prob)
          h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
          [_, _, _, concat_channels] = h_deconv_concat.shape.as_list()
          conv = conv2d_a(h_deconv_concat, concat_channels//2, 1, 1, keep_prob)
          dense  = dense_block(conv, growth_r, dense_depth_up[layer], filter_size, keep_prob)
          in_node = dense
          print("The output of each uplayers is : ", in_node)
  #   # Output Map
  # with tf.name_scope("output_map"):
  #     weight = weight_variable([1, 1, features_root, n_class], stddev)
  #     bias = bias_variable([n_class], name="bias")
  #     conv = conv2d(in_node, weight, bias, tf.constant(1.0))
  #     output_map = tf.nn.relu(conv)
  #     up_h_convs["out"] = output_map

  # if summaries:
  #     with tf.name_scope("summaries"):
  #         for i, (c1, c2) in enumerate(convs):
  #             tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
  #             tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

  #         for k in pools.keys():
  #             tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

  #         for k in deconv.keys():
  #             tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconv[k]))

  #         for k in dw_h_convs.keys():
  #             tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

  #         for k in up_h_convs.keys():
  #             tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

  # variables = []
  # for w1, w2 in weights:
  #     variables.append(w1)
  #     variables.append(w2)

  # for b1, b2 in biases:
  #     variables.append(b1)
  #     variables.append(b2)

  return tf.identity(in_node, name=scope)
