# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)

def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        [batch, height, width, channels] = x.shape.as_list()
        output_shape = [batch, height*2, width*2, channels//2]
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name="conv2d_transpose")

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")


def multihead_attention(inputs, total_key_filters, total_value_filters,
                            output_filters, num_heads, training, layer_type='SAME',
                            name=None):
    """2d Multihead scaled-dot-product attention with input/output transformations.
    
    Args:
        inputs: a Tensor with shape [batch, h, w, channels]
        total_key_filters: an integer. Note that queries have the same number 
            of channels as keys
        total_value_filters: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_filters and total_value_filters
        layer_type: a string, type of this layer -- SAME, DOWN, UP
        name: an optional string
    Returns:
        A Tensor of shape [batch, _h, _w, output_filters]
    
    Raises:
        ValueError: if the total_key_filters or total_value_filters are not divisible
            by the number of attention heads.
    """

    if total_key_filters % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                        "attention heads (%d)." % (total_key_filters, num_heads))
    if total_value_filters % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                        "attention heads (%d)." % (total_value_filters, num_heads))
    if layer_type not in ['SAME', 'DOWN', 'UP']:
        raise ValueError("Layer type (%s) must be one of SAME, "
                        "DOWN, UP." % (layer_type))

    with tf.variable_scope(
            name,
            default_name="multihead_attention_3d",
            values=[inputs]):

        # produce q, k, v
        q, k, v = compute_qkv(inputs, total_key_filters,
                        total_value_filters, layer_type)

        # after splitting, shape is [batch, heads, d, h, w, channels / heads]
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        # normalize
        key_filters_per_head = total_key_filters // num_heads
        q *= key_filters_per_head**-0.5

        # attention
        x = global_attention(q, k, v, training)
        
        x = combine_heads_2d(x)
        x = conv2d_a(x, output_filters, 1, 1)

        return x


def compute_qkv(inputs, total_key_filters, total_value_filters, layer_type):
    """Computes query, key and value.
    Args:
        inputs: a Tensor with shape [batch, h, w, channels]
        total_key_filters: an integer
        total_value_filters: and integer
        layer_type: String, type of this layer -- SAME, DOWN, UP
    
    Returns:
        q: [batch, _d, _h, _w, total_key_filters] tensor
        k: [batch, h, w, total_key_filters] tensor
        v: [batch, h, w, total_value_filters] tensor
    """

    # linear transformation for q
    if layer_type == 'SAME':
        w1 = weight_variable([1, filter_size, features, features // 2], stddev, name="w1")
        b1 = bias_variable([features // 2], name="b1")
        q = conv2d(inputs, w1, b1, keep_prob)

    elif layer_type == 'DOWN':
        q = conv2d_a(inputs, total_key_filters, 3, 2)
    elif layer_type == 'UP':
        q = deconv2d_att(inputs, total_key_filters, 3, 2)

    # linear transformation for k
    k = conv2d_a(inputs, total_key_filters, 1, 1)

    # linear transformation for k
    v = conv2d_a(inputs, total_value_filters, 1, 1)

    return q, k, v


def split_heads(x, num_heads):
    """Split channels (last dimension) into multiple heads (becomes dimension 1).
    
    Args:
        x: a Tensor with shape [batch, h, w, channels]
        num_heads: an integer
    
    Returns:
        a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
    """

    return tf.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
        x: a Tensor with shape [..., m]
        n: an integer.
    Returns:
        a Tensor with shape [..., n, m/n]
    """

    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    
    return ret


def global_attention(q, k, v, training, name=None):
    """global self-attention.
    Args:
        q: a Tensor with shape [batch, heads, _h, _w, channels_k]
        k: a Tensor with shape [batch, heads, h, w, channels_k]
        v: a Tensor with shape [batch, heads, h, w, channels_v]
        name: an optional string
    Returns:
        a Tensor of shape [batch, heads, _h, _w, channels_v]
    """
    with tf.variable_scope(
            name,
            default_name="global_attention",
            values=[q, k, v]):

        new_shape = tf.concat([tf.shape(q)[0:-1], [v.shape[-1].value]], 0)

        # flatten q,k,v
        q_new = flatten_2d(q)
        k_new = flatten_2d(k)
        v_new = flatten_2d(v)

        # attention
        output = dot_product_attention(q_new, k_new, v_new, bias=None,
                    training=training, dropout_rate=0.5, name="global")

        # putting the representations back in the right place
        output = scatter_2d(output, new_shape)

        return output

def flatten_2d(x):
    """flatten x."""

    x_shape = tf.shape(x)
    # [batch, heads, length, channels], length = h*w
    x = reshape_range(x, 2, 4, [tf.reduce_prod(x_shape[2:4])])

    return x


def scatter_2d(x, shape):
    """scatter x."""

    x = tf.reshape(x, shape)

    return x


def dot_product_attention(q, k, v, bias, training, dropout_rate=0.0, name=None):
    """Dot-product attention.
    Args:
        q: a Tensor with shape [batch, heads, length_q, channels_k]
        k: a Tensor with shape [batch, heads, length_kv, channels_k]
        v: a Tensor with shape [batch, heads, length_kv, channels_v]
        bias: bias Tensor
        dropout_rate: a floating point number
        name: an optional string
    Returns:
        A Tensor with shape [batch, heads, length_q, channels_v]
    """

    with tf.variable_scope(
            name,
            default_name="dot_product_attention",
            values=[q, k, v]):

        # [batch, num_heads, length_q, length_kv]
        logits = tf.matmul(q, k, transpose_b=True)

        if bias is not None:
            logits += bias

        weights = tf.nn.softmax(logits, name="attention_weights")

        # dropping out the attention links for each of the heads
        weights = tf.layers.dropout(weights, dropout_rate, training)

        return tf.matmul(weights, v)


def combine_heads_2d(x):
    """Inverse of split_heads_3d.
    Args:
        x: a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
    Returns:
        a Tensor with shape [batch, d, h, w, channels]
    """

    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 3, 1, 4]))


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
        x: a Tensor with shape [..., a, b]
    Returns:
        a Tensor with shape [..., a*b]
    """

    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]

    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)

    return ret

def dense_block(outs, growth_r, depth, kernel, keep_r):
    for i in range(depth):
        cur_outs = conv2d_a(
            outs, growth_r, kernel, 1, keep_r)
        outs = tf.concat([outs, cur_outs], 3)
    return outs


def conv2d_a(x, num_outs, kernel, stride, keep_r):
    outs = tf.contrib.layers.conv2d(
        x, num_outs, kernel, stride=stride, padding='SAME',
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    outs = tf.contrib.layers.dropout(
        outs, keep_r)
    return outs


def deconv2d_a(x, num_outs, kernel, stride, keep_r):
    outs = tf.contrib.layers.conv2d_transpose(x, num_outs, kernel, 
        stride=stride, padding='SAME',
        activation_fn=tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    outs = tf.contrib.layers.dropout(
        outs, keep_r)
    return outs







