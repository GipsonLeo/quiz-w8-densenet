"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            pass
            ##########################
            # Put your code here.
            #------------------------
            # slim function ready:
            # default parameter of slim.conv2d : 
            # https://blog.csdn.net/wuguangbin1230/article/details/79218920 
            # 默认有补零、激活、初始化，不带batchNorm
            # tf.contrib.slim.conv2d(...,padding='SAME',activation_fn=nn.relu,...,normalizer_params=None,...,
            #                weights_initializer=initializers.xavier_initializer(),...,trainable=True)
            
            # default parameter of slim.avg_pool2d :
            # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py
#             def avg_pool2d(inputs,
#                kernel_size,
#                stride=2,
#                padding='VALID',
#                data_format=DATA_FORMAT_NHWC,
#                outputs_collections=None,
#                scope=None):
#             def flatten(inputs, outputs_collections=None, scope=None):
#              # Flattens the input while maintaining the batch_size.
#              # Assumes that the first dimension represents the batch.
    
            #def fully_connected(inputs,
#                     num_outputs,
#                     activation_fn=nn.relu,
#                     normalizer_fn=None,
#                     normalizer_params=None,
#                     weights_initializer=initializers.xavier_initializer(),
#                     weights_regularizer=None,
#                     biases_initializer=init_ops.zeros_initializer(),
#                     biases_regularizer=None,
#                     reuse=None,
#                     variables_collections=None,
#                     outputs_collections=None,
#                     trainable=True,
#                     scope=None):
            #------------------------
            # DenseNet layers are very narrow (e.g., 12 filters per layer),so simply set small value 12 for fast training
            nb_filter = 12 # 64
            # From architecture for cifar10 (Table 1 in the paper)
            nb_layers = [6,12,24,16] # For DenseNet-121  
            
            #------------------------
            # Initial convolution
            # include: padding conv2d batchNorm Activation Pooling
            
            # initial convolution in paper:
            # In our experiments on ImageNet, we use a DenseNet-BC structure 
            # with 4 dense blocks on 224×224 input images.
            # The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2
            
            scope='conv1'
            x = slim.conv2d(inputs=images,num_outputs=2*growth,kernel_size=[7, 7], stride=2, padding='SAME',  
                  weights_initializer=trunc_normal(stddev =0.01),  
                  weights_regularizer=None, scope=scope)  #slim.l2_regularizer(0.0005) #init_conv
            x = slim.batch_norm(x, scope=scope + '_bn')
            # x = tf.nn.relu(x) #there is Activation in conv2d 
            x= slim.max_pool2d(x, [3, 3], stride=2, scope=scope +'_pool')
            # Note : map size = 224/2/2 = 56,that is 56x56
            #------------------------
            # Add dense blocks
            nb_dense_block = len(nb_layers)
            # later work: think about stack and repeat for better reading
            for block_idx in range(nb_dense_block - 1):
                # dense_block
                stage = block_idx+1  
                x = block(net=x, layers=nb_layers[block_idx], growth=growth,scope='dense_block'+str(stage)) 
                # transition_block
                num_outputs = growth #reduce_dim(input_feature = x)
                x = bn_act_conv_drp(current=x, num_outputs=num_outputs, kernel_size=[1, 1], scope='trans_block'+str(stage)+'conv') 
                x = slim.avg_pool2d(inputs=x, kernel_size=[2, 2], stride=2, scope='trans_block'+str(stage)+'pool') 
                
            # Note : map size = 56/2/2/2 = 7,that is 7x7
            #------------------------
            # Classification Layer
            final_stage = stage + 1
            x = block(net=x, layers=nb_layers[-1], growth=growth,scope='dense_block'+str(final_stage))
            #GlobalAveragePooling2D
            x = slim.avg_pool2d(inputs=x,kernel_size=[6, 6], scope='global_pool') 
            x = slim.flatten(x, scope='flatten')
            #fully_connected
            logits = slim.fully_connected(inputs=x, num_outputs=num_classes, activation_fn=None, scope='fc')
            end_points = tf.nn.softmax(logits, name='Predictions')  
            ##########################

    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
        [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
            [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode='FAN_IN', uniform=False),
        activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224
