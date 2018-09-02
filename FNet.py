import tensorflow as tf
slim = tf.contrib.slim

"""
Usage:
  with slim.arg_scope(FNet.vgg_arg_scope()):
    outputs, end_points = FNet.vgg19_slim(inputs)
"""
def vgg_arg_scope(weight_decay=0.0005):
   """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    biases_initializer=tf.zeros_initializer()):
  with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
    return arg_sc


def vgg19_slim(input,num_classes=1000,
              is_training=True,
              dropout_keep_prob=0.5,
              spatial_squeeze=True,
              scope='vgg_19',
              fc_conv_padding='VALID',
              global_pool=False):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope,'vgg_19',[inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv2d,slim.fully_connected,slim.max_pool2d],
                        outputs_collection=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      #use conv2d instead of fully_connected layers
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      # Convert end_points_collection into a end_point dict.
      #location:contrib.layers.python.layers.utils
      """
      output = OrderedDict((alias, tensor)
                       for tensor in ops.get_collection(collection)
                       for alias in get_tensor_aliases(tensor))
      """
      #OrderedDict
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
      end_points[sc.name + '/fc8'] = net
      return net, end_points





def vgg19_my(input):
  with slim.arg_scope([slim.conv2d,slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0,0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
      net = slim.repeat(input,2,conv2d,64,[3,3],scope='conv1')
      net = slim.max_pool2d(net,[2,2],scope='pool1') 
      net = slim.repeat(net,2,conv2d,128,[3,3],scope='conv2')
      net = slim.max_pool2d(net,[2,2],scope='pool2')
      net = slim.repeat(net,4,conv2d,256,[3,3],scope='conv3')
      net = slim.max_pool2d(net,[2,2],scope='pool3')
      net = slim.repeat(net,4,conv2d,512,[3,3],scope='conv4')
      net = slim.max_pool2d(net,[2,2],scope='pool4')
      net = slim.repeat(net,4,conv2d,512,[3,3],scope='conv5')
      net = slim.max_pool2d(net,[2,2],scope='pool5')

      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      net = slim.fully_connected(net, 4096, scope='fc6')
      net = slim.dropout(net, 0.5, scope='dropout6')
      net = slim.fully_connected(net, 4096, scope='fc7')
      net = slim.dropout(net, 0.5, scope='dropout7')   

  return net                   
"""

import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
vgg = nets.vgg

# Load the images and labels.
images, labels = ...

# Create the model.
predictions, _ = vgg.vgg_16(images)

# Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy(predictions, labels)
"""

"""
# Load the images and labels.
images, scene_labels, depth_labels = ...

# Create the model.
scene_predictions, depth_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)

# The following two lines have the same effect:
total_loss = classification_loss + sum_of_squares_loss
total_loss = slim.losses.get_total_loss(add_regularization_losses=False)
"""

"""
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

slim = tf.contrib.slim
vgg = nets.vgg

...

train_log_dir = ...
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
  # Set up the data loading:
  images, labels = ...

  # Define the model:
  predictions = vgg.vgg_16(images, is_training=True)

  # Specify the loss function:
  slim.losses.softmax_cross_entropy(predictions, labels)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total_loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)
"""
