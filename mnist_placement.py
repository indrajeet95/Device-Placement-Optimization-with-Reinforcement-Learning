# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

FLAGS = None
mnist = None
#train_writer,test_writer = None,None
merged, accuracy, train_step = None,None,None
x, y_, keep_prob = None,None,None
mnist_sess = None
Graph = None
init_op = None
def mnist_model():
    # Create a multilayer model.
    # Input placeholders
    global Graph
    Graph = tf.Graph()
    global merged, accuracy, train_step
    global x, y_, keep_prob
    with Graph.as_default():  
      #tf.reset_default_graph()
      with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')

      with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

      # We can't initialize these variables to 0 - the network will get stuck.
      def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

      def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

      def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)

      def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
          """Reusable code for making a simple neural net layer.

          It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
          It also sets up name scoping so that the resultant graph is easy to read,
          and adds a number of summary ops.
          """
          #with tf.device("/device:GPU:0"):
          # Adding a name scope ensures logical grouping of the layers in the graph.
          with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
              weights = weight_variable([input_dim, output_dim])
              variable_summaries(weights)
            with tf.name_scope('biases'):
              biases = bias_variable([output_dim])
              variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
              preactivate = tf.matmul(input_tensor, weights) + biases
              tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

      hidden1 = nn_layer(x, 784, 500, 'layer1')

      with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        #with tf.device("/device:GPU:0"):
        dropped = tf.nn.dropout(hidden1, keep_prob)

      # Do not apply softmax activation yet, see below.
      y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

      with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.losses.sparse_softmax_cross_entropy on the
        # raw logit outputs of the nn_layer above, and then average across
        # the batch.
        with tf.name_scope('total'):
          cross_entropy = tf.losses.sparse_softmax_cross_entropy(
              labels=y_, logits=y)
      tf.summary.scalar('cross_entropy', cross_entropy)
      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

      with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar('accuracy', accuracy)

      # Merge all the summaries and write them out to
      # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
      merged = tf.summary.merge_all()
      global init_op
      init_op = tf.global_variables_initializer() 
      # Train the model, and also write summaries.
      # Every 10th step, measure test-set accuracy, and write test summaries
      # All other steps, run train_step on training data, & add training summaries
    #return G
def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    global x, y_, keep_prob
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

def make_seesion():
    #global train_writer,test_writer
    global mnist_sess
    mnist_sess = tf.Session(graph=Graph,config=tf.ConfigProto(log_device_placement=False,
    allow_soft_placement=False))
    #train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', mnist_sess.graph)
    #tf.train.write_graph(mnist_sess.graph, './', 'train.proto', as_text = False)
    #test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    #return mnist_sess

def apply_placement():
    mnist_sess.run(init_op)
    #with mnist_sess.as_default():
  	#tf.global_variables_initializer().run()
def train():
  #global train_writer,test_writer
  # Import data
  start_time = 0.0
  for i in range(FLAGS.max_steps):
    #if i % 10 == 0:  # Record summaries and test-set accuracy
      start_time = time.time()
      summary, acc = mnist_sess.run([merged, accuracy], feed_dict=feed_dict(False))
      start_time = time.time() - start_time
      #test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
  return start_time
  """
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = mnist_sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = mnist_sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  """
  #var_names = [v.name for v in tf.trainable_variables()]
  """
  with open("./extracted_parameters/0.txt", "w") as text_file:
    for mm in var_names:
        text_file.write(mm)
        text_file.write('\n')
  the_vars = mnist_sess.run( tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) )
  #print (the_vars)
  msd_index = 0
  for i in range(len(the_vars)):
    #print(the_vars[i].shape)
    #print (the_vars[i])
    msd_index+=1
    np.savetxt('./extracted_parameters/'+str(msd_index)+'.xls', the_vars[i],delimiter='\t', newline='\n')
  """
  #train_writer.close()
  #test_writer.close()

def initial_flag():
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  return (FLAGS)

def initial_mnist():
  print (FLAGS)
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    fake_data=FLAGS.fake_data)
  return mnist
def initilaze():
  global FLAGS
  global mnist
  global mnist_sess
  FLAGS = initial_flag()
  mnist = initial_mnist()
  graph = mnist_model()
  #mnist_sess = make_seesion()
  #apply_placement(mnist_sess)
  #train(mnist_sess)


def main(_):
  pass
  """
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()
  """
  pass


if __name__ == '__main__':
  pass
  #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
