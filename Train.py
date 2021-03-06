from __future__ import division

import os, errno
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Data import *
from Util import *

class Train:
    def __init__(self, data, series_length, num_epochs, batch_size, state_size, n_units, num_classes):
        assert isinstance(data, Data)
        self.data = data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.state_size = state_size
        self.n_units = n_units
        self.num_classes = num_classes
        self.num_batches = series_length // batch_size // n_units

        # To be set in loss_func() function
        self.x_batch = None
        self.y_batch = None
        self.init_state = None
        self.all_predictions = None

    def loss_func(self):
        self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.n_units])
        self.y_batch = tf.placeholder(tf.int32, [self.batch_size, self.n_units])
        self.init_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])

        w_state = tf.get_variable("w_state", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.state_size+1, self.state_size], dtype=tf.float32)
        b_state = tf.get_variable("b_state", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.state_size], dtype=tf.float32)
        w_out = tf.get_variable("w_out", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.state_size, self.num_classes], dtype=tf.float32)
        b_out = tf.get_variable("b_out", initializer=tf.contrib.layers.xavier_initializer(), shape=[1, self.num_classes], dtype=tf.float32)

        inputs = tf.unstack(self.x_batch, axis=1)
        outputs = tf.unstack(self.y_batch, axis=1)

        all_states = compute_states(inputs, w_state, b_state, self.batch_size, self.init_state)
        all_logits = compute_logits(all_states, w_out, b_out)
        self.all_predictions = [tf.nn.softmax(logits) for logits in all_logits]

        assert len(all_logits) == len(outputs)
        losses = list()
        for logits, labels in zip(all_logits, outputs):
            curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            losses.append(curr_loss)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    def train(self):
        directory = "checkpoints/vanilla"
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        total_loss = self.loss_func()
        train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            losses = list()
            for i in xrange(self.num_epochs):
                print "Epoch", i
                # Generate random data for each epoch
                array = self.data.get_data()
                inputs = array[0]
                outputs = array[1]
                inputs = np.reshape(inputs, [self.batch_size, -1])
                outputs = np.reshape(outputs, [self.batch_size, -1])
                curr_state = np.zeros((self.batch_size, self.state_size))

                for batch in xrange(self.num_batches):
                    start_idx = batch * self.n_units
                    end_idx = start_idx + self.n_units
                    x_batch = inputs[:, start_idx:end_idx]
                    y_batch = outputs[:, start_idx:end_idx]
                    _loss, _, _preds = sess.run([total_loss, train_step, self.all_predictions], \
                                         feed_dict={self.x_batch:x_batch,
                                                    self.y_batch:y_batch,
                                                    self.init_state:curr_state})
                    #print "Epoch", i, "; Batch", batch, "; Loss=", _loss
                    losses.append(_loss)

                    if batch % 100 == 0:
                        print("Epoch", i, "Step", batch, "Loss", _loss)
                        plot_with_loss(losses, _preds, x_batch, y_batch, self.n_units)
                saver.save(sess, "checkpoints/vanilla/model.ckpt", global_step=i)

if __name__ == "__main__":
    echo_step = 2
    series_length = 50000
    num_epochs = 5
    batch_size = 5
    state_size = 4
    n_units = 15
    num_classes = 2

    d = Data(echo_step, series_length)
    t = Train(d, series_length, num_epochs, batch_size, state_size, n_units, num_classes)
    t.train()

