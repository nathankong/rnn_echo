import tensorflow as tf
import matplotlib.pyplot as plt

from Data import *
from Util import *

class Test:
    def __init__(self, ckpt, data, state_size, num_classes, batch_size, n_units, series_length):
        assert isinstance(data, Data)
        self.data = data
        self.state_size = state_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.n_units = n_units
        self.num_batches = series_length // batch_size // n_units
        self.model()

    def model(self):
        self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.n_units])
        self.y_batch = tf.placeholder(tf.int32, [self.batch_size, self.n_units])
        self.init_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])

        w_state = tf.get_variable("w_state", initializer=tf.random_uniform([self.state_size+1, self.state_size]), dtype=tf.float32)
        b_state = tf.get_variable("b_state", initializer=tf.zeros_initializer([1, self.state_size]), shape=[self.state_size], dtype=tf.float32)
        w_out = tf.get_variable("w_out", initializer=tf.random_uniform([self.state_size, self.num_classes]), dtype=tf.float32)
        b_out = tf.get_variable("b_out", initializer=tf.zeros_initializer([1, self.num_classes]), shape=[self.num_classes], dtype=tf.float32)

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
        self.total_loss = tf.reduce_mean(losses)

    def test(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            data = self.data.get_data()
            inputs = data[0]
            outputs = data[1]
            inputs = np.reshape(inputs, [self.batch_size, -1])
            outputs = np.reshape(outputs, [self.batch_size, -1])
            curr_state = np.zeros((self.batch_size, self.state_size))
            for batch in xrange(self.num_batches):
                start_idx = batch * self.n_units
                end_idx = start_idx + self.n_units
                x_batch = inputs[:, start_idx:end_idx]
                y_batch = outputs[:, start_idx:end_idx]
                _loss, _preds = sess.run([self.total_loss, self.all_predictions], \
                                   feed_dict={self.x_batch:x_batch,
                                              self.y_batch:y_batch,
                                              self.init_state:curr_state})
                plot(_preds, x_batch, y_batch, self.n_units, self.batch_size)
                print "Batch:", batch, "; Loss:", _loss

def plot(predictions_series, batchX, batchY, n_units, batch_size):
    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
       
        plt.subplot(2, 3, batch_series_idx + 1)
        plt.cla()
        plt.axis([0, n_units, 0, 2]) 
        left_offset = range(n_units)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
       
    plt.show()

if __name__ == "__main__":
    ckpt = "checkpoints/model.ckpt-9"
    echo_step = 2
    series_length = 100
    state_size = 4
    num_classes = 2
    batch_size = 5
    n_units = 15

    d = Data(echo_step, series_length)
    t = Test(ckpt, d, state_size, num_classes, batch_size, n_units, series_length)
    t.test()


