import tensorflow as tf
import matplotlib.pyplot as plt

from Data import *
from Model_Rnn import *

class Test:
    def __init__(self, ckpt, data, state_size, num_classes, batch_size, n_units, series_length):
        assert isinstance(data, Data)
        self.data = data
        self.state_size = state_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.n_units = n_units
        self.num_batches = series_length // batch_size // n_units

    def loss_func(self, all_logits, outputs):
        assert len(all_logits) == len(outputs)
        losses = list()
        for logits, labels in zip(all_logits, outputs):
            curr_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            losses.append(curr_loss)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    def test(self):
        model = Model_Rnn(self.batch_size, self.n_units, self.state_size, self.num_classes)
        logits, outputs = model.model_func()
        total_loss = self.loss_func(logits, outputs)
        with tf.Session() as sess:
            saver = tf.train.Saver()
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
                _loss, _preds = sess.run([total_loss, model.all_predictions], \
                                   feed_dict={model.x_batch:x_batch,
                                              model.y_batch:y_batch,
                                              model.init_state:curr_state})
                plot(_preds, x_batch, y_batch, self.n_units, self.batch_size)
                print "Batch: " + str(batch) + "; Loss: " + str(_loss)

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
    ckpt = "checkpoints/rnn_cell/model.ckpt-4"
    echo_step = 2
    series_length = 100
    state_size = 4
    num_classes = 2
    batch_size = 5
    n_units = 15

    d = Data(echo_step, series_length)
    t = Test(ckpt, d, state_size, num_classes, batch_size, n_units, series_length)
    t.test()

