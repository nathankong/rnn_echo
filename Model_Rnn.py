import tensorflow as tf

class Model_Rnn:
    def __init__(self, batch_size, n_units, state_size, num_classes):
        self.batch_size = batch_size
        self.n_units = n_units
        self.state_size = state_size
        self.num_classes = num_classes

        self.x_batch = None
        self.y_batch = None
        self.init_state = None
        self.all_predictions = None

    def model_func(self):
        self.x_batch = tf.placeholder(tf.float32, [self.batch_size, self.n_units])
        self.y_batch = tf.placeholder(tf.int32, [self.batch_size, self.n_units])
        self.init_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])

        w_out = tf.get_variable("w_out", initializer=tf.contrib.layers.xavier_initializer(), shape=[self.state_size, self.num_classes], dtype=tf.float32)
        b_out = tf.get_variable("b_out", initializer=tf.contrib.layers.xavier_initializer(), shape=[1, self.num_classes], dtype=tf.float32)

        inputs = tf.unstack(self.x_batch, axis=1)
        outputs = tf.unstack(self.y_batch, axis=1)

        # RNN code here
        all_logits = list()
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.state_size)
        state = self.init_state
        for i in range(self.n_units):
            curr_input = tf.reshape(inputs[i], [self.batch_size, 1])
            output, state = rnn_cell(curr_input, state)
            logit = tf.matmul(output, w_out) + b_out
            all_logits.append(logit)

        self.all_predictions = [tf.nn.softmax(logits) for logits in all_logits]
        return all_logits, outputs

