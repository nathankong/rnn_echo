import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def compute_states(inputs, w_state, b_state, batch_size, init_state):
    curr_state = init_state
    all_states = list()
    for i in inputs:
        curr_input = tf.reshape(i, [batch_size, 1])
        curr_input_and_state = tf.concat([curr_state, curr_input], axis=1)
        next_state = tf.tanh(tf.matmul(curr_input_and_state, w_state) + b_state)
        all_states.append(next_state)
        curr_state = next_state
    return all_states

def compute_logits(states, w_out, b_out):
    all_logits = list()
    for state in states:
        curr_logit = tf.matmul(state, w_out) + b_out
        all_logits.append(curr_logit)
    return all_logits

def plot_with_loss(loss_list, predictions_series, batchX, batchY, n_units):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    
    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
    
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, n_units, 0, 2])
        left_offset = range(n_units)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
    
    plt.draw()
    plt.pause(0.0001)

def plot_no_loss(predictions_series, batchX, batchY, n_units, batch_size):
    for batch_series_idx in range(batch_size):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
       
        if batch_size == 1:
            plt.subplot(1, 1, 1)
        else:
            plt.subplot(2, 3, batch_series_idx + 1)
        plt.cla()
        plt.axis([0, n_units, 0, 2]) 
        left_offset = range(n_units)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
       
    plt.show()

