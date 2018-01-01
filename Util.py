import tensorflow as tf

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
