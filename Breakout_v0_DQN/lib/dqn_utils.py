import numpy as np
import tensorflow as tf

VALID_ACTIONS = [0, 1, 2, 3]


class StateProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, {self.input_state: state})


def state_process(sess, state_processor, state):
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    return state


def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def populate_replay_buffer(sess, env, state_processor, replay_memory_init_size, VALID_ACTIONS, Transition, policy):
    replay_memory = []
    state = env.reset()
    state = state_process(sess, state_processor, state)
    for i in range(replay_memory_init_size):
        action = np.random.choice(len(VALID_ACTIONS), p=policy(sess, state, 1))
        next_state, reward, done, _ = env.step(action)
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done:
            state = env.reset()
            state = state_process(sess, state_processor, state)

        state = next_state

    return replay_memory















