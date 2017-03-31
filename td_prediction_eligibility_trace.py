import tensorflow as tf
import os

FEATURE_NUMBER = 12

model_path = os.environ.get('MODEL_PATH', 'et_models/')
summary_path = os.environ.get('SUMMARY_PATH', 'et_summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'et_checkpoints/')


# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b


# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')


class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay
        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, 30000, 0.96, staircase=True),
                           name='lambda')

        # learning rate decay
        alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step,
                                                            40000, 0.96, staircase=True), name='alpha')

        tf.summary.scalar('lambda', lamda)
        tf.summary.scalar('alpha', alpha)

        # describe network size
        layer_size_input = FEATURE_NUMBER
        layer_size_hidden = 1000
        layer_size_output = 1

        # placeholders for input and target output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')

        # build network arch. (just 2 layers with sigmoid activation)
        prev_y = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, [layer_size_hidden, layer_size_output], tf.sigmoid, name='layer2')

        # watch the individual value predictions over time
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))

        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')

        # mean squared error of the difference between the next state and the current state
        loss_op = tf.reduce_mean(tf.square(self.V_next - self.V), name='loss')

        # check if the model predicts the correct state
        accuracy_op = tf.reduce_sum(tf.cast(tf.equal(tf.round(self.V_next), tf.round(self.V)), dtype='float'),
                                    name='accuracy')

        # track the number of steps and average loss for the current game
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)

            loss_sum = tf.Variable(tf.constant(0.0), name='loss_sum', trainable=False)
            delta_sum = tf.Variable(tf.constant(0.0), name='delta_sum', trainable=False)
            accuracy_sum = tf.Variable(tf.constant(0.0), name='accuracy_sum', trainable=False)

            loss_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            delta_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)
            accuracy_avg_ema = tf.train.ExponentialMovingAverage(decay=0.999)

            loss_sum_op = loss_sum.assign_add(loss_op)
            delta_sum_op = delta_sum.assign_add(delta_op)
            accuracy_sum_op = accuracy_sum.assign_add(accuracy_op)

            loss_avg_op = tf.div(loss_sum, tf.maximum(game_step, 1.0))
            delta_avg_op = tf.div(delta_sum, tf.maximum(game_step, 1.0))
            accuracy_avg_op = tf.div(accuracy_sum, tf.maximum(game_step, 1.0))

            loss_avg_ema_op = loss_avg_ema.apply([loss_avg_op])
            delta_avg_ema_op = delta_avg_ema.apply([delta_avg_op])
            accuracy_avg_ema_op = accuracy_avg_ema.apply([accuracy_avg_op])

            tf.summary.scalar('game/loss_avg', loss_avg_op)
            tf.summary.scalar('game/delta_avg', delta_avg_op)
            tf.summary.scalar('game/accuracy_avg', accuracy_avg_op)
            tf.summary.scalar('game/loss_avg_ema', loss_avg_ema.average(loss_avg_op))
            tf.summary.scalar('game/delta_avg_ema', delta_avg_ema.average(delta_avg_op))
            tf.summary.scalar('game/accuracy_avg_ema', accuracy_avg_ema.average(accuracy_avg_op))

            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            loss_sum_reset_op = loss_sum.assign(0.0)
            self.reset_op = tf.group(*[loss_sum_reset_op, game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V wrt trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)

        # watch the weight and gradient distributions
        for grad, var in zip(grads, tvars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/gradients/grad', grad)

        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)
                    # tf.summary.scalar(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op
                # tf.summary.scalar(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step and other monitoring variables
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
            loss_sum_op,
            delta_sum_op,
            accuracy_sum_op,
            loss_avg_ema_op,
            delta_avg_ema_op,
            accuracy_avg_ema_op
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.initialize_all_variables())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()

if __name__ == '__main__':
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        model = Model(sess, model_path, summary_path, checkpoint_path)