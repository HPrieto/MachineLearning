# Dependencies
from deepmusic.moduleloader import ModuleLoader
# Predicts next key
from deepmusic.keyboardcell import KeyboardCell
# Encapsulate song data so we can run get_scale, get_relative_methods
import deepmusic.songstruct as music
import numpy as np  # Generate random numbers
import tensorflow as tf  # For flowing


def build_network(self):
    """
    Create computational graph
    """
    input_dim = ModuleLoader.batch_builders.get_module().get_input_dim()

    # Placeholders (Use tf.SparseTensor with training=False instead)
    # (TODO: Try restoring dynamic batch_size)
    with tf.name_scope('placeholder_inputs'):
        # Inputs to our graph
        self.inputs = [
            tf.placeholder(
                tf.float32,  # -1.0/1.0 ? Probably better for the sigmoid
                [self.args.batch_size, input_dim],  # TODO: Get input size from batch_builder
                name='input'
            )
            for _ in range(self.args.sample_length)
        ]
    with tf.name_scope('placeholder_targets'):
        # Classes/Labels (Whether a piano key is pressed or not)
        self.targets = [
            tf.placeholder(
                tf.int32,  # 0/1 TODO: int for softmax, Float for sigmoid
                # TODO: For softmax, only id, for sigmoid, 2d (batch_size, num_class)
                [self.args.batch_size, ],
                name='target')
            for _ in range(self.args.sample_length)
        ]
    with tf.name_scope('placeholder_use_prev'):
        # Previous Hidden States
        self.use_prev = [
            tf.placeholder(
                tf.bool,  # Should we use the previous value or not
                [],
                name='use_prev')
            for _ in range(self.args.sample_length)  # First value will never be used
            # Always takes self.input for the first step
        ]

    # Define the network
    # Manually create a loop
    self.loop_processing = ModuleLoader.loop_processing.build_module(self.args)

    def loop_rnn(prev, i):
        """
            Loop function used to connect one output of the rnn to the next input,
            The previous input and returned value have to be from the same shape.
            This is useful to use the same network for both training and testing
            Args:
                prev: the previoud predicted keyboard configuration at step i-1
                i: the current step id (Warning: start at 1, 0 is ignored)
            Return:
                tf.Tensor: the input at the step i
        """
        next_input = self.loop_processing(prev)

        # On training, we force the correct input, on testing, we use the previous
        # output as next input
        # Returns one or the other on parameters of condition tf statement
        return tf.cond(self.use_prev[i], lambda: next_input, lambda: self.inputs[i])

    # Build sequence to sequence model
    # TODO: Try attention decoder/ use dynamic_rnn instead
    self.outputs, self.final_state = tf.nn.seq2seq.rnn_decoder(
        decoder_inputs=self.inputs,
        initial_state=None,  # The initial state is defined inside KeyboardCell
        cell=KeyboardCell(self.args),
        loop_function=loop_rnn
    )

    # For training only
    if not self.args.test:
        # Finally, we define the loss function

        # The network will predict a mix a wrong and right noes.
        # For the loss function, we would like to penalize note which
        # are wrong. Eventually, the penalty should be less if the network
        # predict the same note but not in the right pitch
        # (ex: C4 instead of C5), with a decay the further the prediction
        # is (D5 and D1 more penalized than D4 and D3 if the target is D2)

        # For the piano roll mode, by using sigmoid_cross_entropy_with_logits,
        # The task is formulates as a NB_NOTES binary classification problems

        # For the relative note experiment, it use a standard SoftMax where the
        # label is the relative position to the previous note

        self.schedule_policy = Model.ScheduledSamplingPolicy(self.args)
        self.target_weights_policy = Model.TargetWeightsPolicy(self.args)
        self.learning_rate_policy = ModuleLoader.learning_rate_policies.build_module(
            self.args)  # Load the chosed policies

        # TODO: If train on different length, check that the loss is proportional
        # to the length or average ???
        # Minimize the difference between two notes/predicted outputs
        loss_fct = tf.nn.seq2seq.sequence_loss(
            self.outputs,
            self.targets,
            [tf.constant(self.target_weights_policy.get_weight(  # Weights
                i), shape=self.targets[0].get_shape()) for i in range(len(self.targets))],
            # softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits,
            # Previous: tf.nn.sigmoid_cross_entropy_with_logits
            # TODO: Use option ot choose. (new module ?)
            average_across_timesteps=True,  # Before: I think it's best for variables length
            # Sequences (specially with target weights = 0)
            # Before: Penalize by sample (should alows dynamic batch size)
            # Warning: Need to tune the learning rate
            average_across_batch=True
        )
        tf.scalar_summary('training_loss', loss_fct)  # Keep track of the cost

        self.current_learning_rate = tf.placeholder(tf.float32, [])

        # Initialize the optimizer
        opt = tf.train.AdamOptimizer(
            learning_rate=self.current_learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )

        # TODO: Also keep track of magnitudes (how much is updated)
        self.opt_op = opt.minimize(loss_fct)
