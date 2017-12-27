import glob
import math
import os
import numpy as np
import tensorflow as tf


class ThreefitAlgorithmV2():
    """
    Threefit playing algorithm.
    The implementation of the algorithm behind the ThreefitGameAgent. Second version.
    """

    """
    :ivar current_tables: current game table (last fed)
    :type current_tables: list[list[int]]
    """
    current_tables = None

    """
    :ivar algo_inputs:
    :type algo_inputs: list[list[int]]
    """
    algo_inputs = None

    """
    :ivar n_current_state: Game state encoded as numpy array
    :type current_state: numpy.ndarray
    """
    current_state = None

    """
    :ivar encoded_table_size: The size of the encoded table (input size for the algorithm)
    :type encoded_table_size: int
    """
    encoded_table_size = 36
    complete_input_size = encoded_table_size * 2
    zeroes_table = np.array([0.0] * encoded_table_size)

    last_column_heights = None

    hidden_units_num_1 = 584
    hidden_units_num_2 = 584
    hidden_units_num_3 = 584
    output_units_num = 3

    # decaying learning rate params
    learning_rate = 0.005
    decay_steps = 5000
    decay_rate = 0.96
    staircase = True
    repeat_positive_feedbacks = 3

    decay_global_step = None
    decaying_learning_rate = None

    tf_input = None
    tf_feedback = None

    input_l_weights = None
    hidden_l_1_weights = None
    hidden_l_2_weights = None
    output_l_weights = None

    biases_input = None
    biases_l1 = None
    biases_l2 = None
    biases_output = None

    input_l = None
    hidden_l_1 = None
    hidden_l_2 = None
    output_l = None
    output_l_softmax = None
    prediction_l = None

    cost_coeff = None
    cost = None
    reward_or_cost = None
    optimizer = None
    tf_init = None
    tf_session = None

    # tf_predict = None

    iteration_no = 0

    latest_predictions = None
    latest_predictions_idx = None

    """
    :ivar column_heights: Calculated height of each column, starting from the botton until the first empty cell.
    :type columne_heights: list[int]
    """
    column_heights = None

    floating_averaged_cost = 0

    model_file_dir = "./model"
    model_file_name = "./model/model.ckpt"
    tf_saver = None

    enable_restore_model = True
    enable_save_model = True
    print_predictions = False
    print_feedback = False
    print_game_state = False

    def current_table(self):
        return self.current_tables[-1]

    def init_algorithm(self):
        self.reset_current_state()

        # input placeholder
        self.tf_input = tf.placeholder(tf.float32, [1, self.complete_input_size])
        # output feedback placeholder
        self.tf_feedback = tf.placeholder(tf.float32, [1, self.output_units_num])

        # init weights
        self.input_l_weights = tf.get_variable("Wi", shape=[self.complete_input_size, self.hidden_units_num_1],
                                               initializer=tf.truncated_normal_initializer())
        self.hidden_l_1_weights = tf.get_variable("Wh1", shape=[self.hidden_units_num_1, self.hidden_units_num_2],
                                                  initializer=tf.truncated_normal_initializer())
        self.hidden_l_2_weights = tf.get_variable("Wh2", shape=[self.hidden_units_num_2, self.hidden_units_num_3],
                                                  initializer=tf.truncated_normal_initializer())
        self.output_l_weights = tf.get_variable("Wo", shape=[self.hidden_units_num_3, self.output_units_num],
                                                initializer=tf.truncated_normal_initializer())

        # init biases
        self.biases_input = tf.get_variable("bi", shape=[self.hidden_units_num_1],
                                                    initializer=tf.zeros_initializer())
        self.biases_l1 = tf.get_variable("bh1", shape=[self.hidden_units_num_2],
                                                 initializer=tf.zeros_initializer())
        self.biases_l2 = tf.get_variable("bh2", shape=[self.hidden_units_num_3],
                                                 initializer=tf.zeros_initializer())
        self.biases_output = tf.get_variable("bo", shape=[self.output_units_num],
                                                     initializer=tf.zeros_initializer())

        # init layers (computational graph)
        self.input_l = tf.nn.relu(tf.add(
            tf.matmul(self.tf_input, self.input_l_weights), self.biases_input), name="input_l")
        self.hidden_l_1 = tf.nn.relu(tf.add(
            tf.matmul(self.input_l, self.hidden_l_1_weights), self.biases_l1), name="hidden_l_1")
        self.hidden_l_2 = tf.nn.relu(
            tf.add(tf.matmul(self.hidden_l_1, self.hidden_l_2_weights), self.biases_l2), name="hidden_l_2")
        self.output_l = tf.add(tf.matmul(self.hidden_l_2, self.output_l_weights), self.biases_output, name="output_l")
        self.output_l_softmax = tf.nn.softmax(self.output_l, name="output_softmax")

        self.prediction_l = tf.argmax(self.output_l_softmax, axis=1, output_type=tf.int32, name="prediction")

        self.decay_global_step = tf.Variable(0, trainable=False)
        self.decaying_learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                 global_step=self.decay_global_step,
                                                                 decay_steps=self.decay_steps,
                                                                 decay_rate=self.decay_rate, staircase=self.staircase)

        self.cost_coeff = tf.placeholder(dtype=tf.float32,shape=(),name="cost_coeff")
        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_feedback,logits=self.output_l,dim=-1,name='smce')
        self.reward_or_cost = tf.multiply(self.cost_coeff, self.cost, name="reward_or_cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decaying_learning_rate).minimize(
            self.reward_or_cost, global_step=self.decay_global_step)
        self.tf_init = tf.global_variables_initializer()

        self.tf_session = tf.Session()
        self.tf_session.run(self.tf_init)

        self.iteration_no = 0

        self.last_column_heights = [0, 0, 0]

        self.floating_averaged_cost = 0
        if self.enable_restore_model or self.enable_save_model:
            self.tf_saver = tf.train.Saver()

        if not os.path.exists(self.model_file_dir):
            os.makedirs(self.model_file_dir)

        if self.enable_restore_model:
            if len(glob.glob(self.model_file_name + '*')):
                self.tf_saver.restore(self.tf_session, self.model_file_name)
                print('TF Session restored.')
            else:
                print('NO TF Session to restore, starting a new model.')

    def save_model(self):
        if self.enable_save_model:
            saved_model_file_path = self.tf_saver.save(self.tf_session, self.model_file_name)
            print('TF Session saved @ ', saved_model_file_path)

    def close_session(self):
        self.tf_session.close()

    def reset_game_status(self):
        self.iteration_no = 0
        self.last_column_heights = [0, 0, 0]
        self.reset_current_state()

    def reset_current_state(self):
        self.current_tables = []
        self.algo_inputs = []
        self.latest_predictions = []
        self.latest_predictions_idx = []

    def get_current_learning_rate(self):
        return self.tf_session.run(self.decaying_learning_rate)
        
    def encode_state(self, table):
        """
        Encodes newly acquired game table.
        :param table: 
        :type table: list[int]
        """
        if len(table) != self.encoded_table_size:
            return None, None
        current_state = np.array(table) / 7.0
        reshaped = current_state.reshape(-1, self.encoded_table_size)
        if self.print_predictions:
          print ("Current table stack size: ", len(self.current_tables))
        if len(self.current_tables) > 0:
            last_encoded_table = self.current_tables[-1]
        else:
            last_encoded_table = reshaped
        diff_state = np.maximum(self.zeroes_table, current_state - last_encoded_table)
        if self.print_game_state:
            ThreefitAlgorithmV2.print_current_state(reshaped, diff_state)
        return reshaped, diff_state

    def last_prediction(self):
        if self.latest_predictions is not None and len(self.latest_predictions) > 0:
            return self.latest_predictions[-1]
        else:
            return None

    def feedback_for_last_action(self, table):
        if table is None:
            # game ended: negative feedback and reset status
            self.negative_feedback_for_latest_actions()
            self.reset_game_status()
            return
        columns_heights = self.calculate_column_heights_delta(table)
        delta = sum(columns_heights) - sum(self.last_column_heights)
        self.last_column_heights = columns_heights
        if self.iteration_no > 1 and self.last_prediction() is not None:
            # print('delta: %d - ' % delta, end='')
            if delta > 0:
                self.negative_feedback_for_latest_actions()
                self.reset_current_state()
            elif delta < 0:
                self.positive_feedback_for_latest_actions(delta)
                self.reset_current_state()

    def iterate(self, table):
        current_state, diff_state = self.encode_state(table)
        if current_state is not None:
            self.current_tables.append(current_state)
            complete_state = np.concatenate([ current_state, diff_state ], axis=1)
            self.algo_inputs.append(complete_state)
            prediction, output_l_v, prediction_idx = \
                self.tf_session.run([self.output_l_softmax, self.output_l, self.prediction_l], feed_dict={self.tf_input: complete_state})
            if self.print_predictions: 
              print('Prediction: %d - ' % prediction_idx, prediction, output_l_v)
            self.iteration_no += 1
            self.latest_predictions.append(prediction[0])
            int_prediction_idx = int(prediction_idx)
            self.latest_predictions_idx.append(int_prediction_idx)
            return int_prediction_idx
        else:
            return None

    @staticmethod
    def adjust_feedback(feedback:np.array):
        tsum = feedback.sum()
        if tsum == 0:
            tsum = 1.0
        return feedback / tsum

    def positive_feedback_for_latest_actions(self, delta):
        """
        Last action had a positive outcome: encourage the algorithm to keep it up
        :return feedback: list[float]
        """
        n = len(self.current_tables)
        if n < 1:
            return
        k = 1 / n
        if self.print_feedback:
            print("Giving positive feedback (x %d) for %d actions" % (self.repeat_positive_feedbacks, n))
        for i in range(n):
            f = 1 + (n - i) * k
            self.current_tables.pop()
            current_full_input = self.algo_inputs.pop()
            decision = self.latest_predictions_idx.pop()
            prediction = self.latest_predictions.pop()
            _output = np.array([j / f for j in prediction])
            _output[decision] *= f * f
            adjusted_feedback = self.adjust_feedback(_output)
            if self.print_feedback:
                print('%2d) Positive feedback [%.3f] on %d: ' % (i, f, decision), adjusted_feedback)
            for i in range(self.repeat_positive_feedbacks):
                self.feedback(current_full_input, adjusted_feedback, -1.0)

    def negative_feedback_for_latest_actions(self):
        """
        Last action had a negative outcome: punish the algorithm for that decision and tell it that it should have
        choosen another action (with a 50% probability)
        :return feedback: list[float]
        """
        n = len(self.current_tables)
        if n < 1:
            return
        k = 1 / n
        if self.print_feedback:
            print("Giving negative feedback for %d actions" % n)
        for i in range(n):
            f = 1 + (n - i) * k
            self.current_tables.pop()
            complete_input = self.algo_inputs.pop()
            decision = self.latest_predictions_idx.pop()
            prediction = self.latest_predictions.pop()
            adjusted_decision_weight = prediction[decision] / f
            delta_weight = 0.5 * (prediction[decision] - adjusted_decision_weight)
            _output = np.array([(0.1 + j + delta_weight * f) for j in prediction])
            _output[decision] = adjusted_decision_weight
            adjusted_feedback = self.adjust_feedback(_output)
            if self.print_feedback:
                print('%2d) Negative feedback [%.3f] on %d: ' % (i, f, decision), prediction, _output, adjusted_feedback)
            self.feedback(complete_input, adjusted_feedback, 1.0)

    @staticmethod
    def calculate_column_heights_delta(table):
        heights = [0, 0, 0, ]
        rows = math.floor(len(table) / 3)
        reverse_table = list(reversed(table))
        keep_going = [1, 1, 1]
        for i in range(rows):
            for j in range(3):
                cell_index = i * 3 + j
                if keep_going[j] and reverse_table[cell_index] != 0:
                    heights[j] += 1
                else:
                    keep_going[j] = 0
            if sum(keep_going) < 1:
                break
        return heights

    def feedback(self, table, feedback, cost_k):
        _reward, _cost, _o = self.tf_session.run([self.reward_or_cost, self.cost, self.optimizer],
                                        feed_dict={self.tf_input: table,
                                                   self.cost_coeff: cost_k,
                                                   self.tf_feedback: [feedback]})
        #print(cost_k, _reward, _cost, feedback)
        self.floating_averaged_cost = self.floating_averaged_cost * 0.7 + 0.3 * _cost

    @staticmethod
    def print_state_row(row_index, state_to_print, line_end):
        for j in range(3):
            cell = state_to_print[row_index * 3 + j]
            print('%+2.2f ' % cell, end='')
        print(line_end, end = line_end)

    @staticmethod
    def print_current_state(current_state, diff_state):
        # print('States shape: ', current_state.shape, diff_state.shape)
        if len(current_state) < 1 or len(diff_state) < 1:
            # print('empty state.')
            return
        print('|--------------------------------------------|')
        print('|---- Current state--|--|--- Diff. state ----|')
        rows = math.floor(len(current_state[0]) / 3)
        current_state_decoded = current_state * 7.0
        diff_state_decoded = diff_state * 7.0
        for i in range(rows):
            print('| ', end ='')
            ThreefitAlgorithmV2.print_state_row(i, current_state_decoded[0], ' | ')
            ThreefitAlgorithmV2.print_state_row(i, diff_state_decoded[0], '')
            print(' |')
        print('|--------------------------------------------|')
