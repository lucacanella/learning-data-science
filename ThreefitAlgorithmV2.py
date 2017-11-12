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
    :ivar n_current_state: Game state encoded as numpy array
    :type current_state: numpy.ndarray
    """
    current_state = None

    """
    :ivar encoded_table_size: The size of the encoded table (input size for the algorithm)
    :type encoded_table_size: int
    """
    encoded_table_size = 36

    last_column_heights = None

    hidden_units_num_1 = 432
    hidden_units_num_2 = 324
    hidden_units_num_3 = 324
    output_units_num = 3

    # decaying learning rate params
    learning_rate = 0.0004
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

    biases_input_weights = None
    biases_l1_weights = None
    biases_l2_weights = None
    biases_output_weights = None

    input_l = None
    hidden_l_1 = None
    hidden_l_2 = None
    output_l = None
    prediction_l = None

    cost = None
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
    print_predictions = True
    print_feedbacks = False
  
    def current_table(self):
        return self.current_tables[-1]

    def init_algorithm(self):
        self.reset_current_state()

        # input placeholder
        self.tf_input = tf.placeholder(tf.float32, [1, self.encoded_table_size])
        # output feedback placeholder
        self.tf_feedback = tf.placeholder(tf.float32, [1, self.output_units_num])

        # init weights
        self.input_l_weights = tf.get_variable("Wi", shape=[self.encoded_table_size, self.hidden_units_num_1],
                                               initializer=tf.contrib.layers.xavier_initializer())
        self.hidden_l_1_weights = tf.get_variable("Wh1", shape=[self.hidden_units_num_1, self.hidden_units_num_2],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.hidden_l_2_weights = tf.get_variable("Wh2", shape=[self.hidden_units_num_2, self.hidden_units_num_3],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.output_l_weights = tf.get_variable("Wo", shape=[self.hidden_units_num_3, self.output_units_num],
                                                initializer=tf.contrib.layers.xavier_initializer())

        # init biases
        self.biases_input_weights = tf.get_variable("bi", shape=[self.hidden_units_num_1],
                                                    initializer=tf.zeros_initializer())
        self.biases_l1_weights = tf.get_variable("bh1", shape=[self.hidden_units_num_2],
                                                 initializer=tf.zeros_initializer())
        self.biases_l2_weights = tf.get_variable("bh2", shape=[self.hidden_units_num_3],
                                                 initializer=tf.zeros_initializer())
        self.biases_output_weights = tf.get_variable("bo", shape=[self.output_units_num],
                                                     initializer=tf.zeros_initializer())

        # init layers (computational graph)
        self.input_l = tf.nn.relu(tf.add(tf.matmul(self.tf_input, self.input_l_weights), self.biases_input_weights))
        self.hidden_l_1 = tf.nn.relu(tf.add(tf.matmul(self.input_l, self.hidden_l_1_weights), self.biases_l1_weights))
        self.hidden_l_2 = tf.nn.relu(
            tf.add(tf.matmul(self.hidden_l_1, self.hidden_l_2_weights), self.biases_l2_weights))
        self.output_l = tf.nn.relu(
            tf.add(tf.matmul(self.hidden_l_2, self.output_l_weights), self.biases_output_weights))

        self.prediction_l = tf.argmax(self.output_l, axis=1, output_type=tf.int32, name="prediction")

        self.decay_global_step = tf.Variable(0, trainable=False)
        self.decaying_learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate,
                                                                 global_step=self.decay_global_step,
                                                                 decay_steps=self.decay_steps,
                                                                 decay_rate=self.decay_rate, staircase=self.staircase)

        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_feedback,logits=self.output_l,dim=-1,name='smce')
        # self.cost = tf.reduce_mean(tf.sqrt(tf.squared_difference(self.tf_feedback, self.output_l, name='sq_diff')), name='cost')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.decaying_learning_rate).minimize(self.cost, global_step=self.decay_global_step)
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

    def reset_game_status(self):
        self.iteration_no = 0
        self.last_column_heights = [0, 0, 0]
        self.reset_current_state()

    def reset_current_state(self):
        self.current_tables = []
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
            return None
        current_state = (np.array(table) / 3.5) - 1
        reshaped = current_state.reshape(-1, self.encoded_table_size)
        return reshaped

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
        current_state = self.encode_state(table)
        if current_state is not None:
            self.current_tables.append(current_state)
            # convert to int because we may use it as an array index.
            prediction, prediction_idx = \
                self.tf_session.run([self.output_l, self.prediction_l], feed_dict={self.tf_input: current_state})
            if self.print_predictions: 
              print('Prediction: %d - ' % prediction_idx, prediction)
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
        if self.print_feedbacks: 
            print("Giving positive feedback (x %d) for %d actions" % (self.repeat_positive_feedbacks, n))
        for i in range(n):
            f = 1 + (n - i) * k
            table = self.current_tables.pop()
            decision = self.latest_predictions_idx.pop()
            prediction = self.latest_predictions.pop()
            _output = np.array([j / f for j in prediction])
            _output[decision] *= f * f
            adjusted_feedback = self.adjust_feedback(_output)
            if self.print_feedbacks: 
                print('%2d) Positive feedback [%.3f] on %d: ' % (i, f, decision), adjusted_feedback)
            for i in range(self.repeat_positive_feedbacks):
                self.feedback(table, adjusted_feedback)

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
        if self.print_feedbacks: 
            print("Giving negative feedback for %d actions" % n)
        for i in range(n):
            f = 1 + (n - i) * k
            table = self.current_tables.pop()
            decision = self.latest_predictions_idx.pop()
            prediction = self.latest_predictions.pop()
            adjusted_decision_weight = prediction[decision] / f
            delta_weight = 0.5 * (prediction[decision] - adjusted_decision_weight)
            _output = np.array([(0.1 + j + delta_weight * f) for j in prediction])
            _output[decision] = adjusted_decision_weight
            adjusted_feedback = self.adjust_feedback(_output)
            if self.print_feedbacks: 
                print('%2d) Negative feedback [%.3f] on %d: ' % (i, f, decision), prediction, _output, adjusted_feedback)
            self.feedback(table, adjusted_feedback)

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

    def feedback(self, table, feedback):
        _o, _cost = self.tf_session.run([self.optimizer, self.cost],
                                        feed_dict={self.tf_input: table, self.tf_feedback: [feedback]})
        # print(algo_in, feedback)
        self.floating_averaged_cost = self.floating_averaged_cost * 0.7 + 0.3 * _cost

    def print_current_state(self):
        print('Current state:')
        cells = math.floor(len(self.current_state) / 3)
        for i in range(cells):
            for j in range(3):
                print('%+2.2f ' % self.current_state[i * 3 + j], end='')
            if i % 3 == 2:
                print()
            else:
                print(' | ', end='')
