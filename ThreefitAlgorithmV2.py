import glob
import math
import os
import numpy as np
import tensorflow as tf

""" Threefit playing algorithm """
class ThreefitAlgorithmV2():
    """
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
    encoded_table_size = None

    learning_rate = 0.001

    hidden_units_num_1 = 72
    hidden_units_num_2 = 36
    hidden_units_num_3 = 36
    output_units_num   = 3

    input_l_weights    = None
    hidden_l_1_weights = None
    output_l_weights  = None

    biases_input_weights  = None
    biases_l1_weights     = None
    biases_output_weights = None

    input_l    = None
    hidden_l_1 = None
    output_l   = None

    cost = None
    optimizer = None
    tf_init = None
    tf_session = None

    #tf_predict = None

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

    def current_table(self):
        return self.current_tables[-1]

    def init_algorithm(self):
        self.reset_current_state()
        # each state counts 36 cells encoded with 3 integers (so 36 * 3)
        self.encoded_table_size = 3 * 36

        #input placeholder
        self.tf_input = tf.placeholder(tf.float32, [1, self.encoded_table_size])
        #output feedback placeholder
        self.tf_feedback = tf.placeholder(tf.float32, [1, self.output_units_num])

        #init weights
        self.input_l_weights    = tf.Variable(tf.random_normal([self.encoded_table_size, self.hidden_units_num_1]))
        self.hidden_l_1_weights = tf.Variable(tf.random_normal([self.hidden_units_num_1, self.hidden_units_num_2]))
        self.hidden_l_2_weights = tf.Variable(tf.random_normal([self.hidden_units_num_2, self.hidden_units_num_3]))
        self.output_l_weights   = tf.Variable(tf.random_normal([self.hidden_units_num_3, self.output_units_num]))

        #init biases
        self.biases_input_weights = tf.Variable(tf.random_normal([self.hidden_units_num_1]))
        self.biases_l1_weights    = tf.Variable(tf.random_normal([self.hidden_units_num_2]))
        self.biases_l2_weights    = tf.Variable(tf.random_normal([self.hidden_units_num_3]))
        self.biases_output_weights = tf.Variable(tf.random_normal([self.output_units_num]))

        #init layers (computational graph)
        self.input_l    = tf.nn.sigmoid(tf.add(tf.matmul(self.tf_input, self.input_l_weights), self.biases_input_weights))
        self.hidden_l_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.input_l, self.hidden_l_1_weights), self.biases_l1_weights))
        self.hidden_l_2 = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_l_1, self.hidden_l_2_weights), self.biases_l2_weights))
        self.output_l = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_l_2, self.output_l_weights), self.biases_output_weights))

        self.cost = tf.reduce_mean(tf.square(self.tf_feedback - self.output_l))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.tf_init = tf.global_variables_initializer()

        self.tf_session = tf.Session()
        self.tf_session.run(self.tf_init)

        self.iteration_no = 0

        self.last_column_heights = [ 0, 0, 0 ]

        self.floating_averaged_cost = 0
        self.tf_saver = tf.train.Saver()

        if not os.path.exists(self.model_file_dir):
            os.makedirs(self.model_file_dir)

        if len(glob.glob(self.model_file_name+'*')):
            self.tf_saver.restore(self.tf_session, self.model_file_name)
            print('TF Session restored.')
        else:
            print('NO TF Session to restore, starting a new model.')

    def save_model(self):
        saved_model_file_path = self.tf_saver.save(self.tf_session, self.model_file_name)
        print('TF Session saved @ ', saved_model_file_path)

    def reset_game_status(self):
        self.iteration_no = 0
        self.last_column_heights = [ 0, 0, 0 ]
        self.reset_current_state()

    def reset_current_state(self):
        self.current_tables = []
        self.latest_predictions = []
        self.latest_predictions_idx = []

    def encode_state(self, table):
        """
        Encodes newly acquired game table.
        :param table: list[int]
        """
        state_index = 0
        current_state = np.zeros((self.encoded_table_size))
        for cell in table:
            # encode the cell value in a binary like flavour where every cell is exploded into trhee values.
            # The decimal value of the cell is encoded in three binary variables expanded to the [-1; 1] interval:
            # 0: -1, -1, -1
            # 1: -1, -1,  1
            # 2: -1,  1,  1
            # ...
            # 7:  1,  1,  1
            # First: binary encoding of the cell
            asbinary = [ math.floor(cell / 4) % 2, math.floor(cell / 2) % 2, cell % 2 ]
            # Second: shift and scale into [ -1; 1 ] interval
            encoded  = [ ( x * 2 ) - 1 for x in asbinary ]
            for i in range(3):
                current_state[state_index+i] = encoded[i]
            state_index += 3
        return current_state.reshape(-1, self.encoded_table_size)

    def last_prediction(self):
        if self.latest_predictions is not None and len(self.latest_predictions) > 0:
            return self.latest_predictions[-1]
        else:
            return None

    def feedback_for_last_action(self, table):
        if table is None:
            #game ended: negative feedback and reset status
            self.negative_feedback_for_latest_actions()
            self.reset_game_status()
            return
        columns_heights = self.calculate_column_heights_delta(table)
        delta = sum(columns_heights) - sum(self.last_column_heights)
        self.last_column_heights = columns_heights
        if self.iteration_no > 1 and self.last_prediction() is not None:
            #print('delta: %d - ' % delta, end='')
            if delta > 0:
                self.negative_feedback_for_latest_actions()
                self.reset_current_state()
            elif delta < 0:
                self.positive_feedback_for_latest_actions(delta)
                self.reset_current_state()

    def iterate(self, table):
        current_state = self.encode_state(table)
        self.current_tables.append(current_state)
        # convert to int because we may use it as an array index.
        prediction = self.tf_session.run(self.output_l, feed_dict={self.tf_input: current_state})
        prediction_idx = np.argmax(prediction)
        #print('Prediction: %d - ' % prediction_idx, prediction)
        self.iteration_no += 1
        self.latest_predictions.append(prediction[0])
        self.latest_predictions_idx.append(prediction_idx)
        return prediction_idx

    def adjust_feedback(self, feedback):
        tsum = sum(feedback) * 0.5
        if tsum == 0:
            tsum = 0.5
        return [v/tsum for v in feedback]

    def positive_feedback_for_latest_actions(self, delta):
        """
        Last action had a positive outcome: encourage the algorithm to keep it up
        :return feedback: list[float]
        """
        n = len(self.current_tables)
        if n < 1:
            return
        k = 1 / n
        #print("Giving positive feedback for %d actions" % n)
        for i in range(n):
            f = 1 + (n - i) * k
            table = self.current_tables.pop()
            decision = self.latest_predictions_idx.pop()
            prediction = self.latest_predictions.pop()
            _output = [ j / f for j in prediction ]
            _output[decision] *= f * f
            adjusted_feedback = self.adjust_feedback(_output)
            #print('%2d) Positive feedback [%.3f] on %d: ' % (i, f, decision), adjusted_feedback)
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
        #print("Giving negative feedback for %d actions" % n)
        for i in range(n):
            f = 1 + (n - i) * k
            table = self.current_tables.pop()
            decision = self.latest_predictions_idx.pop()
            prediction = self.latest_predictions.pop()
            _output = [j * f for j in prediction]
            _output[decision] *= 1 / (f * f)
            adjusted_feedback = self.adjust_feedback(_output)
            #print('%2d) Negative feedback [%.3f] on %d: ' % (i, 1/f, decision), adjusted_feedback)
            self.feedback(table, adjusted_feedback)

    def calculate_column_heights_delta(self, table):
        heights = [ 0, 0, 0, ]
        rows = math.floor(len(table) / 3)
        reverse_table = list(reversed(table))
        keep_going = [ 1, 1, 1 ]
        for i in range(rows):
            for j in range(3):
                cell_index = i*3+j
                if keep_going[j] and reverse_table[cell_index] != 0:
                    heights[j] += 1
                else:
                    keep_going[j] = 0
            if sum(keep_going) < 1:
                break
        return heights

    def feedback(self, table, feedback):
        algo_in = table.reshape(-1, self.encoded_table_size)
        _o, _cost = self.tf_session.run([self.optimizer, self.cost], feed_dict={self.tf_input: algo_in, self.tf_feedback: [feedback]})
        self.floating_averaged_cost = self.floating_averaged_cost * 0.7 + 0.3 * _cost

    def print_current_state(self):
        print ('Current state:')
        cells = math.floor(len(self.current_state) / 3)
        for i in range(cells):
            for j in range(3):
                print('%+2.2f ' % self.current_state[i*3+j], end='')
            if i%3 == 2:
                print()
            else:
                print(' | ', end='')
