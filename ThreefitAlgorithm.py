import math
import numpy as np
import tensorflow as tf

""" Threefit playing algorithm """
class ThreefitAlgorithm():
    """
    The implementation of the algorithm behind the ThreefitGameAgent.
    """

    """
    :ivar current_table: current game table (last fed)
    :type current_table: list[int]
    """
    current_table = None

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

    learning_rate = 0.02

    hidden_units_num_1 = 36
    hidden_units_num_2 = 10
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

    tf_predict = None

    iteration_no = 0

    last_prediction = None

    """
    :ivar column_heights: Calculated height of each column, starting from the botton until the first empty cell.
    :type columne_heights: list[int]
    """
    column_heights = None

    floating_averaged_cost = 0

    def init_algorithm(self):
        # each state counts 36 cells encoded with 3 integers (so 36 * 3)
        self.encoded_table_size = 3 * 36
        self.current_state = np.zeros((self.encoded_table_size))

        #input placeholder
        self.tf_input = tf.placeholder(tf.float32, [1, self.encoded_table_size])
        #output feedback placeholder
        self.tf_feedback = tf.placeholder(tf.float32, [1, self.output_units_num])

        #init weights
        self.input_l_weights    = tf.Variable(tf.random_normal([self.encoded_table_size, self.hidden_units_num_1]))
        self.hidden_l_1_weights = tf.Variable(tf.random_normal([self.hidden_units_num_1, self.hidden_units_num_2]))
        self.output_l_weights   = tf.Variable(tf.random_normal([self.hidden_units_num_2, self.output_units_num]))

        #init biases
        self.biases_input_weights = tf.Variable(tf.random_normal([self.hidden_units_num_1]))
        self.biases_l1_weights = tf.Variable(tf.random_normal([self.hidden_units_num_2]))
        self.biases_output_weights = tf.Variable(tf.random_normal([self.output_units_num]))

        #init layers (computational graph)
        self.input_l    = tf.nn.relu(tf.add(tf.matmul(self.tf_input, self.input_l_weights), self.biases_input_weights))
        self.hidden_l_1 = tf.nn.relu(tf.add(tf.matmul(self.input_l, self.hidden_l_1_weights), self.biases_l1_weights))
        self.output_l = tf.nn.sigmoid(tf.add(tf.matmul(self.hidden_l_1, self.output_l_weights), self.biases_output_weights))

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_l, labels=self.tf_feedback))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.tf_init = tf.global_variables_initializer()

        self.tf_session = tf.Session()
        self.tf_session.run(self.tf_init)
        self.tf_predict = tf.arg_max(self.output_l, 1)

        self.iteration_no = 0

        self.last_column_heights = [ 0, 0, 0 ]

        self.floating_averaged_cost = 0

    def feed_table(self, table):
        """
        Updates status by feed a newly acquired game table.
        :param table: list[int]
        """
        state_index = 0
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
                self.current_state[state_index+i] = encoded[i]
            state_index += 3
        self.current_table = table

    def iterate_for_next_action(self):
        if self.iteration_no > 0:
            columns_delta = self.calculate_column_heights_delta()
            if columns_delta < 1:
                feedback = self.positive_feedback_for_last_action()
            elif columns_delta > 0:
                feedback = self.negative_feedback_for_last_action()
            else:
                feedback = self.neutral_feedback_for_last_action()
            self.feedback([feedback])
        # convert to int because we may use it as an array index.
        prediction = int(self.tf_predict.eval({self.tf_input: self.current_state.reshape(-1, self.encoded_table_size)}, session=self.tf_session))
        #print('Prediction %d: %d' % (self.iteration_no, prediction))
        self.iteration_no += 1
        self.last_prediction = prediction
        return prediction

    def positive_feedback_for_last_action(self):
        """
        Last action had a positive outcome: encourage the algorithm to keep it up
        :return feedback: list[float]
        """
        feedback = [0, 0, 0]
        feedback[self.last_prediction] = 1.0
        return feedback

    def negative_feedback_for_last_action(self):
        """
        Last action had a negative outcome: punish the algorithm for that decision and tell it that it should have
        choosen another action (with a 50% probability)
        :return feedback: list[float]
        """
        feedback = [1.0, 1.0, 1.0]
        feedback[self.last_prediction] = 0
        return feedback

    def neutral_feedback_for_last_action(self):
        """
        Last action had a neutral outcome: we want the algorithm to be a fast player so we'll give him a little anxiety
        and apply a little punishment for its last decision by telling it that it should have chosen a better action
        :return feedback: list[float]
        """
        feedback = [0.5, 0.5, 0.5]
        feedback[self.last_prediction] = 0.0
        return feedback

    def calculate_column_heights_delta(self):
        heights = [ 0, 0, 0, ]
        rows = math.floor(len(self.current_table) / 3)
        reverse_table = list(reversed(self.current_table))
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
        delta = sum(heights) - sum(self.last_column_heights)
        #print('Heights [ %2d, %2d, %2d ] : delta -> %d '%(heights[0],heights[1],heights[2],delta))
        self.last_column_heights = heights
        return delta


    def feedback(self, feedback):
        algo_in = self.current_state.reshape(-1, self.encoded_table_size)
        _o, _cost = self.tf_session.run([self.optimizer, self.cost], feed_dict={self.tf_input: algo_in, self.tf_feedback: feedback})
        self.floating_averaged_cost = self.floating_averaged_cost * 0.6 + 0.4 * _cost

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
