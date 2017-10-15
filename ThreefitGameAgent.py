# The game agent for Threefit

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from threading import Timer
import selenium.common.exceptions as scExc
import datetime
import re

"""The Threefit game agent"""

class ThreefitGameAgent():
    """
    A Game Agent for Threefit game
    """

    tables_buffer_size = 3
    tables_buffer = []
    stop_procedure = False
    iterations = 0
    debug = 0
    browser = None

    """ regex to extract ball id """
    balltype_regex = re.compile('.*ball([0-9])\.gif')
    """ score input selector """
    score_xpath = '/html/body/form/div/table[1]/tbody/tr[2]/td[4]/input'
    """ selector for balls on the table """
    ball_images_xpath = '/html/body/form/div/table[2]/tbody/tr/td/img'

    def __init__(self, _webdriver: WebDriver, _tables_buffer_size: int = 3, _debug_level: int = 0):
        self.browser = _webdriver
        self.tables_buffer_size = _tables_buffer_size
        self.debug = _debug_level
        self.init_tables_buffer()

    def init_tables_buffer(self):
        """ Initializes the tables buffer with empty tables (all zeroes) """
        for i in range(self.tables_buffer_size):
            tables_buffer = self.get_empty_table()

    def get_empty_table(self):
        """
        Get an empty table (a table made of all zeroes)
        :return a table made of all zeroes:
        :rtype: list[int]
        """
        return [0] * 36

    def read_game_table(self):
        """
        Fetch game table to array data structure

        :returns: array of integers
        :rtype: list[int] | None
        """
        table = []
        balls = self.browser.find_elements(By.XPATH, self.ball_images_xpath)
        for ball in balls:
            ballsrc = ball.get_attribute('src')
            balltype = self.balltype_regex.match(ballsrc)
            table.append(int(balltype.group(1)))
        return table

    def print_game_table(self, table: list):
        """
        Print game table
        :param table:
        :type table: list[int]
        """
        col = 0
        for itm in table:
            print(itm, end='')
            col = (col + 1) % 3
            if col == 0:
                print('')
        print('', flush=True)

    def update_tables_buffer(self, table:list):
        """
        Updates the tables buffer by prepending a new table to it
        :param table:
        :type table: list[int]
        """
        self.tables_buffer.insert(0, table)
        self.tables_buffer.pop()

    def play_game(self):
        """
        The play game procedure runs periodically every 0.1 seconds until we reach certain conditions.
        The procedure consists of:
        1. fetching the table status from the page
        2. feeding the algorithm with table status and choosing next action to perform
        3. perform the action on the table
        4. updating internal status if needed
        """
        try:
            # if next call doesn't rise an exception it means we have an alert in the browser (game has finished)
            if self.browser.switch_to.alert.text:
                if self.debug > 0:
                    print('Game ended.')
                self.stop_procedure = True
                iterations = 0
        except scExc.NoAlertPresentException:
            # exception occurred, this means no alert has been show, we're playing
            iterations += 1
            table = self.read_game_table()
            self.update_tables_buffer(table)
            action = self.feed_algorithm_and_get_action()
            self.perform_action(action)
            if self.debug > 1:
                self.print_game_table(table)
            if self.debug > 0 and (iterations % 50 == 0):
                print(' (', iterations, ') @ ', datetime.datetime.now())
        except scExc.WebDriverException:
            self.stop_procedure = True

    def start_game_loop(self):
        """
        Starts game loop executing the play_game procedure every 0.1 seconds
        """
        while not self.stop_procedure:
            timer = Timer(0.1, self.play_game)
            timer.start()
            timer.join()
        try:
            self.browser.quit()
        except:
            pass


    def perform_action(self, action:str):
        """
        Performs given action
        :param action: the action the agent should perform on the game
        :type action: str
        """

    def feed_algorithm_and_get_action(self):
        """
        Feeds the algorithm with new data and fetches the output to the action the agent should perform
        :returns: the action to perform ('l' for left, 'r' for right, 'd' for down, '' to stand still)
        :rtype: string
        """
        return ''