# The game agent for Threefit

from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from threading import Timer
import selenium.common.exceptions as scExc
import datetime
import re
from ThreefitAlgorithmV2 import ThreefitAlgorithmV2
from collections import deque
import sys

"""The Threefit game agent"""

class ThreefitGameAgentV2():
    """
    A Game Agent for Threefit game
    """

    stop_procedure = False
    iterations = 0
    debug = 0
    browser = None

    max_scores = 5
    scores = None
    n_games_played = 0


    """
    :ivar algorithm:
    :type ThreefitAlgorithmV2:
    """
    algorithm = None

    """ regex to extract ball id """
    balltype_regex = re.compile('.*ball([0-9])\.gif')
    """ score input selector """
    score_xpath = '/html/body/form/div/table[1]/tbody/tr[2]/td[4]/input'
    """ selector for balls on the table """
    ball_images_xpath = '/html/body/form/div/table[2]/tbody/tr/td/img'

    """ Actions map: maps an action id to the method that implements it. """
    actions_switcher = {}

    def __init__(self, _webdriver: WebDriver, _debug_level: int = 0):
        self.scores = deque(maxlen=self.max_scores)
        self.browser = _webdriver
        self.debug = _debug_level
        self.algorithm = ThreefitAlgorithmV2()
        self.actions_switcher = {
            0: self.action_left,
            1: self.action_down,
            2: self.action_right
        }

    def quitting(self):
        self.algorithm.save_model()

    def action_left(self):
        self.browser.find_element(By.XPATH, '/html/body/form/div/table[3]/tbody/tr/td[1]/input').click()
        pass

    def action_down(self):
        self.browser.find_element(By.XPATH, '/html/body/form/div/table[3]/tbody/tr/td[2]/input').click()
        pass

    def action_right(self):
        self.browser.find_element(By.XPATH, '/html/body/form/div/table[3]/tbody/tr/td[2]/input').click()
        pass

    def get_empty_table(self):
        """
        Get an empty table (a table made of all zeroes)
        :return a table made of all zeroes:
        :rtype: list[int]
        """
        return [0] * 36

    def read_game_score(self):
        score_input = self.browser.find_element(By.XPATH, self.score_xpath)
        last_score = score_input.get_attribute('value')
        if last_score is not None:
            self.last_read_score = int(last_score)
        else:
            self.last_read_score = 0

    def read_game_table(self):
        """
        Fetch game table to array data structure

        :returns: array of integers
        :rtype: list[int] | None
        """
        table = []
        balls = self.browser.find_elements(By.XPATH, self.ball_images_xpath)
        for ball in balls:
            try:
                ballsrc = ball.get_attribute('src')
                if type(ballsrc) is not str:
                    return None
                balltype = self.balltype_regex.match(ballsrc)
                table.append(int(balltype.group(1)))
            except scExc.UnexpectedAlertPresentException:
                pass
            except:
                print(' *** Exception occurred while reading table: %s' % sys.exc_info()[0])
                pass
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

    def start_game_loop(self):
        """
        Starts game loop executing the play_game procedure every 0.1 seconds
        """
        self.algorithm.init_algorithm()
        while not self.stop_procedure:
            timer = Timer(0.05, self.play_game)
            timer.start()
            timer.join()
        try:
            self.browser.quit()
        except:
            pass

    def game_ended(self):
        self.scores.append(self.last_read_score)

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
                self.algorithm.feedback_for_last_action(None) # game ended, send None as table
                self.game_ended()
                self.n_games_played += 1;
                print('Game %d ended. Average score over last %d games: %d' % (self.n_games_played, self.max_scores, sum(self.scores) / len(self.scores)))
                self.browser.switch_to.alert.accept()
        except scExc.NoAlertPresentException:
            if self.debug > 0 and (self.iterations % 25 == 0):
                print('Iterations: (', self.iterations,') @ %s, cost %f' % (datetime.datetime.now(), self.algorithm.floating_averaged_cost))
            # exception occurred, this means no alert has been show, we're playing
            self.iterations += 1
            self.read_game_score()
            table = self.read_game_table()
            self.algorithm.feedback_for_last_action(table)
            if table:
                action = self.algorithm.iterate(table)
                self.perform_action(action)
                if self.debug > 1:
                    self.print_game_table(table)
            else:
                print('No table found.')
        except OSError:
            pass
        except scExc.WebDriverException:
            self.stop_procedure = True
            self.quitting()
        except TypeError:
            #usually thrown when game ends but for some reason an alert is not yet present: just print the message and pass
            print(' *** Exception occurred: %s' % sys.exc_info()[0])
            pass

    def perform_action(self, action:int):
        """
        Performs given action
        :param action: the action the agent should perform on the game (0: left, 1: down, 2: right)
        :type action: int
        """
        try:
            self.actions_switcher[action]()
        except scExc.UnexpectedAlertPresentException:
            pass
        except:
            print(' *** Exception occurred while trying to take action: %s' % sys.exc_info()[0])
            pass
