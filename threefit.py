# Try to play Threefit on http://www.lutanho.net/play/threefit.html

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import wait
from selenium.webdriver.support import expected_conditions as exp
from ThreefitGameAgentV2 import ThreefitGameAgentV2
import os

# nascondi warnings di tensorflow per compilazione
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    #Start the game by opening the browser and looping the learning procedure.

    browser = webdriver.Chrome()
    browser.set_window_size(350, 660)
    browser.set_window_position(980, 10)
    browser.get('http://www.lutanho.net/play/threefit.html')
    # wait for title to show
    wait.WebDriverWait(browser, 5, 0.1).until(exp.title_is('Threefit'))
    # wait for game table to appear
    wait.WebDriverWait(browser, 3, 0.1).until(
        exp.presence_of_element_located((By.XPATH, "/html/body/form/div/table[2]"))
    )
    wait.WebDriverWait(browser, 3, 0.1).until(
        exp.visibility_of_element_located((By.XPATH, '/html/body/form/div/table[1]/tbody/tr[2]/td[3]/input'))
    )

    # set game difficulty to 5
    diff_plus_button = browser.find_element(By.XPATH, '/html/body/form/div/table[1]/tbody/tr[2]/td[3]/input')
    diff_plus_button.click()
    diff_plus_button.click()

    # game starts on his own we look at the table every 0.1 seconds
    agent = ThreefitGameAgentV2(_webdriver=browser, _debug_level=1)
    agent.start_game_loop()
