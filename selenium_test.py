from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import wait
from selenium.webdriver.support import expected_conditions as exp
import contextlib
from PIL import Image

browser = webdriver.Chrome()
browser.set_window_size(800,600)
browser.set_window_position(50,50)

browser.get('http://seleniumhq.org/')
wait.WebDriverWait(browser, 5, 0.1).until(exp.presence_of_element_located((By.ID, 'header')))
screen = browser.get_screenshot_as_png()

with open('./screen.png', 'wb') as f:
    f.write(screen)
    f.close()

elem = browser.find_element_by_id('header')
if elem.is_displayed():
    x1 = elem.location['x']
    width = elem.size['width']
    x2 = elem.location['x'] + width
    y1 = elem.location['y']
    height = elem.size['height']
    y2 = elem.location['y'] + height
    strdsize = browser.execute_script(script='return (window.visualViewport.width + (document.body.scrollWidth - document.body.clientWidth))+\',\'+window.visualViewport.height').split(',')
    dsize = tuple(int(e) for e in strdsize)
    im = Image.frombytes('RGB', dsize, screen)
    im.crop([x1,y1,x2,y2])
    im.save('screen_title.png')
script = "console.log('Ok!'); alert('OK!');"
browser.execute_script(script=script)
#browser.quit()