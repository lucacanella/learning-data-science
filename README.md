# Learning data science
Always wanted to learn data-science and machine learning, and here I'm trying.

## A little data science experiments and data gather tests

_Nothing interesting **yet**_

### twitter_test.py
Gather some data from Twitter with python-twitter

1. don't forget to ```pip install -Ur requirements.txt```
    * use ```--user``` parameter to install locally
2. create twitter_config.py file and specify twitter api parameters.
    * twitter_config.py file will be git ignored for obvious reasons.

*twitter_config.py* file example:
```
tw_user_screen_name='your_screen_name'
tw_user_id='your_user_id'
tw_consumer_key='app_consumer_key'
tw_consumer_secret='app_consumer_secret'
tw_access_token_key='app_access_token'
tw_access_token_secret='app_access_token_secret'
```

### scikit_learn_test.py
A couple of examples of scikit-learn usage.

In case this error is shown:
```
ImportError: cannot import name 'NUMPY_MKL'
```
You probably need to install numpy+mkl, see:
1. Look for required package here (depends on your architecture): http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
2. Stack overflow related answer: https://stackoverflow.com/questions/37267399/importerror-cannot-import-name-numpy-mkl
3. Ho to install whl packages: https://stackoverflow.com/questions/27885397/how-do-i-install-a-python-package-with-a-whl-file

ie. in my case: 
```
pip install --user numpy-1.13.1+mkl-cp35-cp35m-win_amd64.whl
```

#### does nothing yet...

### selenium_test.py

A couple of tests with Selenium on Python, ref: ```https://pypi.python.org/pypi/selenium```

Chrome driver is required (ie. put chromedriver.exe in local folder), see @ http://www.seleniumhq.org/download/.

#### Currently broken working on image crop and save...

### threefit.py

Just trying to make my computer play a simple online game.

### Not working yet.
 