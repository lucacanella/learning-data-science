import matplotlib as mpl
import sklearn as skl
import pandas as pd
import twitter as twit
import twitter_config as twCfg #create twitter_config.py file and put here credentials and other params

twApi = twit.Api(access_token_key= twCfg.tw_access_token_key,
                 access_token_secret= twCfg.tw_access_token_secret,
                 consumer_key= twCfg.tw_consumer_key,
                 consumer_secret= twCfg.tw_consumer_secret)

