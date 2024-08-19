from dotenv import load_dotenv
import os
import requests

import pandas as pd
import numpy as np

import finnhub


load_dotenv()

finnhub_key = os.getenv('FINNHUB_KEY')
finnhub_client = finnhub.client(api_key=finnhub_key)

newsapi_key = os.getenv('NEWSAPI_KEY')
