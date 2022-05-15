import pandas as pd
import numpy as np
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)
from utils import TelegramPreprocessor, MarketTwitsPreprocessor

if __name__ == '__main__':
    # read data
    # tg_data_toloka = pd.read_csv('../data/tg_toloka.csv')
    tg_data_other = pd.read_csv('../data/tg_other.csv').loc[:5000, :]
    mt_data = pd.read_csv('../data/mt.csv')

    # process data
    # tg_toloka_transformer = TelegramPreprocessor(tg_data_toloka)
    tg_other_transformer = TelegramPreprocessor(tg_data_other)
    mt_transformer = MarketTwitsPreprocessor(mt_data)

    # tg_toloka_processed = tg_toloka_transformer.transform()
    tg_other_processed = tg_other_transformer.transform()
    # mt_processed = mt_transformer.transform()

    # save data
    # tg_toloka_processed.to_csv('../data/tg_toloka_processed.csv')
    tg_other_processed.to_csv('../data/tg_other_processed.csv')
    # mt_processed.to_csv('../data/mt_processed.csv')
