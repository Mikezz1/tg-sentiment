import pandas as pd
import numpy as np


class SentimentAdjuster:
    def __init__(self, series: pd.DataFrame, weekend_type: str,
                 smoothing_type: str, smoothing_window: int):
        self.weekend_type = weekend_type
        self.smoothing_type = smoothing_type
        self.smoothing_window = smoothing_window
        self.series = series.copy()

    def _calc_weekends(self):
        weekend_sentiment = []
        weekend_sentiment_cnt = []
        cum_sent = 0
        cnt = 0
        # итерируемся с конца тк дообавлять нужно к предшествующему элементу
        for row in self.series.iloc[::-1].iterrows():
            if np.isnan(row[1]['price']):  # check for weekend
                cum_sent += row[1]['sentiment']
                weekend_sentiment.append(0)
                cnt += 1
                weekend_sentiment_cnt.append(0)
            else:
                if cnt > 0:
                    weekend_sentiment.append(cum_sent/cnt)
                else:
                    weekend_sentiment.append(cum_sent)
                weekend_sentiment_cnt.append(cnt)
                cum_sent = 0
                cnt = 0
        return weekend_sentiment, weekend_sentiment_cnt

    def _adj_weekends(self, weekend_sentiment, weekend_sentiment_cnt):
        self.series.loc[:, 'weekend_sentiment'] = weekend_sentiment[::-1]
        self.series.loc[:,
                        'weekend_sentiment_cnt'] = weekend_sentiment_cnt[::-1]
        self.series.loc[:, 'last_weekend_sent'] = pd.concat(
            [pd.Series([0]), self.series .loc[:, 'weekend_sentiment'][:-1]])
        self.series.loc[:, 'adj_sentiment'] = self.series.loc[:,
                                                              'sentiment'] + self.series.loc[:, 'weekend_sentiment']

    def _apply_smoothing(self):
        if self.smoothing_type == 'expontneial':
            self.series.loc[:, 'adj_sentiment'] = self.series.loc[:,
                                                                  'adj_sentiment'].ewm(self.smoothing_window).mean()
        elif self.smoothing_type == 'rolling':
            self.series.loc[:, 'adj_sentiment'] = self.series.loc[:,
                                                                  'adj_sentiment'].rolling(self.smoothing_window).mean()

    def transform(self):
        weekend_sentiment, weekend_sentiment_cnt = self._calc_weekends()
        self._adj_weekends(weekend_sentiment, weekend_sentiment_cnt)
        self._apply_smoothing()
        print('Done')
