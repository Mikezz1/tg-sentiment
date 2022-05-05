import numpy as np
import pandas as pd
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)


class SentimentAdjuster:  
    def __init__(self, series: pd.DataFrame, weekend_type: str, smoothing_type:str, smoothing_window: int):
        self.weekend_type = weekend_type
        self.smoothing_type = smoothing_type
        self.smoothing_window = smoothing_window
        self.series = series.copy()
        
    def _calc_weekends(self):
        weekend_sentiment = []
        weekend_sentiment_cnt=[]
        cum_sent  = 0
        cnt = 0
        for row in self.series.iloc[::-1].iterrows(): # итерируемся с конца тк дообавлять нужно к предшествующему элементу
            if np.isnan(row[1]['price']): #  check for weekend
                cum_sent+=row[1]['sentiment']
                weekend_sentiment.append(0)
                cnt+=1
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
        self.series.loc[:, 'weekend_sentiment_cnt'] = weekend_sentiment_cnt[::-1]
        self.series.loc[:, 'last_weekend_sent'] = pd.concat([pd.Series([0]), self.series\
                .loc[:, 'weekend_sentiment'][:-1]])
        self.series.loc[:, 'adj_sentiment'] = self.series.loc[:, 'sentiment'] + self.series.loc[:, 'weekend_sentiment']
            
            
    def _apply_smoothing(self):
        if self.smoothing_type == 'expontneial':
            self.series.loc[:, 'adj_sentiment'] = self.series.loc[:, 'adj_sentiment'].ewm(self.smoothing_window).mean()
        elif self.smoothing_type == 'rolling':
            self.series.loc[:, 'adj_sentiment'] = self.series.loc[:, 'adj_sentiment'].rolling(self.smoothing_window).mean()
            
    def transform(self):
        weekend_sentiment, weekend_sentiment_cnt = self._calc_weekends()
        self._adj_weekends(weekend_sentiment, weekend_sentiment_cnt)
        self._apply_smoothing()
        print('Done')
        
        
class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def clean_text(self):
        """
        Function to remove all special symbols
        """
        self.df.loc[:, 'message'] = self.df.loc[:, 'message']\
                        .apply(self._remove_emojis)\
                        .apply(self._remove_tags)\
                        .apply(self._remove_spaces)\
                        .apply(self._remove_links)
                        
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.rename(columns={'Unnamed: 0':'idx'})
            
        return self.df
    
    def _remove_emojis(text):
        eng = 'abcdefghijklmnopqrstuvwxyz'
        rus = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        numbers = '0123456789'
        symbols = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        eng += eng.upper()
        rus += rus.upper()
        stop_list = eng + rus + numbers + symbols
        text = [char for char in text if char not in stop_list]
        return text

    def _remove_tags(text):
        a = text.split()
        return " ".join([word for word in a if (word[0] != '#')\
                                    and (not word.startswith('#',1))\
                                    and (word[0] != '@')\
                                    and (not word.startswith('@',1))])

    def _remove_spaces(text):
        return text.replace('\n', ' ')
    
    def _remove_links(string):
        return ' '.join([x for x in string.split(' ') if 'http' not in x])
    
    
    def get_entities(text, segmenter, emb, ner_tagger):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        num_entities = len(doc.spans)
        entities = []
        if num_entities > 0:
            for span in doc.spans:
                if span.type  == 'ORG':
                    entities.append(span.text)
                    
        return entities
    
    
    
class TelegramPreprocessor(Preprocessor):
    
    def _find_mentions(self, data_exp):
        data_exp.loc[data_exp['entities'].str.startswith(('тиньк','tink','tcs')), 'entities'] = 'тинькофф'
        data_exp.loc[data_exp['entities'].str.startswith(('сбер','sber')), 'entities'] = 'сбер'
        data_exp.loc[data_exp['entities'].str.startswith(('втб','vtb')), 'entities'] = 'втб'
        data_exp.loc[data_exp['entities'].str.startswith(('газпр','gazp')), 'entities'] = 'газпром'
        data_exp.loc[data_exp['entities'].str.startswith(('яндек','yand','yndx')), 'entities'] = 'яндекс'
        data_exp.loc[data_exp['entities'].str.startswith(('роснеф','rosn')), 'entities'] = 'роснефть'
        data_exp.loc[data_exp['entities'].str.startswith(('лукой','luko', 'lkoh')), 'entities'] = 'лукойл'
        data_exp.loc[data_exp['entities'].str.startswith(('полюс','polus', 'plzl')), 'entities'] = 'полюс'
        data_exp.loc[data_exp['entities'].str.startswith(('мтс','mts')), 'entities'] = 'мтс'
        data_exp.loc[data_exp['entities'].str.startswith(('полиметал','poly')), 'entities'] = 'полиметалл'
        data_exp.loc[data_exp['entities'].str.startswith(('норник','norn','gmkn')), 'entities'] = 'норникель'
        data_exp.loc[data_exp['entities'].str.startswith(('татне','татн','tatn')), 'entities'] = 'татнефть'
        data_exp.loc[data_exp['entities'].str.startswith(('лента','lnta')), 'entities'] = 'лента'
        data_exp.loc[data_exp['entities'].str.startswith(('алрос','alrs')), 'entities'] = 'алроса'
        data_exp.loc[data_exp['entities'].str.startswith(('афк','afks')), 'entities'] = 'система'
        data_exp.loc[data_exp['entities'].str.startswith(('магнит','magn')), 'entities'] = 'магнит'
        data_exp.loc[data_exp['entities'].str.startswith(('пятерочк','x5','five')), 'entities'] = 'x5'
        data_exp.loc[data_exp['entities'].str.startswith(('фосагро','phor')), 'entities'] = 'фосагро'
        data_exp.loc[data_exp['entities'].str.startswith(('северстал','chmf')), 'entities'] = 'северсталь'
        data_exp.loc[data_exp['entities'].str.startswith(('нлмк','nlmk')), 'entities'] = 'нлмк'
        data_exp.loc[data_exp['entities'].str.startswith(('россети','rsti')), 'entities'] = 'россети'
        data_exp.loc[data_exp['entities'].str.startswith(('уралкали','urka')), 'entities'] = 'уралкалий'
        data_exp.loc[data_exp['entities'].str.startswith(('озон','ozon')), 'entities'] = 'озон'
        data_exp.loc[data_exp['entities'].str.startswith(('ростелек','rtkm')), 'entities'] = 'ростелеком'
        data_exp.loc[data_exp['entities'].str.startswith(('aэрофл','aflt')), 'entities'] = 'aэрофлот'
        data_exp.loc[data_exp['entities'].str.startswith(('новатэк','nvtk')), 'entities'] = 'новатэк'
        data_exp.loc[data_exp['entities'].str.startswith(('детский','dsky')), 'entities'] = 'детск.мир'
        data_exp.loc[data_exp['entities'].str.startswith(('киви','qiwi')), 'entities'] = 'qiwi'
        data_exp.loc[data_exp['entities'].str.startswith(('газпром нефть','sibn','газпром-нефть','газпромнефть')), 'entities'] = 'газпромнефть'
        data_exp.loc[data_exp['entities'].str.startswith(('башнефт','bane')), 'entities'] = 'башнефть'
        data_exp.loc[data_exp['entities'].str.startswith(('мосбирж','moex')), 'entities'] = 'moex'
        data_exp.loc[data_exp['entities'].str.startswith(('русагро','agro')), 'entities'] = 'русагро'
        data_exp.loc[data_exp['entities'].str.startswith(('вк','vkco','vk','mail','мейл','мейл.ру')), 'entities'] = 'вк'
        data_exp.loc[data_exp['entities'].str.startswith(('интер рао','agro')), 'irao'] = 'интеррао'
        data_exp.loc[data_exp['entities'].str.startswith(('сургутнефтегаз','sngs','sngsp')), 'entities'] = 'сургутнефтегаз'
        data_exp.loc[data_exp['entities'].str.startswith(('пик','pikk')), 'entities'] = 'пикк'
        data_exp.loc[data_exp['entities'].str.startswith(('hh','hh.tu')), 'entities'] = 'hh'
        data_exp.loc[data_exp['entities'].str.startswith(('транснефть','trnf','trnfp')), 'entities'] = 'транснефть'
        data_exp.loc[data_exp['entities'].str.startswith(('петропавловск','pogr')), 'entities'] = 'петропавловск'
        data_exp.loc[data_exp['entities'].str.startswith(('нефть','мосбирж','ртс','rts','imoex','micex','ммвб','rtsi','moex')), 'entities'] = 'imoex'
        
        
        return data_exp
    
    def _extract_tickers(self):
        segmenter = Segmenter()
        emb = NewsEmbedding()
        ner_tagger = NewsNERTagger(emb)
        
        self.df['entities'] = self.df['message']\
            .apply(lambda a: self.get_entities(a, segmenter, emb, ner_tagger)) \
            .apply(np.unique)
            
        data_exp = self.df.explode('entities')
        data_exp['entities'] = data_exp['entities'].str.lower().fillna('')
        data_exp = self._find_mentions(data_exp)
        return data_exp
        
    def _add_ticker_column(self, data_exp):
        all_tickers = data_exp['entities'] \
                .groupby('idx',as_index=False) \
                .agg({'entities': lambda a: ' '.join(list(np.unique(a)))}) \
                .rename(columns={'entities':'tickers'})
                    
        self.df = self.df.merge(all_tickers, on='idx', how='left')
        self.df.loc[:, 'entities'] = self.df.loc[:, 'entities'] \
                                        .apply(lambda a: '_'.join(a))
                                        
        return self.df
        
    def transform(self):
        return self._add_ticker_column(self._extract_tickers())
                                        
        
    
class MarketTwitsPreprocessor(Preprocessor):
    def __init__():
        pass