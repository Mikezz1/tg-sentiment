from html import entities
import numpy as np
import pandas as pd
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)


class Preprocessor:
    def __init__(self, df):
        self.df = df.copy()

    def clean_text(self, df):
        """
        Function to remove all special symbols
        """
        df.message = self.df.message.apply(str)
        df.loc[:, 'message'] = df.loc[:, 'message']\
            .apply(self._remove_emojis)\
            .apply(self._remove_tags)\
            .apply(self._remove_spaces)\
            .apply(self._remove_links)

        if 'Unnamed: 0' in df.columns:
            df = df.rename(columns={'Unnamed: 0': 'idx'})

        return df

    @staticmethod
    def _remove_emojis(text):
        eng = 'abcdefghijklmnopqrstuvwxyz'
        rus = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
        numbers = '0123456789'
        symbols = ' !"#$%&()*+,-./:;<=>?@[]^_`{|}~'
        eng += eng.upper()
        rus += rus.upper()
        approved_list = eng + rus + numbers + symbols
        text = [char for char in text if char in approved_list]
        return ''.join(text)

    @staticmethod
    def _remove_tags(text):
        a = text.split()
        return ' '.join([word for word in a if (word[0] != '  # ')
                         and (not word.startswith('#', 1))
                         and (word[0] != '@')
                         and (not word.startswith('@', 1))])

    @staticmethod
    def _remove_spaces(text):
        return text.replace('\n', ' ')

    @staticmethod
    def _remove_links(string):
        return ' '.join([x for x in string.split(' ') if 'http' not in x])

    @staticmethod
    def get_entities(text, segmenter, emb, ner_tagger):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_ner(ner_tagger)
        num_entities = len(doc.spans)
        entities = []
        if num_entities > 0:
            for span in doc.spans:
                if span.type == 'ORG':
                    entities.append(span.text)
        return entities


class TelegramPreprocessor(Preprocessor):

    def _find_mentions(self, data_exp):
        data_exp.loc[data_exp['entities'].str.startswith(
            ('тиньк', 'tink', 'tcs')), 'entities'] = 'тинькофф'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('сбер', 'sber')), 'entities'] = 'сбер'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('втб', 'vtb')), 'entities'] = 'втб'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('газпр', 'gazp')), 'entities'] = 'газпром'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('яндек', 'yand', 'yndx')), 'entities'] = 'яндекс'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('роснеф', 'rosn')), 'entities'] = 'роснефть'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('лукой', 'luko', 'lkoh')), 'entities'] = 'лукойл'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('полюс', 'polus', 'plzl')), 'entities'] = 'полюс'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('мтс', 'mts')), 'entities'] = 'мтс'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('полиметал', 'poly')), 'entities'] = 'полиметалл'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('норник', 'norn', 'gmkn')), 'entities'] = 'норникель'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('татне', 'татн', 'tatn')), 'entities'] = 'татнефть'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('лента', 'lnta')), 'entities'] = 'лента'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('алрос', 'alrs')), 'entities'] = 'алроса'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('афк', 'afks')), 'entities'] = 'система'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('магнит', 'magn')), 'entities'] = 'магнит'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('пятерочк', 'x5', 'five')), 'entities'] = 'x5'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('фосагро', 'phor')), 'entities'] = 'фосагро'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('северстал', 'chmf')), 'entities'] = 'северсталь'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('нлмк', 'nlmk')), 'entities'] = 'нлмк'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('россети', 'rsti')), 'entities'] = 'россети'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('уралкали', 'urka')), 'entities'] = 'уралкалий'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('озон', 'ozon')), 'entities'] = 'озон'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('ростелек', 'rtkm')), 'entities'] = 'ростелеком'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('aэрофл', 'aflt')), 'entities'] = 'aэрофлот'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('новатэк', 'nvtk')), 'entities'] = 'новатэк'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('детский', 'dsky')), 'entities'] = 'детск.мир'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('киви', 'qiwi')), 'entities'] = 'qiwi'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('газпром нефть', 'sibn', 'газпром-нефть',
             'газпромнефть')),
            'entities'] = 'газпромнефть'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('башнефт', 'bane')), 'entities'] = 'башнефть'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('мосбирж', 'moex')), 'entities'] = 'moex'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('русагро', 'agro')), 'entities'] = 'русагро'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('вк', 'vkco', 'vk', 'mail', 'мейл', 'мейл.ру')), 'entities'] = 'вк'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('интер рао', 'agro')), 'irao'] = 'интеррао'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('сургутнефтегаз', 'sngs', 'sngsp')),
            'entities'] = 'сургутнефтегаз'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('пик', 'pikk')), 'entities'] = 'пикк'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('hh', 'hh.tu')), 'entities'] = 'hh'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('транснефть', 'trnf', 'trnfp')),
            'entities'] = 'транснефть'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('петропавловск', 'pogr')),
            'entities'] = 'петропавловск'
        data_exp.loc[data_exp['entities'].str.startswith(
            ('нефть', 'мосбирж', 'ртс', 'rts', 'imoex', 'micex',
             'ммвб', 'rtsi', 'moex')),
            'entities'] = 'imoex'
        return data_exp

    def _extract_tickers(self):
        segmenter = Segmenter()
        emb = NewsEmbedding()
        ner_tagger = NewsNERTagger(emb)

        self.df['entities'] = self.df['message'].apply(
            lambda a: self.get_entities(a, segmenter, emb, ner_tagger)).apply(
            np.unique)
        data_exp = self.df.explode('entities')
        data_exp['entities'] = data_exp['entities'].str.lower().fillna('')
        data_exp = self._find_mentions(data_exp)

        return data_exp

    def _add_ticker_column(self, data_exp):
        tickers = ['татнефть', 'норникель', 'полиметалл', 'мтс',
                   'полюс', 'лукойл', 'роснефть', 'яндекс',
                   'газпром', 'втб', 'сбер', 'тинькофф',
                   'лента', 'алроса', 'система', 'магнит',
                   'x5', 'фосагро', 'северсталь', 'нлмк',
                   'россети', 'уралкалий', 'озон', 'ростелеком',
                   'aэрофлот', 'новатэк', 'детск.мир', 'qiwi',
                   'газпромнефть', 'башнефть', 'русагро', 'moex',  # новое
                   'вк', 'интеррао', 'сургутнефтегаз', 'пик', 'hh',
                   'транснефть', 'петропавловск'
                   ]

        all_tickers = data_exp[data_exp['entities']
                               .isin(tickers)] \
            .groupby('idx', as_index=False) \
            .agg({'entities': lambda a: ' '.join(list(np.unique(a)))}) \
            .rename(columns={'entities': 'tickers'})

        self.df = self.df.merge(all_tickers, on='idx', how='left')
        self.df.loc[:, 'entities'] = self.df.loc[:, 'entities'] \
            .apply(lambda a: '_'.join(a))
        self.df = self.df[self.df.tickers != ' '].dropna()

    def _exclude_channels(self, channels=None):
        return self.df.loc[~self.df.channels.isin(channels), :]

    def _filter_mentions_count(self, mentions_cnt=[1, 2, 3, 4]):
        self.df.loc[:, 'entities_cnt'] = self.df.loc[:,
                                                     'entities'].str.split('_').apply(len)
        result = self.df.loc[self.df['entities_cnt'].isin(mentions_cnt), :]
        return result

    def transform(self):
        self.df = self.clean_text(self.df)
        self._add_ticker_column(self._extract_tickers())
        return self._filter_mentions_count()


class MarketTwitsPreprocessor(Preprocessor):
    def __init__(self, df):
        self.df = df.copy().dropna()
        self.pos1 = u"\U0001F4A5"
        self.neg1 = u"\u26A0"
        self.neut1 = u"\u2757"

    def get_labelled_items(self) -> pd.DataFrame:

        positive = self.df[self.df['message'].str.contains(self.pos1)]
        negative = self.df[self.df['message'].str.contains(self.neg1)]

        a = ~(self.df['message'].str.contains(self.pos1) |
              self.df['message'].str.contains(self.neg1))
        neutral = self.df[(self.df['message'].str.contains(self.neut1)) & a]

        positive = self.clean_text(positive)
        negative = self.clean_text(negative)
        neutral = self.clean_text(neutral)

        positive.loc[:, 'label'] = 1
        neutral.loc[:, 'label'] = 0
        negative.loc[:, 'label'] = -1

        self.df = pd.concat([positive, neutral, negative])

    def find_mentions(self, tickers, tags):
        entities = tickers + tags
        entities = [a.lower() for a in entities]
        data_tickers = self.df[self.df['message'].str.lower().str.contains(
            '|'.join(entities))]
        return data_tickers

    def transform(self):
        tickers = [
            'AFKS', 'AFLT', 'AKRN', 'ALRS', 'CBOM', 'CHMF', 'DSKY', 'ENPG',
            'FIVE', 'FIXP', 'GAZP', 'GMKN', 'HHRU', 'HYDR', 'IRAO', 'LKOH',
            'LNTA', 'MAGN', 'MGNT', 'MOEX', 'MTSS', 'NLMK', 'NVTK', 'OZON',
            'PHOR', 'PIKK', 'PLZL', 'POGR', 'POLY', 'QIWI', 'RNFT', 'ROSN',
            'RSTI', 'RTKM', 'RUAL', 'SBER', 'SIBN', 'SNGS', 'TATN', 'TCS',
            'TCSG', 'TRNFP', 'URKA', 'VEON', 'VKCO', 'VTBR', 'YNDX', 'MAIL']
        other = ['#спг', '#отчетности',
                 '#отчетность', '#газ', '#золото', '#РТС']

        self.get_labelled_items()
        return self.find_mentions(tickers, other)
