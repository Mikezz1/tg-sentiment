# tg-sentiment: анализ сентимента финансовых сообщений в Телеграмм

- `data` - папка для всех данных
- `preprocess`  - очистка данных и NER
- `nn` - обучение и inference классификатора
- `index` - построение индексов

Данные для обучения доступны по [ссылке](https://www.kaggle.com/datasets/mikezz11/telegram-financial-sentiment-ru), финальные данные для построения индексов можно скачать в виде архива по [ссылке](https://drive.google.com/file/d/1RBVVPb9CEljgQBDiNkxEoKLL43iNjn-O/view?usp=sharing) или при помощи скрипта `load_data.sh`. Веса для модели также доступны по [ссылке](https://drive.google.com/file/d/1MnL4jJ4WGJo28mxMK2DGPyyfgZilvoXl/view?usp=sharing) и в виде соответствующего скрипта
