import os
import re
import gc
import pandas as pd

from time import time, sleep
from pathlib import Path
from glob import glob
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, ratio_groups

__import__("warnings").filterwarnings('ignore')

from mts_paths import *

start_time = print_msg(f'Читаю файлы...')
cpe_models = pd.read_feather(file_cpe_models)

fill_price = False
if fill_price:
    # заполнение пропусков price_mean на основе модели по первому слову,
    # по фирме и далее просто средней цене

    cpe_models['word'] = cpe_models.cpe_model_name.map(
        lambda x: x.lower().split()[0] if ' ' in x else '')
    cpe_models.loc[
        cpe_models.price_mean.isnull(), 'price_mean'] = cpe_models.groupby(
        ['firm_id', 'word']).price_mean.transform('mean')
    cpe_models.loc[
        cpe_models.price_mean.isnull(), 'price_mean'] = cpe_models.groupby(
        ['firm_id']).price_mean.transform('mean')
    cpe_models.loc[
        cpe_models.price_mean.isnull(), 'price_mean'] = cpe_models.price_mean.mean()

    cpe_models.reset_index(drop=True).to_feather(file_cpe_models)

cpe_id_price = dict(
    cpe_models[['cpe_id', 'price_mean']].to_dict(orient='split')['data'])

# обновление в файлах колонки 'price_mean'
for file in (df_train_users, df_test_users):
    df = pd.read_feather(file)
    df.price_mean = df.cpe_id.map(cpe_id_price)
    print(df.isna().sum().sum())
    df.reset_index(drop=True).to_feather(file)

print_time(start_time)
