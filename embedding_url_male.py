import os
import gc
import pandas as pd
import numpy as np
import time

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import bisect

import scipy
import implicit

from time import time, sleep
from pathlib import Path
from glob import glob
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, ratio_groups
from mts_paths import *

__import__("warnings").filterwarnings('ignore')

# тут отфильтрованы только нужные user_id
targets = pd.read_feather(file_users)

# получение ембендингов по сайтам с % посещаемости по полу

# количество признаков

name_csv = f'url_male_agg_factors.csv'
url_male_agg_factors = WORK_PATH.joinpath(name_csv.replace('.csv', '.feather'))

if not url_male_agg_factors.is_file():
    start_time = print_msg(f'Читаю файл {file_preprocess_4}')
    data = pd.read_feather(file_preprocess_4)
    print_time(start_time)

    start_time = print_msg('Преобразование в формат pyarrow...')
    # преобразование в формат pyarrow
    data = pa.Table.from_pandas(data)
    print_time(start_time)

    start_time = print_msg('Группировка...')

    # эта группировка получается после пункта
    # --> получение ембендингов по сайтам
    data_agg = data.select(['user_id', 'url_host', 'request_cnt']). \
        group_by(['user_id', 'url_host']). \
        aggregate([('request_cnt', "sum")])
    data = None
    data_agg = data_agg.to_pandas()
    url_male = pd.read_feather(file_url_male)
    url_age = pd.read_feather(file_url_age)
    agg_col = ['user_id', 'url_host']
    url_col = ['url_host', 'male_prs_1', 'male_user_prs_1',
               'male_avg_prs_1']
    print_time(start_time)
    start_time = print_msg('Объединение...')
    # колонки из группировки по возрастным группам
    age_col = ['url_host']
    # нужно выделить индекс группы, в которой встречается True
    for symbol in 'gp':
        a_col = [f'url_{symbol}_{i}' for i in range(1, 7)]
        age_col.append(f'age_{symbol}')
        url_age[f'age_{symbol}'] = url_age.apply(
            lambda row: row[a_col].to_list().index(True) + 1, axis=1)

    data_agg = data_agg[agg_col].merge(url_male[url_col],
                                       on=['url_host'], how='left')
    data_agg = data_agg.merge(url_age[age_col],
                              on=['url_host'], how='left')

    data_agg = memory_compression(data_agg)
    print_time(start_time)
    start_time = print_msg('Сохранение...')
    data_agg.reset_index(drop=True).to_feather(url_male_agg_factors)
    data_agg.to_csv(url_male_agg_factors.with_suffix('.csv'), index=False)
    print(data_agg.columns)
    print(data_agg.info())
    print_time(start_time)

else:
    data_agg = pd.read_feather(url_male_agg_factors)

url_set = set(data_agg.url_host.unique())
print(f'{len(url_set)} urls')
url_dict = {url: idurl for url, idurl in zip(url_set, range(len(url_set)))}
usr_set = set(data_agg.user_id.unique())
print(f'{len(usr_set)} users')
usr_dict = {usr: user_id for usr, user_id in
            zip(usr_set, range(len(usr_set)))}

start_time = print_msg('Получение ембендингов по сайтам...')
# в качестве значений попробовать все колонки:
# 'male_prs_1', 'male_user_prs_1', 'male_avg_prs_1'
values = np.array(data_agg['male_user_prs_1'])
rows = np.array(data_agg['user_id'].map(usr_dict), dtype=int)
cols = np.array(data_agg['url_host'].map(url_dict), dtype=int)
