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

__import__("warnings").filterwarnings('ignore')

WORK_PATH = Path(__file__).parent.joinpath('context_data')

file_urls = WORK_PATH.joinpath('file_urls.feather')
file_users = WORK_PATH.joinpath('file_users.feather')

file_dataset_full = WORK_PATH.joinpath('dataset_full.feather')
file_user_part_of_day = WORK_PATH.joinpath('file_user_part_of_day.feather')

file_preprocess_0 = WORK_PATH.joinpath('data_set_preprocess_0.feather')
file_preprocess_1 = WORK_PATH.joinpath('data_set_preprocess_1.feather')
file_preprocess_2 = WORK_PATH.joinpath('data_set_preprocess_2.feather')
file_preprocess_3 = WORK_PATH.joinpath('data_set_preprocess_3.feather')
file_preprocess_4 = WORK_PATH.joinpath('data_set_preprocess_4.feather')
file_preprocess_5 = WORK_PATH.joinpath('data_set_preprocess_5.feather')

file_user_cpe = WORK_PATH.joinpath('file_user_cpe.feather')
file_cpe_models = WORK_PATH.joinpath('file_cpe_models.feather')
file_user_city_cpe = WORK_PATH.joinpath('file_user_city_cpe.feather')

file_url_male = WORK_PATH.joinpath('file_url_male.feather')
file_url_age = WORK_PATH.joinpath('file_url_age.feather')

file_train_pre = WORK_PATH.joinpath('file_train_pre.feather')
file_train_df = WORK_PATH.joinpath('file_train_df.feather')
file_test_df = WORK_PATH.joinpath('file_test_df.feather')
file_merge_train = WORK_PATH.joinpath('file_merge_train.feather')
file_merge_test = WORK_PATH.joinpath('file_merge_test.feather')
df_train_users = WORK_PATH.joinpath('train_users.feather')
df_test_users = WORK_PATH.joinpath('train_users.feather')

# тут отфильтрованы только нужные user_id
targets = pd.read_feather(file_users)

start_time = print_msg(f'Читаю файл {file_dataset_full}')
data = pd.read_feather(file_dataset_full)
# удаление неинформативных колонок
data.drop(['cpe_type_cd', 'cpe_model_os_type'], axis=1, inplace=True)
print_time(start_time)

start_time = print_msg('Группирую данные...')
data = pa.Table.from_pandas(data)
df = data.select(['user_id', 'part_of_day']). \
    group_by(['user_id', 'part_of_day']). \
    aggregate([('part_of_day', "count")]).to_pandas()
print_time(start_time)

start_time = print_msg('Перестраиваю данные...')
users = pd.DataFrame(df.user_id.unique(), columns=['user_id'])
users.sort_values('user_id', inplace=True, ignore_index=True)
codes = range(4)
for cod, part_day in enumerate(['morning', 'day', 'evening', 'night']):
    tmp = df[df['part_of_day'] == part_day].rename(
        columns={'part_of_day_count': f'part_day_{cod}_count'})
    merge_columns = ['user_id', f'part_day_{cod}_count']
    users = users.merge(tmp[merge_columns], on='user_id', how='left')
users.fillna(0, inplace=True)
cnt_columns = [f'part_day_{cod}_count' for cod in codes]
total_counts = users[cnt_columns].sum(axis=1)
for cod in codes:
    ratio_name = f'pd_prs_{cod}'
    count_name = f'part_day_{cod}_count'
    users[ratio_name] = users[count_name] / total_counts
users.fillna(0, inplace=True)
print_time(start_time)

users.drop(cnt_columns, axis=1, inplace=True)
users.reset_index(drop=True).to_feather(file_user_part_of_day)
