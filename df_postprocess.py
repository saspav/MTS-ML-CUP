import os
import re
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

from mts_paths import *


def post_process(df):
    """Замена колонок на их процентное содержание:
    'male_0_count_sum', 'male_1_count_sum',
    'male_user_0_count_sum', 'male_user_1_count_sum',
    'age_1_count_sum', 'age_2_count_sum', 'age_3_count_sum',
    'age_4_count_sum', 'age_5_count_sum', 'age_6_count_sum',
    'age_user_1_count_sum', 'age_user_2_count_sum', 'age_user_3_count_sum',
    'age_user_4_count_sum', 'age_user_5_count_sum', 'age_user_6_count_sum'
    :param df: входной ДФ
    :return: обработанный и сжатый ДФ
    """
    if all(df.city_name_count == df.date_count):
        df.drop('city_name_count', axis=1, inplace=True)

    mask_columns = ('male_{}_count_sum', 'male_user_{}_count_sum',
                    'age_{}_count_sum', 'age_user_{}_count_sum')
    prs_columns = []
    for mask_col in mask_columns:
        cod_range = (0, 1) if mask_col.startswith('male') else range(1, 7)
        cnt_columns = [mask_col.format(cod) for cod in cod_range]
        print('Обрабатываю колонки:', cnt_columns)
        total_counts = df[cnt_columns].sum(axis=1)
        for cod in cod_range:
            ratio_name = f'{mask_col.split("{")[0]}prs_{cod}'
            count_name = mask_col.format(cod)
            prs_columns.append(ratio_name)
            print(count_name, ratio_name, sep=' --> ')
            df[ratio_name] = df[count_name] / total_counts
        df.drop(cnt_columns, axis=1, inplace=True)
        df.fillna(0, inplace=True)
        if 'user' in mask_col:
            print('Обрабатываю колонки:', prs_columns)
            half = len(cod_range)
            all_ratio_cols = [*zip(prs_columns[:half], prs_columns[half:])]
            for ratio_cols in all_ratio_cols:
                ma = 'age' if 'age' in mask_col else 'male'
                # получение последнего символа из наименования колонки
                name_avg = f"{ma}_avg_{ratio_cols[-1][-1]}"
                print(ratio_cols, name_avg, sep=': --> ')
                df[name_avg] = df[list(ratio_cols)].mean(axis=1)
            prs_columns = []
    drop_cols = [col for col in file_df.columns if re.match('male_.+_0', col)]
    df.drop(drop_cols, axis=1, inplace=True)
    df = df.merge(user_part_of_day, on='user_id', how='left')
    # возможно нужно убрать колонку pd_prs_0, т.к. она = 1-(prs_1+prs_2+prs_3)
    # df.drop('pd_prs_0', axis=1, inplace=True)
    return memory_compression(df)


start_time = print_msg(f'Читаю файлы...')
targets = pd.read_feather(file_users)
user_part_of_day = pd.read_feather(file_user_part_of_day)

replace_files = {file_train_df_pre: file_train_df,
                 file_test_df_pre: file_test_df}
for file in (file_train_df_pre, file_test_df_pre):
    file_df = pd.read_feather(file)
    file_df = post_process(file_df)
    file_df.reset_index(drop=True).to_feather(replace_files[file])
print_time(start_time)
