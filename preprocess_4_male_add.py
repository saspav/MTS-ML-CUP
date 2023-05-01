import os
import re
import gc
import pandas as pd
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import bisect

from time import time, sleep
from pathlib import Path
from glob import glob
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, ratio_groups
from mts_paths import *

__import__("warnings").filterwarnings('ignore')


def process_step4_add(df, train=True):
    """
    Добавление сгруппированных данных в ДФ c url_host для группировки по user_id,
    чтобы посчитать какие url_host он посещал: сложить url_type, fame_user,
    получить кол-во посещенных уникальных сайтов, общее кол-во запросов
    :return: None
    """
    start_time = print_msg('Добавляю информацию по сайтам...')

    url_male = pd.read_feather(file_url_male)
    url_age = pd.read_feather(file_url_age)

    # эту секцию проделать для url_age ########################################

    # необходимые колонки
    male_columns = ['url_host',
                    'fm_r0', 'fm_u0', 'fm_r1', 'fm_u1',
                    'fame0user', 'fame_user'
                    ]
    # колонки с маркировкой
    male_prs = [f"url_{pref[0] if pref else 'm'}_{cod}"
                for pref in ['', 'user_', 'avg_'] for cod in (0, 1)]
    # хватит ли памяти при объединении ДФ ???
    male_columns.extend(male_prs)

    col_patterns = ('url_g_{}', 'url_p_{}', 'fa_r{}', 'fa_u{}')
    age_prs = [patt.format(cod) for patt in col_patterns
               for cod in range(1, 7)]
    age_columns = ['url_host']
    # хватит ли памяти при объединении ДФ ???
    age_columns.extend(age_prs)

    # объединение сделать по кускам
    chunk_size = 30_000_000
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size:
        num_chunks += 1
    total = 0
    name_files = []
    for n in range(num_chunks):
        temp = df[n * chunk_size:(n + 1) * chunk_size]
        size = len(temp.index)
        total += size
        print(f'Добавляю {n + 1:02} часть, размер {size:_}, всего: {total:_}')
        temp = temp.merge(url_male[male_columns], on='url_host', how='left')
        temp = temp.merge(url_age[age_columns], on='url_host', how='left')
        file_temp = TEMP_PATH.joinpath(f'step4_part{n:02}.feather')
        temp.reset_index(drop=True).to_feather(file_temp)
        name_files.append(file_temp)

    print('temp.columns:', temp.columns)

    df = temp = None
    gc.collect()
    print_time(start_time)

    print('Объединение файлов step4_part*.feather')
    # объединение обработанных файлов в один ДФ

    df = pa.concat_tables(pa.Table.from_pandas(pd.read_feather(name_file))
                          for name_file in name_files)
    print('df.columns:', df.column_names)

    # удаление временных файлов
    for name_file in name_files:
        Path(name_file).unlink()

    start_time = print_msg('Группирую данные...')

    #     df = pa.Table.from_pandas(df)

    from_df = ['user_id'] + male_columns + age_prs
    print('from_df:', from_df)

    grp_users = df.select(from_df).group_by(['user_id']). \
        aggregate([
        # ('request_cnt', 'sum'),
        # ('url_host', 'count'),
        # ('url_host', 'count_distinct'),
        # ('male_0_count', 'sum'),
        # ('male_1_count', 'sum'),
        # ('male_user_0_count', 'sum'),
        # ('male_user_1_count', 'sum'),
        ('url_m_0', 'sum'),
        ('url_m_1', 'sum'),
        ('url_u_0', 'sum'),
        ('url_u_1', 'sum'),
        ('url_a_0', 'sum'),
        ('url_a_1', 'sum'),
        ('fm_r0', 'sum'),
        ('fm_u0', 'sum'),
        ('fm_r1', 'sum'),
        ('fm_u1', 'sum'),
        ('fame0user', 'sum'),
        ('fame_user', 'sum'),
        # ('age_1_count', 'sum'),
        # ('age_2_count', 'sum'),
        # ('age_3_count', 'sum'),
        # ('age_4_count', 'sum'),
        # ('age_5_count', 'sum'),
        # ('age_6_count', 'sum'),
        # ('age_user_1_count', 'sum'),
        # ('age_user_2_count', 'sum'),
        # ('age_user_3_count', 'sum'),
        # ('age_user_4_count', 'sum'),
        # ('age_user_5_count', 'sum'),
        # ('age_user_6_count', 'sum'),
        ('url_g_1', 'sum'),
        ('url_g_2', 'sum'),
        ('url_g_3', 'sum'),
        ('url_g_4', 'sum'),
        ('url_g_5', 'sum'),
        ('url_g_6', 'sum'),
        ('url_p_1', 'sum'),
        ('url_p_2', 'sum'),
        ('url_p_3', 'sum'),
        ('url_p_4', 'sum'),
        ('url_p_5', 'sum'),
        ('url_p_6', 'sum'),
        ('fa_r1', 'sum'),
        ('fa_r2', 'sum'),
        ('fa_r3', 'sum'),
        ('fa_r4', 'sum'),
        ('fa_r5', 'sum'),
        ('fa_r6', 'sum'),
        ('fa_u1', 'sum'),
        ('fa_u2', 'sum'),
        ('fa_u3', 'sum'),
        ('fa_u4', 'sum'),
        ('fa_u5', 'sum'),
        ('fa_u6', 'sum'),
    ]).to_pandas()
    # #########################################################################

    df = None
    gc.collect()

    grp_users.reset_index(drop=True, inplace=True)

    #     for col in ('url_male_sum', 'url_user_sum', 'url_avg_sum'):
    #         grp_users[col] = np.sign(grp_users[col])

    # удалить левые колонки
    for col in ('level_0', 'index'):
        if col in grp_users.columns:
            grp_users.drop(col, axis=1, inplace=True)

    if train:
        name_merge_file = df_train_users_add
    else:
        name_merge_file = df_test_users_add

    grp_users.reset_index(drop=True).to_feather(name_merge_file)

    print_time(start_time)
    return grp_users


found_agg_male = False
if file_url_male.is_file():
    url_male = pd.read_feather(file_url_male)
    if 'fame0user' in url_male.columns:
        found_agg_male = True

if not found_agg_male:
    start_times = print_msg('Готовлю тренировочный датасет')
    targets = pd.read_feather(file_users)
    data = pd.read_feather(file_preprocess_4)

    df_users = pa.Table.from_pandas(targets)

    # получение кол-ва пользователей is_male = 1
    male_users = targets.groupby('is_male').user_id.count()
    male0users = male_users[0]
    male_users = male_users[1]

    df = pa.Table.from_pandas(data[data.user_id.isin(targets.user_id)])
    df = df.join(df_users, 'user_id')
    print_time(start_times)

    start_time = print_msg('Группирую данные по полу...')
    grp_m = df.select(['url_host', 'is_male', 'user_id', 'request_cnt']). \
        group_by(['url_host', 'is_male']). \
        aggregate([('request_cnt', "sum"),
                   ('user_id', "count_distinct")
                   ]).to_pandas()
    grp_m.rename({'request_cnt_sum': 'male_count',
                  'user_id_count_distinct': 'male_user_count'},
                 axis=1, inplace=True)
    print(grp_m.sort_values(['url_host', 'is_male']))

    grp_m = memory_compression(grp_m, use_category=False)
    print_time(start_time)

    # Обработка сгруппированных данных по полу

    start_time = print_msg('Обрабатываю группировки...')

    url_male = ratio_groups(grp_m, 'is_male', 'male', (0, 1), url_rf=False)
    # print(url_male)

    edge = 0.55  # порог для маркировки сайта DS1
    edge = 0.53  # порог для маркировки сайта DS2
    edge = 0.60  # порог для маркировки сайта DS3
    edge = 0.63  # порог для маркировки сайта DS4

    # маркировка сайта -1 - женский, 0 - нейтральный, 1 - мужской
    # male_prs_1, male_user_prs_1, male_avg_prs_1
    # url_male['url_male'] = url_male.apply(
    #     lambda row: int(row.male_prs_1 > edge) - int(row.male_prs_0 > edge),
    #     axis=1)
    # url_male['url_user'] = url_male.apply(
    #     lambda row: int(row.male_user_prs_1 > edge) -
    #                 int(row.male_user_prs_0 > edge), axis=1)
    # url_male['url_avg'] = url_male.apply(
    #     lambda row: int(row.male_avg_prs_1 > edge) -
    #                 int(row.male_avg_prs_0 > edge), axis=1)

    # маркировка сайта Х_0 = 1 - женский Х_1 = 1 - мужской, оба = 0 -> нейтральный
    # url_m - request_cnt, url_u - user_id, url_a - среднее между ними
    for pref in ['', 'user_', 'avg_']:
        for cod in (0, 1):
            smb = pref[0] if pref else 'm'
            col = f'male_{pref}prs_{cod}'
            url_male[f'url_{smb}_{cod}'] = url_male[col] > edge

    # маркировка популярности сайта по полу
    for cod in (0, 1):
        a_cnt = f'male_{cod}_count'
        u_cnt = f'male_user_{cod}_count'
        url_male[f'fm_r{cod}'] = url_male[a_cnt] / url_male[a_cnt].sum()
        url_male[f'fm_u{cod}'] = url_male[u_cnt] / url_male[u_cnt].sum()

    # маркировка популярности сайта - старая версия
    url_male['fame0user'] = url_male.male_user_0_count / male0users
    url_male['fame_user'] = url_male.male_user_1_count / male_users

    # суммирование количества пользователей по колонкам
    cnt_columns = [c for c in url_male.columns if re.match('male_.+_count', c)]
    total_counts = url_male[cnt_columns].sum(axis=0)
    print('Итоговое кол-во:')
    print(total_counts)
    url_male.fillna(0, inplace=True)
    url_male = memory_compression(url_male, use_float=False)

    print_time(start_times)

    start_time = print_msg(f'Сохраняю файлы')

    url_male.reset_index(drop=True).to_feather(file_url_male)
    url_male.to_csv(file_url_male.with_suffix('.csv'), index=False)

    df_users = grp_a = grp_m = url_male = url_age = None
    gc.collect()
    print_time(start_time)

#

found_agg_age = False
if file_url_age.is_file():
    url_age = pd.read_feather(file_url_age)
    if 'fa_u6' in url_age.columns:
        found_agg_age = True

if not found_agg_age:
    start_times = print_msg('Готовлю тренировочный датасет')
    targets = pd.read_feather(file_users)
    data = pd.read_feather(file_preprocess_4)

    df_users = pa.Table.from_pandas(targets)

    df = pa.Table.from_pandas(data[data.user_id.isin(targets.user_id)])
    df = df.join(df_users, 'user_id')
    print_time(start_times)

    start_time = print_msg('Группирую данные по возрастным группам...')
    grp_a = df.select(['url_host', 'age', 'user_id', 'request_cnt']). \
        group_by(['url_host', 'age']). \
        aggregate([('request_cnt', "sum"),
                   ('user_id', "count_distinct")
                   ]).to_pandas()
    grp_a.rename({'request_cnt_sum': 'age_count',
                  'user_id_count_distinct': 'age_user_count'},
                 axis=1, inplace=True)
    print(grp_a.sort_values(['url_host', 'age']))

    grp_a = memory_compression(grp_a, use_category=False)
    print_time(start_time)

    # Обработка сгруппированных данных по возрастным категориям

    start_time = print_msg('Обрабатываю группировки...')

    url_age = ratio_groups(grp_a, 'age', 'age', range(1, 7), url_rf=False)
    # print(url_age)

    # маркировка сайта по возрастным группам
    # П.1 - по максимальной принадлежности к группе
    edge = 0.97  # порог определения максимальной принадлежности DS1
    edge = 0.98  # порог определения максимальной принадлежности DS2
    edge = 0.95  # порог определения максимальной принадлежности DS3
    edge = 0.98  # порог определения максимальной принадлежности DS4

    # max (age_1_count age_2_count age_3_count ...)
    age_cnt = [f'age_{cod}_count' for cod in range(1, 7)]
    max_cnt = url_age[age_cnt].max(axis=1)
    for cod in range(1, 7):
        url_age[f'url_g_{cod}'] = url_age[f'age_{cod}_count'] > max_cnt * edge

    age_cnt = [f'age_user_{cod}_count' for cod in range(1, 7)]
    max_cnt = url_age[age_cnt].max(axis=1)
    for cod in range(1, 7):
        url_age[f'url_p_{cod}'] = url_age[
                                      f'age_user_{cod}_count'] > max_cnt * edge
    max_cnt = None

    # П.2 - по популярности сайта в группе
    for cod in range(1, 7):
        a_cnt = f'age_{cod}_count'
        u_cnt = f'age_user_{cod}_count'
        url_age[f'fa_r{cod}'] = url_age[a_cnt] / url_age[a_cnt].sum()
        url_age[f'fa_u{cod}'] = url_age[u_cnt] / url_age[u_cnt].sum()

    url_age.fillna(0, inplace=True)
    url_age = memory_compression(url_age, use_float=False)
    print(url_age)

    print_time(start_times)

    start_time = print_msg(f'Сохраняю файлы')

    url_age.reset_index(drop=True).to_feather(file_url_age)
    url_age.to_csv(file_url_age.with_suffix('.csv'), index=False)

    df_users = grp_a = grp_m = url_male = url_age = None
    gc.collect()
    print_time(start_time)

#

start_times = print_msg(f'Готовлю датасеты')
for idx, file in enumerate((file_train_df, file_test_df)):
    train = not idx

    start_time = print_msg(f'Читаю файл {file_preprocess_4}')
    df = pd.read_feather(file_preprocess_4)
    df_users = pd.read_feather(file)
    print_time(start_time)

    # фильтруем только пользователей, которые есть в ДФ df_users
    df = df[df.user_id.isin(df_users.user_id)]

    grp_users = process_step4_add(df, train=train)

    df_users = df_users.merge(grp_users, on='user_id', how='left')

    df_users = memory_compression(df_users)

    if train:
        name_merge_file = df_train_users
    else:
        name_merge_file = df_test_users

    df_users.reset_index(drop=True).to_feather(name_merge_file)
    print_time(start_time)

print_time(start_times)
