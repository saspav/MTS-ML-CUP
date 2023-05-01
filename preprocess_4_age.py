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

from time import time, sleep
from pathlib import Path
from glob import glob
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, ratio_groups

__import__("warnings").filterwarnings('ignore')

WORK_PATH = Path(__file__).parent.joinpath('context_data')

file_urls = WORK_PATH.joinpath('file_urls.feather')
file_users = WORK_PATH.joinpath('file_users.feather')

file_preprocess_0 = WORK_PATH.joinpath('data_set_preprocess_0.feather')
file_preprocess_1 = WORK_PATH.joinpath('data_set_preprocess_1.feather')
file_preprocess_2 = WORK_PATH.joinpath('data_set_preprocess_2.feather')
file_preprocess_3 = WORK_PATH.joinpath('data_set_preprocess_3.feather')

file_preprocess_4 = WORK_PATH.joinpath('data_set_preprocess_4.feather')
file_preprocess_4_sample = WORK_PATH.joinpath(
    'file_preprocess_4_sample.feather')

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

# тут отфильтровать только нужные user_id
targets = pd.read_feather(WORK_PATH.joinpath('target_train.feather'))
targets.age = pd.to_numeric(targets.age, errors='coerce')
# уберем лиц младше 19 лет
targets.dropna(inplace=True)
targets.age = targets.age.astype(int)
targets = targets[targets.age > 18]
# уберем бесполых лиц
targets.is_male = pd.to_numeric(targets.is_male, errors='coerce')
targets.dropna(inplace=True)
targets.is_male = targets.is_male.astype(int)
targets.age = targets.age.map(age_bucket)
targets = memory_compression(targets)

targets.reset_index(drop=True).to_feather(file_users)
targets.to_csv(file_users.with_suffix('.csv'), index=False)

start_time = print_msg('Готовлю тренировочный датасет')
data = pd.read_feather(file_preprocess_4_sample)

targets = targets.head(200)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!

df_users = pa.Table.from_pandas(targets)
df = pa.Table.from_pandas(data[data.user_id.isin(targets.user_id)])
df = df.join(df_users, 'user_id')
print_time(start_time)

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

# Обработка сгруппированных данных по полу и возрастным категориям

start_time = print_msg('Обрабатываю группировки...')

url_male = ratio_groups(grp_m, 'is_male', 'male', (0, 1), url_rf=False)
print(url_male)
# маркировка сайта -1 - женский, 0 - нейтральный, 1 - мужской
# male_prs_1, male_user_prs_1, male_avg_prs_1
edge = 0.55
# url_male['url_male'] = url_male.apply(
#     lambda row: int(row.male_prs_1 > edge) - int(row.male_prs_0 > edge),
#     axis=1)
# url_male['url_user'] = url_male.apply(
#     lambda row: int(row.male_user_prs_1 > edge) -
#                 int(row.male_user_prs_0 > edge), axis=1)
# url_male['url_avg'] = url_male.apply(
#     lambda row: int(row.male_avg_prs_1 > edge) -
#                 int(row.male_avg_prs_0 > edge), axis=1)
# маркировка сайта Х_0 = 1 - женский Х_1 = 1 - мужской, оба = 0 - нейтральный
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
# суммирование количества пользователей по колонкам
cnt_columns = [c for c in url_male.columns if re.match('male_.+_count', c)]
total_counts = url_male[cnt_columns].sum(axis=0)
print('Итоговое кол-во:')
print(total_counts)
url_male.fillna(0, inplace=True)
url_male = memory_compression(url_male, use_float=False)

url_age = ratio_groups(grp_a, 'age', 'age', range(1, 7), url_rf=False)
print(url_age)

# маркировка сайта по возрастным группам
# П.1 - по максимальной принадлежности к группе

# max (age_1_count age_2_count age_3_count age_4_count age_5_count age_6_count)
age_cnt = [f'age_{cod}_count' for cod in range(1, 7)]
max_cnt = url_age[age_cnt].max(axis=1)
for cod in range(1, 7):
    url_age[f'url_g_{cod}'] = url_age[f'age_{cod}_count'] > max_cnt * 0.97

age_cnt = [f'age_user_{cod}_count' for cod in range(1, 7)]
max_cnt = url_age[age_cnt].max(axis=1)
for cod in range(1, 7):
    url_age[f'url_p_{cod}'] = url_age[f'age_user_{cod}_count'] > max_cnt * 0.97

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

print_time(start_time)


def process_step4_old(df, train=True):
    """
    Добавление сгруппированных данных в ДФ c url_host для группировки по user_id,
    чтобы посчитать какие url_host он посещал: сложить url_type, fame_user,
    получить кол-во посещенных уникальных сайтов, общее кол-во запросов
    :return: None
    """
    start_time = print_msg('Добавляю информацию по сайтам...')

    url_male = pd.read_feather(file_url_male)
    url_age = pd.read_feather(file_url_age)

    #  эту секцию проделать для url_age ###########################################

    #  Добавление сгруппированных данных в ДФ c url_host для группировки по user_id,
    #  чтобы посчитать какие url_host он посещал: сложить url_type, fame_user,
    #  получить кол-во посещенных уникальных сайтов, общее кол-во запросов

    # все колонки
    male_columns = url_male.columns
    # эти колонки нужны для дополнительных эмбендингов
    male_prs = ['male_prs_1', 'male_user_prs_1', 'male_avg_prs_1']
    age_prs1 = ['age_prs_1', 'age_prs_2', 'age_prs_3', 'age_prs_4',
                'age_prs_5', 'age_prs_6']
    age_prs2 = ['age_user_prs_1', 'age_user_prs_2', 'age_user_prs_3',
                'age_user_prs_4', 'age_user_prs_5', 'age_user_prs_6']

    # необходимые колонки
    male_columns = ['url_host', 'url_male', 'url_user', 'url_avg', 'fm_r1',
                    'fm_u1']
    #  url_age, url_age_u, fa_r1, fa_u1, fa_r2, fa_u2, fa_r3, fa_u3, fa_r4, fa_u4, fa_r5, fa_u5, fa_r6, fa_u6
    age_columns = ['url_host', 'url_age', 'url_age_u']

    male_columns.extend(male_prs)
    age_columns.extend(age_prs1)

    # объединение сделать по кускам
    chunk_size = 20_000_000
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size:
        num_chunks += 1
    total = 0
    for n in range(num_chunks):
        temp = df[n * chunk_size:(n + 1) * chunk_size]
        size = len(temp.index)
        total += size
        print(f'Обрабатываю {n + 1:02} часть, размер {size:_}, всего: {total:_}')
        temp = temp.merge(url_male[male_columns], on='url_host', how='left')
        temp.reset_index(drop=True).to_feather(f'step4_part{n:02}.feather')

    df = temp = None
    gc.collect()
    print_time(start_time)

    print('Объединение файлов step4_part*.feather')
    # объединение обработанных файлов в один ДФ
    name_files = sorted(glob('step4_part*.feather'))
    df = pd.concat(pd.read_feather(name_file) for name_file in name_files)
    df.reset_index(drop=True, inplace=True)
    print(df.info())

    # удаление временных файлов
    for name_file in name_files:
        Path(name_file).unlink()

    if train:
        name_merge_file = file_merge_train
    else:
        name_merge_file = file_merge_test

    print(f'Сохраняю файл {name_merge_file}')
    df.reset_index(drop=True).to_feather(name_merge_file)
    print_time(start_time)

    start_time = print_msg('Группирую данные...')

    df = pa.Table.from_pandas(df)

    # ['user_id', 'url_host', 'request_cnt', 'age', 'is_male']
    from_df = ['user_id', 'request_cnt'] + male_columns
    grp_users = df.select(from_df).group_by(['user_id']). \
        aggregate([('request_cnt', "sum"),
                   ('url_host', "count"),
                   ('url_host', "count_distinct"),
                   ('url_male', 'sum'),
                   ('url_user', 'sum'),
                   ('url_avg', 'sum'),
                   #                    ('fm_r0', 'sum'), ('fm_u0', 'sum'),
                   ('fm_r1', 'sum'), ('fm_u1', 'sum')
                   ]).to_pandas()
    # ###################################################################################

    df = None
    gc.collect()

    grp_users.reset_index(drop=True, inplace=True)

    for col in ('url_male_sum', 'url_user_sum', 'url_avg_sum'):
        grp_users[col] = np.sign(grp_users[col])

    # удалить левые колонки
    for col in ('level_0', 'index'):
        if col in grp_users.columns:
            grp_users.drop(col, axis=1, inplace=True)

    if train:
        name_merge_file = df_train_users
    else:
        name_merge_file = df_test_users

    grp_users.reset_index(drop=True).to_feather(name_merge_file)
    grp_users.to_csv(name_merge_file.with_suffix('.csv'), index=False)

    print_time(start_time)
    return grp_users


def process_step4(df, train=True):
    """
    Добавление группированных данных в ДФ c url_host для группировки по user_id
    чтобы посчитать какие url_host он посещал: сложить url_type, fame_user,
    получить кол-во посещенных уникальных сайтов, общее кол-во запросов
    :return: None
    """
    start_time = print_msg('Добавляю информацию по сайтам...')

    url_male = pd.read_feather(file_url_male)
    url_age = pd.read_feather(file_url_age)

    # эту секцию проделать для url_age ########################################
    # все колонки
    male_columns = url_male.columns

    # необходимые колонки
    male_columns = ['url_host', 'male_0_count', 'male_1_count',
                    'male_user_0_count', 'male_user_1_count'
                    # 'url_male', 'url_user', 'url_avg', 'fm_r1', 'fm_u1'
                    ]
    # колонки с маркировкой
    male_prs = [f"url_{pref[0] if pref else 'm'}_{cod}"
                for pref in ['', 'user_', 'avg_'] for cod in (0, 1)]
    # хватит ли памяти при объединении ДФ ???
    # male_columns.extend(male_prs)

    age_cnt = [f'age_{cod}_count' for cod in range(1, 7)]
    age_users = [f'age_user_{cod}_count' for cod in range(1, 7)]
    age_columns = ['url_host'] + age_cnt + age_users

    age_prs1 = ['age_prs_1', 'age_prs_2', 'age_prs_3', 'age_prs_4',
                'age_prs_5', 'age_prs_6']
    age_prs2 = ['age_user_prs_1', 'age_user_prs_2', 'age_user_prs_3',
                'age_user_prs_4', 'age_user_prs_5', 'age_user_prs_6']
    # хватит ли памяти при объединении ДФ ???
    # age_columns.extend(age_prs1)

    # объединение сделать по кускам
    chunk_size = 20_000_000
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size:
        num_chunks += 1
    total = 0
    for n in range(num_chunks):
        temp = df[n * chunk_size:(n + 1) * chunk_size]
        size = len(temp.index)
        total += size
        print(f'Добавляю {n + 1:02} часть, размер {size:_}, всего: {total:_}')
        temp = temp.merge(url_male[male_columns], on='url_host', how='left')
        temp = temp.merge(url_age[age_columns], on='url_host', how='left')
        temp.reset_index(drop=True).to_feather(f'step4_part{n:02}.feather')

    df = temp = None
    gc.collect()
    print_time(start_time)

    print('Объединение файлов step4_part*.feather')
    # объединение обработанных файлов в один ДФ
    name_files = sorted(glob('step4_part*.feather'))
    df = pd.concat(pd.read_feather(name_file) for name_file in name_files)
    df.reset_index(drop=True, inplace=True)
    print(df.info())

    # удаление временных файлов
    for name_file in name_files:
        Path(name_file).unlink()

    if train:
        name_merge_file = file_merge_train
    else:
        name_merge_file = file_merge_test

    print(f'Сохраняю файл {name_merge_file}')
    df.reset_index(drop=True).to_feather(name_merge_file)
    print_time(start_time)

    start_time = print_msg('Группирую данные...')

    df = pa.Table.from_pandas(df)

    # ['user_id', 'url_host', 'request_cnt', 'age', 'is_male']
    from_df = ['user_id', 'request_cnt'] + male_columns
    grp_users = df.select(from_df).group_by(['user_id']). \
        aggregate([('request_cnt', 'sum'),
                   ('url_host', 'count'),
                   ('url_host', 'count_distinct'),
                   ('male_0_count', 'sum'),
                   ('male_1_count', 'sum'),
                   ('male_user_0_count', 'sum'),
                   ('male_user_1_count', 'sum'),
                   # ('url_male', 'sum'),
                   # ('url_user', 'sum'),
                   # ('url_avg', 'sum'),
                   # ('fm_r1', 'sum'), ('fm_u1', 'sum')
                   ('age_1_count', 'sum'),
                   ('age_2_count', 'sum'),
                   ('age_3_count', 'sum'),
                   ('age_4_count', 'sum'),
                   ('age_5_count', 'sum'),
                   ('age_6_count', 'sum'),
                   ('age_user_1_count', 'sum'),
                   ('age_user_2_count', 'sum'),
                   ('age_user_3_count', 'sum'),
                   ('age_user_4_count', 'sum'),
                   ('age_user_5_count', 'sum'),
                   ('age_user_6_count', 'sum'),
                   ]).to_pandas()
    # 'male_0_count', 'male_1_count', 'male_user_0_count', 'male_user_1_count'
    # #########################################################################

    df = None
    gc.collect()

    grp_users.reset_index(drop=True, inplace=True)

    prefix = 'male'
    prefixes = [prefix, 'male_user']
    codes = (0, 1)
    if prefix.startswith('male'):
        codes = (1,)
    for pref in prefixes:
        total_sum = sum(grp_users[f'{pref}_{cod}_count_sum'] for cod in codes)
        for cod in codes:
            ratio_name = f'{pref}_prs_{cod}'
            count_name = f'{pref}_{cod}_count_sum'
            grp_users[ratio_name] = grp_users[count_name] / total_sum
    grp_users.fillna(0, inplace=True)
    # посчитаем средний рейтинг между запросами и уникальными пользователями
    for cod in codes:
        prs_cols = [f'{pref}_prs_{cod}' for pref in prefixes]
        grp_users[f'{prefix}_avg_prs_{cod}'] = grp_users[prs_cols].mean(axis=1)

    for col in ('url_male_sum', 'url_user_sum', 'url_avg_sum'):
        grp_users[col] = np.sign(grp_users[col])

    # удалить левые колонки
    for col in ('level_0', 'index'):
        if col in grp_users.columns:
            grp_users.drop(col, axis=1, inplace=True)

    if train:
        name_merge_file = df_train_users
    else:
        name_merge_file = df_test_users

    grp_users.reset_index(drop=True).to_feather(name_merge_file)
    grp_users.to_csv(name_merge_file.with_suffix('.csv'), index=False)

    print_time(start_time)
    return grp_users
