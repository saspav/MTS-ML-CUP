import os
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

file_user_cpe = WORK_PATH.joinpath('file_user_cpe.feather')
file_cpe_models = WORK_PATH.joinpath('file_cpe_models.feather')
file_user_city_cpe = WORK_PATH.joinpath('file_user_city_cpe.feather')

file_train_df = WORK_PATH.joinpath('file_train_df.feather')
file_test_df = WORK_PATH.joinpath('file_test_df.feather')
df_train_users = WORK_PATH.joinpath('train_users.feather')

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

if file_train_df.is_file():
    print('Читаю тренировочный файл')

    df_train = pd.read_feather(file_train_df)

else:
    start_time = print_msg('Готовлю тренировочный датасет')
    data = pd.read_feather(file_preprocess_4)

    df_users = pa.Table.from_pandas(targets)
    df = pa.Table.from_pandas(data[data.user_id.isin(targets.user_id)])
    df = df.join(df_users, 'user_id')
    print_time(start_time)

    start_time = print_msg('Группирую данные по полу...')
    # группировка по полу
    #     grp_m = df.groupby(['url_host', 'is_male']).agg(
    #         male_count=('request_cnt', 'sum'),
    #         male_user_count=('user_id', lambda x: len(set(x.to_list())))
    #     )
    #     grp_m.reset_index(inplace=True)
    grp_m = df.select(['url_host', 'is_male', 'user_id', 'request_cnt']). \
        group_by(['url_host', 'is_male']). \
        aggregate([('request_cnt', "sum"),
                   ('user_id', "count_distinct")
                   ]).to_pandas()
    grp_m.rename({'request_cnt_sum': 'male_count',
                  'user_id_count_distinct': 'male_user_count'},
                 axis=1, inplace=True)
    print(grp_m)

    grp_m = memory_compression(grp_m, use_category=False)
    print_time(start_time)

    start_time = print_msg('Группирую данные по возрастным группам...')
    # группировка по возрастным группам
    # grp_a = df.groupby(['url_host', 'age']).agg(
    #     age_count=('request_cnt', 'sum'),
    #     age_user_count=('user_id', lambda x: len(set(x.to_list())))
    # )
    # grp_a.reset_index(inplace=True)
    grp_a = df.select(['url_host', 'age', 'user_id', 'request_cnt']). \
        group_by(['url_host', 'age']). \
        aggregate([('request_cnt', "sum"),
                   ('user_id', "count_distinct")
                   ]).to_pandas()
    grp_a.rename({'request_cnt_sum': 'age_count',
                  'user_id_count_distinct': 'age_user_count'},
                 axis=1, inplace=True)
    print(grp_a)

    grp_a = memory_compression(grp_a, use_category=False)
    print_time(start_time)

    # Обработка сгруппированных данных по полу и возрастным категориям

    start_time = print_msg('Обрабатываю группировки...')

    # получение кол-ва пользователей is_male = 1
    male_users = df_users.select(['is_male']).to_pandas()['is_male'].sum()

    url_male = ratio_groups(grp_m, 'is_male', 'male', (0, 1))
    # маркировка сайта -1 - женский, 0 - нейтральный, 1 - мужской
    # male_ratio_1, male_user_ratio_1, male_avg_1
    url_male['url_type'] = url_male.male_user_ratio_1.map(
        lambda x: (x > 0.45) + (x > 0.55) - 1)
    # маркировка популярности сайта
    url_male['fame_recs'] = url_male.male_1_count / url_male.male_1_count.sum()
    url_male['fame_user'] = url_male.male_user_1_count / male_users
    # суммирование количества пользователей по колонкам
    count_columns = ['male_1_count', 'male_user_1_count']
    total_counts = url_male[count_columns].sum(axis=0)
    print('Итоговое кол-во:\n', total_counts)
    print(url_male['fame_recs'].min(), url_male['fame_recs'].max())
    print(url_male['fame_user'].min(), url_male['fame_user'].max())

    url_male = memory_compression(url_male)

    url_age = ratio_groups(grp_a, 'age', 'age', range(1, 7))

    # маркировка сайта по группам - это пока не реализовано
    # П.1 - по максимальной принадлежности к группе
    # П.2 - по популярности сайта в группе

    url_age = memory_compression(url_age)

    print_time(start_time)


def process_step4(df, url_male):
    """
    Добавление сгруппированных данных в ДФ c url_id для группировки по user_id,
    чтобы посчитать какие url_id он посещал: сложить url_type, fame_user,
    получить кол-во посещенных уникальных сайтов, общее кол-во запросов
    :return: None
    """
    #  эту секцию проделать для url_age

    start_times = print_msg('Добавляю информацию по сайтам...')
    #  Добавление сгруппированных данных в ДФ c url_id для группировки по user_id,
    #  чтобы посчитать какие url_id он посещал: сложить url_type, fame_user,
    #  получить кол-во посещенных уникальных сайтов, общее кол-во запросов

    # все колонки
    merge_columns = url_male.columns
    # необходимые колонки
    merge_columns = ['url_id', 'url_type', 'fame_recs', 'fame_user']
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
        print(
            f'Обрабатываю {n + 1:02} часть, размер {size:_}, всего: {total:_}')
        temp = temp.merge(url_male[merge_columns], on='url_id', how='left')
        temp.to_pickle(f'step4_part{n:02}.pkl')

    df = temp = None
    gc.collect()
    print_time(start_times)

    print('Объединение файлов step4_part*.pkl')
    # объединение обработанных файлов в один ДФ
    name_files = sorted(glob('step4_part*.pkl'))
    df = pd.concat(pd.read_pickle(name_file) for name_file in name_files)
    print(df.info())

    # удаление временных файлов
    for name_file in name_files:
        Path(name_file).unlink()

    df = pa.Table.from_pandas(df)

    start_times = print_msg('Группирую данные...')
    # grp_users = df.groupby(['user_id']).agg(
    #     reqs_counts=('request_cnt', 'sum'),
    #     urls_counts=('url_host', 'count'),
    #     urls_unique=('url_host', lambda x: x.nunique()),
    #     # #
    #     # male_1_count=('male_1_count', 'sum'),
    #     # male_ratio_1=('male_ratio_1', 'sum'),
    #     # male_user_1_count=('male_user_1_count', 'sum'),
    #     # male_user_ratio_1=('male_user_ratio_1', 'sum'),
    #     # male_avg_1=('male_avg_1', 'sum'),
    #     # #
    #     url_type=('url_type', 'sum'),
    #     fame_recs=('fame_recs', 'sum'),
    #     fame_user=('fame_user', 'sum')
    # )
    grp_users = df.select(['user_id', 'url_host', 'request_cnt', 'url_type',
                           'fame_recs', 'fame_user']). \
        group_by(['user_id']). \
        aggregate([('request_cnt', "sum"),
                   ('url_host', "count"),
                   ('url_host', "count_distinct"),
                   ('url_type', 'sum'),
                   ('fame_recs', 'sum'),
                   ('fame_user', 'sum')
                   ]).to_pandas()
    # grp_users.rename({'request_cnt_sum': 'reqs_counts',
    #                   'user_id_count_distinct': 'age_user_count'},
    #                  axis=1, inplace=True)

    # #################################

    grp_users.reset_index(inplace=True)

    grp_users.reset_index(drop=True).to_feather(df_train_users)
    grp_users.to_csv(df_train_users.with_suffix('.csv'), index=False)
    print_time(start_times)



#
#
#
