from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import time
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, ratio_groups

__import__("warnings").filterwarnings('ignore')

LOCAL_DATA_PATH = Path(__file__).parent.joinpath('context_data')
TARGET_FILE = LOCAL_DATA_PATH.joinpath('public_train.pqt')
SUBMIT_FILE = LOCAL_DATA_PATH.joinpath('submit.pqt')

file_urls = LOCAL_DATA_PATH.joinpath(f'file_urls.pkl')
file_urls_kgl = LOCAL_DATA_PATH.joinpath(f'file_urls_kgl.pkl')
file_users = LOCAL_DATA_PATH.joinpath(f'file_users.pkl')
file_train = LOCAL_DATA_PATH.joinpath(f'train_df.pkl')
df_file_train = LOCAL_DATA_PATH.joinpath(f'train_df_step1.pkl')
df_file_train4 = LOCAL_DATA_PATH.joinpath(f'train_df_step4.pkl')
df_train_users = LOCAL_DATA_PATH.joinpath(f'train_users.pkl')


# start_times = print_msg('Читаю файл url_male.pkl')
# file_male = LOCAL_DATA_PATH.joinpath(f'url_male.pkl')
# df_male = pd.read_pickle(file_male)
# print(df_male.info())


def process_step1():
    """
    Обработка тренировочного файла по пользователям этап 1:
    - получение списка url_host --> создание справочника
    - замена url_host на url_id
    :return: None
    """
    start_times = print_msg('Читаю файл train_df.pkl')
    df = pd.read_pickle(file_train)
    # уберем лиц младше 19 лет
    df = df[df.age > 18]
    # По возрасту классы: Класс 1 — 19-25, Класс 2 — 26-35, Класс 3 — 36-45,
    # Класс 4 — 46-55, Класс 5 — 56-65, Класс 6 — 66+
    # pd.cut - нижняя граница не входит
    # age_bins = [0, 18, 25, 35, 45, 55, 65, 999]
    # df['age_cat'] = pd.cut(df.age, bins=age_bins, labels=False)
    df['age_cat'] = df['age'].map(age_bucket)
    df.drop('age', axis=1, inplace=True)
    print_time(start_times)
    print(df.info())
    print(df.isna().sum())

    df.url_host.fillna('0', inplace=True)

    df.url_host = df.url_host.astype('category')
    # df.is_male = df.is_male.astype('category')
    # df.age_cat = df.age_cat.astype('category')
    print(df.info())

    if not file_urls_kgl.is_file():
        start_times = print_msg('Сохраняю файл file_urls.pkl')
        url_hosts = pd.DataFrame(df.url_host.unique(), columns=['url_host'])
        url_hosts.sort_values('url_host', inplace=True, ignore_index=True)
        url_hosts['url_id'] = url_hosts.index
        url_hosts = memory_compression(url_hosts)
        url_hosts.to_pickle(file_urls)
        url_hosts.to_csv(file_urls.with_suffix('.csv'), index=False)
        print_time(start_times)
    else:
        url_hosts = pd.read_pickle(file_urls_kgl)

    df = df.merge(url_hosts, on='url_host', how='left')
    df.drop('url_host', axis=1, inplace=True)

    start_times = print_msg(f'Сохраняю файл {df_file_train}')
    df.to_pickle(df_file_train)
    df.to_csv(df_file_train.with_suffix('.csv'), index=False)
    print_time(start_times)


def process_step2():
    """
    Группировка данных по полу и возрастным категориям
    :return: None
    """
    start_times = print_msg(f'Читаю файл {df_file_train}')
    df = pd.read_pickle(df_file_train)
    print_time(start_times)

    start_times = print_msg(f'формирую файл {file_users}')
    df_users = df[['user_id', 'is_male', 'age_cat']].drop_duplicates()
    print(f'Общее количество пользователей = {len(df_users)}')
    print('\t\tРазбивка по полу:')
    print(df_users.groupby('is_male', as_index=False).user_id.count())
    df_users.to_pickle(file_users)
    df_users.to_csv(file_users.with_suffix('.csv'), index=False)
    print_time(start_times)

    start_time = print_msg('Группирую данные по полу...')
    # группировка по полу
    grp_m = df.groupby(['url_id', 'is_male']).agg(
        male_count=('request_cnt', 'sum'),
        male_user_count=('user_id', lambda x: len(set(x.to_list())))
    )
    grp_m.reset_index(inplace=True)
    print(grp_m.columns)
    print(grp_m)

    grp_m = memory_compression(grp_m, use_category=False)
    grp_male_pickle = LOCAL_DATA_PATH.joinpath(f'grp_male_step2.pkl')
    grp_m.to_pickle(grp_male_pickle)
    grp_m.to_csv(grp_male_pickle.with_suffix('.csv'), index=False)
    print_time(start_time)

    start_time = print_msg('Группирую данные по возрастным группам...')
    # группировка по возрастным группам
    grp_a = df.groupby(['url_id', 'age_cat']).agg(
        age_count=('request_cnt', 'sum'),
        age_user_count=('user_id', lambda x: len(set(x.to_list())))
    )
    grp_a.reset_index(inplace=True)
    print(grp_a.columns)
    print(grp_a)

    grp_a = memory_compression(grp_a, use_category=False)
    grp_age_pickle = LOCAL_DATA_PATH.joinpath(f'grp_age_step2.pkl')
    grp_a.to_csv(grp_age_pickle.with_suffix('.csv'), index=False)
    grp_a.to_pickle(grp_age_pickle)
    print_time(start_time)


def process_step3():
    """
    Обработка сгруппированных данных по полу и возрастным категориям
    :return: None
    """
    start_time = print_msg('Обрабатываю группировки...')

    # получение кол-ва пользователей is_male = 1
    df_users = pd.read_pickle(file_users)
    male_users = df_users.groupby('is_male', as_index=False).user_id.count()
    male_users = male_users.loc[1, 'user_id']

    grp_male_pickle = LOCAL_DATA_PATH.joinpath(f'grp_male_step2.pkl')
    grp_m = pd.read_pickle(grp_male_pickle)
    url_male = ratio_groups(grp_m, 'is_male', 'male', (0, 1), url_col='url_id')
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
    url_male_pickle = LOCAL_DATA_PATH.joinpath(f'grp_male_step3.pkl')
    url_male.to_pickle(url_male_pickle)
    url_male.to_csv(url_male_pickle.with_suffix('.csv'), index=False)
    print(url_male)

    grp_age_pickle = LOCAL_DATA_PATH.joinpath(f'grp_age_step2.pkl')
    grp_age = pd.read_pickle(grp_age_pickle)
    url_age = ratio_groups(grp_age, 'age_cat', 'age', range(1, 7),
                           url_col='url_id')

    # маркировка сайта по группам - это пока не реализовано
    # П.1 - по максимальной принадлежности к группе
    # П.2 - по популярности сайта в группе
    print_time(start_time)
    url_age = memory_compression(url_age)
    url_age_pickle = LOCAL_DATA_PATH.joinpath(f'grp_age_step3.pkl')
    url_age.to_pickle(url_age_pickle)
    url_age.to_csv(url_age_pickle.with_suffix('.csv'), index=False)
    print(url_age)

    print_time(start_time)


def process_step4():
    """
    Добавление сгруппированных данных в ДФ c url_id для группировки по user_id,
    чтобы посчитать какие url_id он посещал: сложить url_type, fame_user,
    получить кол-во посещенных уникальных сайтов, общее кол-во запросов
    :return: None
    """
    start_times = print_msg(f'Читаю файл {df_file_train}')
    df = pd.read_pickle(df_file_train)
    print_time(start_times)
    print(df.columns)

    start_times = print_msg('Добавляю информацию по сайтам...')
    url_male_pickle = LOCAL_DATA_PATH.joinpath(f'grp_male_step3.pkl')
    url_male = pd.read_pickle(url_male_pickle)
    print(url_male.columns)
    # все колонки
    merge_columns = url_male.columns
    # необходимые колонки
    merge_columns = ['url_id', 'url_type', 'fame_recs', 'fame_user']
    df = df.merge(url_male[merge_columns], on='url_id', how='left')
    print_time(start_times)

    start_times = print_msg(f'Сохраняю файл {df_file_train4}')
    df.to_pickle(df_file_train4)
    df.to_csv(df_file_train4.with_suffix('.csv'), index=False)
    print_time(start_times)

    start_times = print_msg('Группирую данные...')
    grp_users = df.groupby(['user_id']).agg(
        reqs_counts=('request_cnt', 'sum'),
        urls_counts=('url_id', 'count'),
        urls_unique=('url_id', lambda x: x.nunique()),
        # #
        # male_1_count=('male_1_count', 'sum'),
        # male_ratio_1=('male_ratio_1', 'sum'),
        # male_user_1_count=('male_user_1_count', 'sum'),
        # male_user_ratio_1=('male_user_ratio_1', 'sum'),
        # male_avg_1=('male_avg_1', 'sum'),
        # #
        url_type=('url_type', 'sum'),
        fame_recs=('fame_recs', 'sum'),
        fame_user=('fame_user', 'sum')
    )
    print(grp_users.columns)

    grp_users.reset_index(inplace=True)
    grp_users.to_pickle(df_train_users)
    grp_users.to_csv(df_train_users.with_suffix('.csv'), index=False)
    print_time(start_times)


# process_step1()
process_step2()
# process_step3()
# process_step4()

# url_male_pickle = LOCAL_DATA_PATH.joinpath(f'grp_male_step3.pkl')
# url_male = pd.read_pickle(url_male_pickle)
# print(url_male.columns)
# print(url_male.url_id.nunique())

# start_time = print_msg(f'Читаю файл {df_train_users}')
# grp_users = pd.read_pickle(df_train_users)
# tmp = grp_users.sample(20_000)
# tmp.to_csv('train_users_sample.csv', index=False)
# print_time(start_time)

# if file_urls_kgl.is_file():
#     url_host = pd.read_pickle(file_urls_kgl)
# else:
#     url_host= pd.read_pickle(file_urls)
#
# print(url_host.info())
#
# url_col = 'url_host'
#
# if 'url_host' in url_host.columns:
#     url_host.drop('url_host', axis=1, inplace=True)
# url_host.columns = [url_col]
# print(url_host)


# start_time = print_msg(f'Читаю файл {df_file_train}')
# df = pd.read_pickle(df_file_train)
# print_time(start_time)
# print(df.info())
# print(df.isna().sum())
