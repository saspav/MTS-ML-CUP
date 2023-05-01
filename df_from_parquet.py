import numpy as np
import pandas as pd
import time
import pyarrow.parquet as pq
from pathlib import Path
from glob import glob
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, concat_pickles

__import__("warnings").filterwarnings('ignore')

LOCAL_DATA_PATH = Path(__file__).parent.joinpath('context_data')
TARGET_FILE = LOCAL_DATA_PATH.joinpath('public_train.pqt')
SUBMIT_FILE = LOCAL_DATA_PATH.joinpath('submit.pqt')

MANUALS = ['region_name', 'city_name', 'cpe_manufacturer_name',
           'cpe_model_name', 'cpe_type_cd', 'cpe_model_os_type', 'price']

KIND_JOB = {0: ('df', ['user_id', 'url_host', 'request_cnt']),
            1: ('rb', MANUALS),
            2: ('rc', MANUALS[:2]),
            3: ('cpe', MANUALS[2:]),
            4: ('tst', ['user_id', 'url_host', 'request_cnt']),
            }


def transform_df(num_part, name_file, kind=0):
    """
    Преобразование данных
    :param num_part: номер части
    :param name_file: имя файла
    :param kind: вид работ: номер - кортеж (префикс, колонки для обработки)
                 0 - тренировочные данные
                 1 - справочники все
                 2 - области + населенные пункты
                 3 - справочники устройств
                 4 - тестовые данные
    :return:
    """

    pref_job = KIND_JOB[kind][0]
    cut_columns = KIND_JOB[kind][1]

    # если есть обработанный файл - прочитаем его, чтобы не формировать заново
    file_pickle = LOCAL_DATA_PATH.joinpath(f'part_{pref_job}_{num_part}.pkl')
    if file_pickle.is_file():
        start_time = print_msg(f'Читаю файл: {file_pickle}')
        df = pd.read_pickle(file_pickle)

    else:
        start_time = print_msg(f'Читаю файл: {name_file}')
        df = pq.read_table(name_file).to_pandas()
        print_time(start_time)
        # print(df.columns)

        start_time = print_msg('Обрабатываю данные...')

        if not kind:
            # возьмем пока только эти колонки
            df = df[df.user_id.isin(targets.user_id)][cut_columns]
            df = df.merge(targets, on='user_id')

            df.age = pd.to_numeric(df.age, errors='coerce')
            # уберем лиц младше 19 лет
            df.dropna(inplace=True)
            df.age = df.age.astype(int)
            df = df[df.age > 18]
            # уберем бесполых лиц
            df.is_male = pd.to_numeric(df.is_male, errors='coerce')
            df.dropna(inplace=True)
            df.is_male = df.is_male.astype(int)

            print(df.info())
            print(df.isna().sum())

            # деление на возрастные группы:
            # Класс 1 — 19-25, Класс 2 — 26-35, Класс 3 — 36-45,
            # Класс 4 — 46-55, Класс 5 — 56-65, Класс 6 — 66+
            # age_bins = [0, 18, 25, 35, 45, 55, 65, 999]
            # df['age_cat'] = pd.cut(df.age, bins=age_bins, labels=False)
            df['age_cat'] = df['age'].map(age_bucket)

        elif kind in (1, 2, 3):
            df = memory_compression(df[cut_columns].drop_duplicates())
            df.to_pickle(file_pickle)

            if kind == 1:
                for kind in (2, 3):
                    pref_job = KIND_JOB[kind][0]
                    cut_columns = KIND_JOB[kind][1]
                    file_pickle = LOCAL_DATA_PATH.joinpath(
                        f'part_{pref_job}_{num_part}.pkl')
                    tmp = memory_compression(df[cut_columns].drop_duplicates())
                    tmp.to_pickle(file_pickle)
            return
        elif kind == 4:
            # возьмем пока только эти колонки
            df = df[df.user_id.isin(id_to_submit.user_id)][cut_columns]

        df = memory_compression(df)
        print(df.info())

        df.to_pickle(file_pickle)

    print_time(start_time)

    # далее группировка только для тренировочных данных
    if kind:
        return

    # print(df.info())

    start_time = print_msg('Группирую данные...')

    # группировка по полу
    grp_m = df.groupby(['url_host', 'is_male']).agg(
        male_count=('request_cnt', 'sum'),
        users=('user_id', lambda x: x.to_list())
    )
    grp_m.reset_index(inplace=True)
    grp_m.users = grp_m.users.map(
        lambda x: set(x) if isinstance(x, list) else set())
    grp_m = memory_compression(grp_m, use_category=False)
    grp_male_pickle = LOCAL_DATA_PATH.joinpath(f'part_grp_male_{num_part}.pkl')
    grp_m.to_pickle(grp_male_pickle)

    # группировка по возрастным группам
    grp_a = df.groupby(['url_host', 'age_cat']).agg(
        age_count=('request_cnt', 'sum'),
        users=('user_id', lambda x: x.to_list())
    )
    grp_a.reset_index(inplace=True)
    grp_a.users = grp_a.users.map(
        lambda x: set(x) if isinstance(x, list) else set())
    grp_a = memory_compression(grp_a, use_category=False)
    grp_age_pickle = LOCAL_DATA_PATH.joinpath(f'part_grp_age_{num_part}.pkl')
    grp_a.to_pickle(grp_age_pickle)
    print_time(start_time)


LOCAL_DATA_PATH = Path(__file__).parent.joinpath('context_data')
print(LOCAL_DATA_PATH)

start_times = print_msg('Читаю TARGET_FILE и SUBMIT_FILE...')
targets = pq.read_table(TARGET_FILE).to_pandas()
targets.to_csv(TARGET_FILE.with_suffix('.csv'), index=False)
targets.to_pickle(TARGET_FILE.with_suffix('.pkl'))

id_to_submit = pq.read_table(SUBMIT_FILE).to_pandas()
id_to_submit.to_csv(SUBMIT_FILE.with_suffix('.csv'), index=False)
id_to_submit.to_pickle(SUBMIT_FILE.with_suffix('.pkl'))
print_time(start_times)

start_times = print_msg('Читаю файлы part*.parquet ...')
# 0 - тренировочные данные
# 1 - справочники все
# 2 - области + населенные пункты
# 3 - справочники устройств
# 4 - тестовые данные
kind_all_files = {0: 'train', 1: 'rb', 2: 'rc', 3: 'cpe', 4: 'test'}
for kind_task in (4,):
    # преобразование исходны файлов из паркета
    files = glob(f'{LOCAL_DATA_PATH}/part*.parquet')
    # for i, file in enumerate(files):
    #     transform_df(i, file, kind=kind_task)

    # объединение файлов в 1 ДФ
    all_tasks = [kind_task]
    if kind_task == 1:
        all_tasks.extend([2, 3])
    for task in all_tasks:
        pref_pkl = KIND_JOB[task][0]
        pkl_files = [LOCAL_DATA_PATH.joinpath(f'part_{pref_pkl}_{i}.pkl')
                     for i, _ in enumerate(files)]
        print(pkl_files)
        file = kind_all_files.get(task, 'all')
        all_df = concat_pickles(pkl_files)
        all_df = memory_compression(all_df)
        all_df.to_csv(LOCAL_DATA_PATH.joinpath(f'{file}_df.csv'), index=False)
        all_df.to_pickle(LOCAL_DATA_PATH.joinpath(f'{file}_df.pkl'))
print_time(start_times)
