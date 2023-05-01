from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import time
from print_time import print_time, print_msg
from df_addons import age_bucket, memory_compression, ratio_groups

__import__("warnings").filterwarnings('ignore')

LOCAL_DATA_PATH = Path(__file__).parent.joinpath('context_data')

MANUALS = ['region_name', 'city_name', 'cpe_manufacturer_name',
           'cpe_model_name', 'cpe_type_cd', 'cpe_model_os_type', 'price']

KIND_JOB = {0: ('df', ['user_id', 'url_host', 'request_cnt']),
            1: ('rb', MANUALS),
            2: ('rc', MANUALS[:2]),
            3: ('cpe', MANUALS[2:]),
            4: ('tst', ['user_id', 'url_host', 'request_cnt']),
            }

file_sample = LOCAL_DATA_PATH.joinpath(f'file_sample.pkl')
file_cities = LOCAL_DATA_PATH.joinpath(f'rc_df.pkl')
file_cities_id = LOCAL_DATA_PATH.joinpath(f'rc_df_id.pkl')
file_devices = LOCAL_DATA_PATH.joinpath(f'cpe_df.pkl')
user_id_city_cpe = LOCAL_DATA_PATH.joinpath(f'user_id_city_cpe.pkl')
user_idx_city_cpe = LOCAL_DATA_PATH.joinpath(f'user_idx_city_cpe.pkl')


def process_step5():
    """
    Обработка справочников этап 5:
    - получение списка городов --> создание справочника
    :return: None
    """
    start_times = print_msg('Читаю файл rc_df.pkl')
    df = pd.read_pickle(file_cities)
    print_time(start_times)

    print(df.columns)

    df.drop_duplicates(inplace=True)
    df.sort_values('city_name', inplace=True, ignore_index=True)
    df['city_id'] = df.index

    regions = pd.DataFrame(df['region_name'].drop_duplicates(),
                           columns=['region_name'])
    regions.sort_values('region_name', inplace=True, ignore_index=True)
    regions['region_id'] = regions.index
    regions.to_csv(LOCAL_DATA_PATH.joinpath(f'regions.csv'), index=False)

    df = df.merge(regions, on='region_name', how='left')

    # df['region_city'] = (df.region_name.astype(str) + '|' +
    #                      df.city_name.astype(str))

    print(df.info())
    df = memory_compression(df)
    # print(df.info())
    regs = dict(regions.to_dict(orient='split')['data'])
    print(regs)

    start_times = print_msg(f'Сохраняю файл {file_cities_id}')
    df.to_pickle(file_cities_id)
    df.to_csv(file_cities_id.with_suffix('.csv'), index=False)
    print_time(start_times)

    data = pd.read_pickle(file_sample)
    data = data.merge(df, on=['region_name', 'city_name'], how='left')
    print(data)
    print(data.isna().sum())


def process_step6():
    """
    Обработка справочников этап 6:
    - добавление к user_id_city_cpe.csv информации по id нас.пунктов
    :return: None
    """
    start_times = print_msg('Обрабатываю файл user_idx_city_cpe.pkl')
    df = pd.read_pickle(file_cities_id)
    # print(df.info())
    users = pd.read_pickle(user_id_city_cpe)
    print(users.info())
    users = users.merge(df, on=['region_name', 'city_name'], how='left')
    print(users.columns)
    save_cols = ['user_id', 'region_id', 'city_id', 'city_name_count',
                 'request_cnt_sum', 'different_city', 'firm_id', 'cpe_id',
                 'cpe_model_name_count']
    users = users[save_cols].rename(
        columns={'cpe_model_name_count': 'cpe_count'})
    cat_cols = ['region_id', 'city_id', 'firm_id', 'cpe_id']
    for cat_col in cat_cols:
        users[cat_col] = users[cat_col].astype('category')

    users.to_pickle(user_idx_city_cpe)
    users.to_csv(user_idx_city_cpe.with_suffix('.csv'), index=False)
    print_time(start_times)


# process_step5()
# process_step6()

# grp = pd.DataFrame()
# cities = grp[['city_name', 'region_name']].drop_duplicates()
# cities.sort_values(['city_name', 'region_name'], inplace=True, ignore_index=True)
# cities['city_id'] = cities.index
#
# regions = pd.DataFrame(cities['region_name'].drop_duplicates(),
#                        columns=['region_name'])
# regions.sort_values('region_name', inplace=True, ignore_index=True)
# regions['region_id'] = regions.index
# cities = cities.merge(regions, on='region_name', how='left')
#
# grp = grp.merge(cities, on=['city_name', 'region_name'], how='left')
# grp.drop(['city_name', 'region_name'], axis=1, inplace=True)

start_times = print_msg('Читаю файл rc_df.pkl')
cities = pd.read_pickle(file_cities)
data = pd.read_pickle(file_sample)
print_time(start_times)

print(data.columns)
print(cities.columns)


def make_region_city():
    cities = data[['region_name', 'city_name']].drop_duplicates()
    cities.sort_values(['city_name', 'region_name'], inplace=True,
                       ignore_index=True)
    cities['city_id'] = cities.index

    regions = pd.DataFrame(cities['region_name'].drop_duplicates(),
                           columns=['region_name'])
    regions.sort_values('region_name', inplace=True, ignore_index=True)
    regions['region_id'] = regions.index
    cities = cities.merge(regions, on='region_name', how='left')

    cities['region_city'] = (cities.region_id.astype(str) + '|' +
                             cities.city_name.astype(str))

    cities = memory_compression(cities)
    cities.to_csv('cities.csv', index=False)
    print(cities.info())

    regions = dict(regions.to_dict(orient='split')['data'])
    cities = dict(
        cities[['region_city', 'city_id']].to_dict(orient='split')['data'])
    return regions, cities


# удаляю из-за нехватки памяти
data.drop(['cpe_manufacturer_name', 'cpe_model_name', 'price'], axis=1,
          inplace=True)

# получение индексов для region_name, city_name
regions, cities = make_region_city()

chunk_size = 20_000_000
num_chunks = len(data) // chunk_size
if len(data) % chunk_size:
    num_chunks += 1
for n in range(num_chunks):
    temp = data[n * chunk_size:(n + 1) * chunk_size]
    print(f'Обрабатываю {n + 1} часть, размер {len(temp.index)}')
    # замена region_name, city_name на индексы
    temp.region_name = temp.region_name.map(regions)
    temp.city_name = temp.region_name.astype(
        str) + '|' + temp.city_name.astype(str)
    temp.city_name = temp.city_name.map(cities)
    temp.to_pickle(f'step2_part{n:02}.pkl')
