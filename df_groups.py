from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import time
from print_time import print_time
from df_addons import memory_compression, concat_pickles, ratio_groups

__import__("warnings").filterwarnings('ignore')

LOCAL_DATA_PATH = Path(__file__).parent.joinpath('context_data')
TARGET_FILE = LOCAL_DATA_PATH.joinpath('public_train.pqt')
SUBMIT_FILE = LOCAL_DATA_PATH.joinpath('submit.pqt')

print(LOCAL_DATA_PATH)

start_time = time.time()
id_to_submit = pd.read_pickle(SUBMIT_FILE.with_suffix('.pkl'))
print_time(start_time)

start_time = time.time()
targets = pd.read_pickle(TARGET_FILE.with_suffix('.pkl'))
print_time(start_time)

# объединение файлов с группировкой по полу
print('Объединение файлов с группировкой по полу...')
start_time = time.time()
files = glob(f'{LOCAL_DATA_PATH}/part_grp_male*.pkl')
df_grp_male = concat_pickles(files, 'is_male', 'male')
df_grp_male.to_csv(LOCAL_DATA_PATH.joinpath('df_grp_male.csv'), index=False)
print(df_grp_male.info())
url_male = ratio_groups(df_grp_male, 'is_male', 'male', (0, 1))
url_male.to_csv(LOCAL_DATA_PATH.joinpath('url_male.csv'), index=False)
print(url_male)
print_time(start_time)

# объединение файлов с группировкой по возрасту
print('Объединение файлов с группировкой по возрасту...')
start_time = time.time()
files = glob(f'{LOCAL_DATA_PATH}/part_grp_age*.pkl')
df_grp_age = concat_pickles(files, 'age_cat', 'age')
df_grp_age.to_csv(LOCAL_DATA_PATH.joinpath('df_grp_age.csv'), index=False)
print(df_grp_age)
url_age = ratio_groups(df_grp_age, 'age_cat', 'age', range(1, 7))
url_age.to_csv(LOCAL_DATA_PATH.joinpath('url_age.csv'), index=False)
print(url_age)
print_time(start_time)
