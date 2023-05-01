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

start_time = print_msg(f'Читаю файл {file_preprocess_4}')
data = pd.read_feather(file_preprocess_4)
print_time(start_time)

start_time = print_msg('Преобразование в формат pyarrow...')
# преобразование в формат pyarrow
data = pa.Table.from_pandas(data)

data_agg = data.select(['user_id', 'url_host', 'request_cnt']). \
    group_by(['user_id', 'url_host']).aggregate([('request_cnt', "sum")])

pq.write_table(data_agg, file_data_agg)
print_time(start_time)

start_time = print_msg(f'Читаю файл {file_data_agg}')
data_agg = pq.read_table(file_data_agg)
print_time(start_time)