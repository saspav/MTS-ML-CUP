import re
import pandas as pd
from glob import glob
from mts_paths import *

__import__("warnings").filterwarnings('ignore')

# file_submit_age = PREDICTIONS_DIR.joinpath('cb_submit_age_024.csv')
file_submit_age = PREDICTIONS_DIR.joinpath('cb_submit_age_292.csv')
# file_submit_male = PREDICTIONS_DIR.joinpath('cb_submit_male_308.csv')
file_submit_male = PREDICTIONS_DIR.joinpath('cb_submit_male_308.csv')

num_age = re.findall('age_\d+', file_submit_age.name)[0]
num_male = re.findall('male_\d+', file_submit_male.name)[0]
new_submit = PREDICTIONS_DIR.joinpath(f'submit_{num_age}_{num_male}.csv')
print(new_submit)

if file_submit_age.is_file():
    df_age = pd.read_csv(file_submit_age, index_col='user_id')
else:
    files = glob(f'{PREDICTIONS_DIR}/{file_submit_age.stem}*.csv')
    print(files)
    df_age = pd.concat([pd.read_csv(file, header=0, names=['user_id', 'old'],
                                    index_col='user_id') for file in files],
                       axis=1)
    df_age['age'] = df_age.mean(axis=1).round(0).astype(int)
    df_age = pd.DataFrame(df_age['age'], columns=['age'])
    print(df_age)

if file_submit_male.is_file():
    df_male = pd.read_csv(file_submit_male, index_col='user_id')
else:
    files = glob(f'{PREDICTIONS_DIR}/{file_submit_male.stem}*.csv')
    print(files)
    df_male = pd.concat([pd.read_csv(file, header=0, names=['user_id', 'sex'],
                                     index_col='user_id') for file in files],
                        axis=1)
    df_male['is_male'] = df_male.mean(axis=1)
    print(df_male)

df_age = df_age.merge(df_male['is_male'], on='user_id')
df_age.to_csv(new_submit)
