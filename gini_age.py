import numpy as np
import pandas as pd

import sklearn.metrics as m
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from ast import literal_eval
from print_time import print_time, print_msg
from df_addons import df_to_excel
from mts_paths import *

__import__("warnings").filterwarnings('ignore')

build_model = False

start_time = print_msg('Запускаю классификатор...')

SEED = 2023
MY_DATA_PATH = WORK_PATH

file_logs = Path(r'D:\python-txt\mts\gini_age.logs')
if not file_logs.is_file():
    with open(file_logs, mode='a') as log:
        log.write('num;DS;als_factors;als_column;url_male_als_factors;'
                  'WF1;2*WF1;name_emb_factors;model_columns;'
                  'category_columns;comment\n')
    max_num = 0
else:
    df = pd.read_csv(file_logs, sep=';')
    if 'num' not in df.columns:
        df.insert(0, 'num', 0)
        df.num = df.index + 1
    if 'DS' not in df.columns:
        df.insert(1, 'DS', 'DS1')
        df.num = df.index + 1
    if 'comment' not in df.columns:
        df['comment'] = ''
    df.als_column = df.als_column.map(
        lambda x: literal_eval(x) if x[0] == '(' else x)
    df.als_column = df.als_column.map(
        lambda x: ", ".join(x) if isinstance(x, (list, tuple)) else x)
    df.num = df.index + 1
    max_num = df.num.max()

# for als_factors in range(114, 127, 1):
for als_factors in (97,):

    name_emb_factors = []
    # количество признаков
    als_factors = 97
    # als_factors = 104
    # als_factors = 100
    # als_factors = 150
    name_csv = f'url_usr_emb_factors_{als_factors}.csv'
    url_usr_emb_factors = MY_DATA_PATH.joinpath(name_csv.replace('.csv',
                                                                 '.feather'))
    name_emb_factors.append(url_usr_emb_factors)

    # количество признаков
    region_als_factors = 20
    name_csv = f'region_usr_emb_factors_{region_als_factors}.csv'
    region_usr_emb_factors = MY_DATA_PATH.joinpath(name_csv.replace('.csv',
                                                                    '.feather'))
    name_emb_factors.append(region_usr_emb_factors)

    # количество признаков
    city_als_factors = 30
    name_csv = f'city_usr_emb_factors_{city_als_factors}.csv'
    city_usr_emb_factors = MY_DATA_PATH.joinpath(name_csv.replace('.csv',
                                                                  '.feather'))
    # name_emb_factors.append(city_usr_emb_factors)

    # количество признаков
    url_male_als_factors = 100
    # male_prs_1=0.730036, male_user_prs_1=0.730056, male_avg_prs_1=0.729514
    als_columns_male = ('male_prs_1', 'male_user_prs_1', 'male_avg_prs_1')
    # als_columns_male = ('male_prs_1', 'male_user_prs_1')
    # als_columns_male = ('male_prs_1',)
    # als_columns_male = ('male_user_prs_1',)
    als_columns_male = ('male_avg_prs_1',)
    for als_column in als_columns_male:
        name_a = f'url_{als_column}_emb_factors_{url_male_als_factors}.feather'
        url_male_emb_factors = MY_DATA_PATH.joinpath(name_a)
        if not url_male_emb_factors.is_file():
            continue
        print(url_male_emb_factors)
        name_emb_factors.append(url_male_emb_factors)
    #

    targets = pd.read_feather(file_users)
    df = pd.read_feather(df_train_users)
    df = targets.merge(df, how='left', on='user_id')
    df.fillna(0, inplace=True)
    # удалить колонку index
    for col in ('index', 'url_host_count_distinct_y'):
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    print(df.columns)

    # df['is_male'].value_counts()

    all_model_columns = [
        'age', 'is_male', 'user_id',
        # 'region_name', 'city_name', 'cpe_id', 'firm_id',
        'request_cnt_sum',
        'url_host_count',
        'url_host_count_distinct',
        'different_city',
        'date_count',
        'city_name_count_distinct',
        'region_name_count_distinct',
        'date_count_distinct',
        'male_prs_1',
        # 'male_user_prs_1',  # с этим меньше
        # 'male_avg_1',  #
        'age_prs_1', 'age_prs_2', 'age_prs_3', 'age_prs_4', 'age_prs_5',
        'age_prs_6',
        'age_user_prs_1', 'age_user_prs_2', 'age_user_prs_3', 'age_user_prs_4',
        'age_user_prs_5', 'age_user_prs_6',
        # 'age_avg_1','age_avg_2', 'age_avg_3', 'age_avg_4', 'age_avg_5', 'age_avg_6',
        'pd_prs_0', 'pd_prs_1', 'pd_prs_2', 'pd_prs_3',
        'url_m_0_sum', 'url_m_1_sum',  # GINI по полу 0.724
        'url_u_0_sum', 'url_u_1_sum',  # GINI по полу 0.724
        # 'url_a_0_sum', 'url_a_1_sum', # GINI по полу 0.716
        'fm_r0_sum', 'fm_u0_sum',
        'fm_r1_sum', 'fm_u1_sum',
        'fame0user_sum', 'fame_user_sum'
    ]

    age_model_columns = ['url_g_1_sum', 'url_g_2_sum', 'url_g_3_sum',
                         'url_g_4_sum', 'url_g_5_sum', 'url_g_6_sum',
                         'url_p_1_sum', 'url_p_2_sum', 'url_p_3_sum',
                         'url_p_4_sum', 'url_p_5_sum', 'url_p_6_sum',
                         'fa_r1_sum', 'fa_r2_sum', 'fa_r3_sum',
                         'fa_r4_sum', 'fa_r5_sum', 'fa_r6_sum',
                         'fa_u1_sum', 'fa_u2_sum', 'fa_u3_sum',
                         'fa_u4_sum', 'fa_u5_sum', 'fa_u6_sum']

    # для предсказания пола - это плохо годится
    all_model_columns.extend(age_model_columns)

    # колонки с категориями
    all_categories = (
        'cpe_id',
        'firm_id',
        # 'region_name',
        'city_name',
        'url_m',
        'url_u',
        'url_a',
    )

    learn_exclude = ['fame0user_sum', 'region_name', 'price_mean', ]

    model_columns = []
    for col in all_model_columns:
        if col in df.columns and col not in learn_exclude:
            if col not in model_columns:
                model_columns.append(col)

    # маркировка сайта  -1 - женский,  1 - мужской, 0 - нейтральный
    # df['url_m'] = np.sign(df.url_m_1_sum - df.url_m_0_sum)
    # df['url_u'] = np.sign(df.url_u_1_sum - df.url_u_0_sum)
    # df['url_a'] = np.sign(df.url_a_1_sum - df.url_a_0_sum)

    # формирование списка колонок с категориями и
    # добавление маркеров в категории если они присутствуют
    category_columns = []
    for col in all_categories:
        if col in df.columns and col not in learn_exclude:
            category_columns.append(col)
            if col not in model_columns:
                model_columns.append(col)

    df = df[model_columns]

    print('Обучаюсь на колонках:', model_columns)
    print('Категорийные колонки:', category_columns)
    print('Исключенные колонки:', learn_exclude)

    for col in category_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)

    train_columns = df.columns[2:]
    train_dtypes = df.dtypes[2:]

    # url_usr_emb_factors, region_usr_emb_factors, city_usr_emb_factors,
    # url_male_emb_factors
    # без ембендингов по городам GINI по полу 0.710
    # name_emb_factors = (url_usr_emb_factors, region_usr_emb_factors,
    #                     url_male_emb_factors)
    # name_emb_factors = (url_usr_emb_factors, region_usr_emb_factors)
    for emb_file in name_emb_factors:
        usr_emb = pd.read_feather(emb_file)
        df = df.merge(usr_emb, how='left', on='user_id')

    train = df.drop(['user_id', 'age', 'is_male'], axis=1)
    target = df['age']

    X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                          test_size=0.25,
                                                          shuffle=True,
                                                          random_state=SEED,
                                                          stratify=target
                                                          # stratify = 0.723963
                                                          )

    clf = CatBoostClassifier(cat_features=category_columns,
                             auto_class_weights='Balanced',
                             loss_function='MultiClass',
                             eval_metric='TotalF1',
                             early_stopping_rounds=30,
                             random_seed=SEED
                             )
    clf.fit(X_train, y_train, verbose=20, cat_features=category_columns)

    f1w = m.f1_score(y_valid, clf.predict(X_valid), average='weighted')
    print(f'Weighted F1-score = {f1w:.6f}')

    print_time(start_time)

    comment = clf.get_params()

    max_num += 1
    with open(file_logs, mode='a') as log:
        log.write(f'{max_num};{als_factors};'
                  f'{", ".join(als_columns_male)};{url_male_als_factors};'
                  f'{f1w:.6f};{2 * f1w:.6f};'
                  f'{[file.name for file in name_emb_factors]};'
                  f'{model_columns};{category_columns};{comment}\n')

    if build_model:
        # обучение на всей тренировочной выборке
        clf.fit(train, target, verbose=100, cat_features=category_columns)

        f1w = m.f1_score(target, clf.predict(train), average='weighted')
        print(f'Weighted F1-score = {f1w:.6f}')
        print(m.classification_report(target, clf.predict(train),
                                      target_names=['19-25', '25-34', '35-44',
                                                    '45-54', '55-65', '65+']))

        file_submit = WORK_PATH.joinpath('submission.feather')
        submit = pd.read_feather(file_submit)

        test = pd.read_feather(df_test_users)
        test = test[train_columns]

        # установить тип данных как в тренировочном ДФ
        for col in test.columns:
            test[col] = test[col].astype(train_dtypes[col])

        #  добавление эмбендингов
        for emb_file in name_emb_factors:
            usr_emb = pd.read_feather(emb_file)
            test = test.merge(usr_emb, how='left', on='user_id')

        test['age'] = clf.predict(test)

        submit_prefix = 'cb_'
        submit_csv = f'{submit_prefix}submit_age_{max_num:03}.csv'
        file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)

        submit_columns = ['user_id', 'age']
        submit = test[submit_columns]
        submit.to_csv(file_submit_csv, index=False)
