import numpy as np
import pandas as pd

import sklearn.metrics as m
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from ast import literal_eval
from print_time import print_time, print_msg
from df_addons import df_to_excel
from mts_paths import *

__import__("warnings").filterwarnings('ignore')

start_time = print_msg('Запускаю классификатор...')

SEED = 2023
MY_DATA_PATH = WORK_PATH
LOCAL_DATA_PATH = PREDICTIONS_DIR

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


def predict_test(idx_fold, model):
    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    test['age'] = model.predict(test.drop('user_id', axis=1))

    submit_prefix = 'cb_'
    submit_csv = f'{submit_prefix}submit_age_{max_num:03}{nfld}.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)

    submit_columns = ['user_id', 'age']
    submit = test[submit_columns]
    submit.to_csv(file_submit_csv, index=False)

    f1w = m.f1_score(target, clf.predict(train), average='weighted')
    print(f'Weighted F1-score на всем train = {f1w:.6f}')
    print(m.classification_report(target, clf.predict(train),
                                  target_names=['19-25', '25-34', '35-44',
                                                '45-54', '55-65', '65+']))


# for url_male_als_factors in (*range(80, 140, 20), *range(150, 401, 50)):
# for als_factors in (97,):
als_range = (*range(80, 140, 20), *range(150, 401, 50))

ranges = ([105] + [320, 335],
          ['male_prs_1'] + ['male_avg_prs_1'] * 2)

ranges = ([*als_range], ['male_avg_prs_1'] * len(als_range))

ranges = ([100, 100],
          ['male_prs_1', 'male_avg_prs_1'])

for url_male_als_factors, column_male in zip(*ranges):

    # текущий номер итерации из лога
    max_num += 1

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
    #     name_emb_factors.append(city_usr_emb_factors)

    # количество признаков
    #     url_male_als_factors = 100
    # male_prs_1=0.730036, male_user_prs_1=0.730056, male_avg_prs_1=0.729514
    als_columns_male = ('male_prs_1', 'male_user_prs_1', 'male_avg_prs_1')
    #     als_columns_male = ('male_prs_1', 'male_user_prs_1')
    #     als_columns_male = ('male_prs_1',)
    #     als_columns_male = ('male_user_prs_1',)
    #     als_columns_male = ('male_avg_prs_1',)

    als_columns_male = (column_male,)

    for als_column in als_columns_male:
        name_a = f'url_{als_column}_emb_factors_{url_male_als_factors}.feather'
        url_male_emb_factors = WORK_PATH.joinpath(name_a)
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

    #     print(df.columns)

    # df['is_male'].value_counts()

    all_model_columns = [
        'age', 'is_male', 'user_id',
        # 'region_name', 'city_name', 'cpe_id', 'firm_id',
        'price_mean',
        'request_cnt_sum',
        'url_host_count',
        'url_host_count_distinct',
        'different_city',
        'date_count',
        'city_name_count_distinct',
        'region_name_count_distinct',
        'date_count_distinct',
        'male_prs_1',
        'male_user_prs_1',
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
                         #  'fa_u1_sum', 'fa_u2_sum', 'fa_u3_sum',
                         #  'fa_u4_sum', 'fa_u5_sum', 'fa_u6_sum',
                         ]

    # для предсказания пола - это плохо годится
    all_model_columns.extend(age_model_columns)

    # колонки с категориями
    all_categories = (
        'cpe_id',
        'firm_id',
        'region_name',
        'city_name',
        'url_m',
        'url_u',
        'url_a',
    )

    learn_exclude = [
        'fame0user_sum',
        #  'region_name',
        'price_mean',
    ]

    model_columns = []
    for col in all_model_columns:
        if col in df.columns and col not in learn_exclude:
            if col not in model_columns:
                model_columns.append(col)

    # маркировка сайта  -1 - женский,  1 - мужской, 0 - нейтральный
    #     df['url_m'] = np.sign(df.url_m_1_sum - df.url_m_0_sum)
    #     df['url_u'] = np.sign(df.url_u_1_sum - df.url_u_0_sum)
    #     df['url_a'] = np.sign(df.url_a_1_sum - df.url_a_0_sum)

    # формирование списка колонок с категориями и
    # добавление маркеров в категории если они присутствуют
    category_columns = []
    for col in all_categories:
        if col in df.columns and col not in learn_exclude:
            category_columns.append(col)
            if col not in model_columns:
                model_columns.append(col)

    df = df[model_columns]

    #     print('Обучаюсь на колонках:', model_columns)
    #     print('Категорийные колонки:', category_columns)
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

    #
    # данные для предсказаний
    test = pd.read_feather(df_test_users)
    test = test[train_columns]
    # установить тип данных как в тренировочном ДФ
    for col in test.columns:
        test[col] = test[col].astype(train_dtypes[col])
    #  добавление эмбендингов
    for emb_file in name_emb_factors:
        usr_emb = pd.read_feather(emb_file)
        test = test.merge(usr_emb, how='left', on='user_id')
    #

    X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                          test_size=0.25,
                                                          shuffle=True,
                                                          random_state=SEED,
                                                          stratify=target
                                                          )
    pool_train = Pool(data=X_train, label=y_train,
                      cat_features=category_columns)
    pool_valid = Pool(data=X_valid, label=y_valid,
                      cat_features=category_columns)

    clf = CatBoostClassifier(cat_features=category_columns,
                             auto_class_weights='Balanced',
                             loss_function='MultiClass',
                             iterations=3000,
                             # learning_rate=0.1,
                             # learning_rate=0.15,
                             task_type="GPU",
                             devices='0:1',
                             eval_metric='TotalF1',
                             early_stopping_rounds=50,
                             silent=True,
                             random_seed=SEED
                             )
    num_folds = 4
    skf = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)
    split_kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    use_grid_search = False
    use_cv_folds = True
    build_model = True
    stratified = True

    models = []

    if use_grid_search:
        grid_params = {
            'max_depth': [5, 6],
            'learning_rate': [0.13248400390148163, 0.1, 0.15],
        }
        grid_search_result = clf.grid_search(grid_params, train, target,
                                             cv=skf,
                                             stratified=True,
                                             refit=True,
                                             plot=True
                                             )
        best_params = grid_search_result['params']
        models.append(clf)
        if build_model:
            predict_test(0, clf)

    elif use_cv_folds:
        if stratified:
            skf_folds = skf.split(train, target)
        else:
            skf_folds = split_kf.split(train)

        for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
            print(f'Фолд {idx} из {num_folds}')
            train_data = Pool(data=train.iloc[train_idx],
                              label=target.iloc[train_idx],
                              cat_features=category_columns)
            valid_data = Pool(data=train.iloc[valid_idx],
                              label=target.iloc[valid_idx],
                              cat_features=category_columns)
            model = clf
            model.fit(train_data, eval_set=valid_data, use_best_model=True, verbose=100)
            models.append(model)
            if build_model:
                predict_test(idx, model)

        best_params = {'iterations': [clf.tree_count_ for clf in models]}

    else:
        clf.fit(pool_train, eval_set=pool_valid, use_best_model=True, verbose=100)

        best_params = {'iterations': clf.tree_count_}
        models.append(clf)
        if build_model:
            model = CatBoostClassifier(cat_features=category_columns,
                                       auto_class_weights='Balanced',
                                       loss_function='MultiClass',
                                       iterations=int(clf.tree_count_ * 1.1),
                                       learning_rate=clf.get_all_params()[
                                           'learning_rate'],
                                       task_type="GPU",
                                       devices='0:1',
                                       eval_metric='TotalF1',
                                       early_stopping_rounds=80,
                                       silent=True,
                                       random_seed=SEED
                                       )
            model.fit(train, target, verbose=100, cat_features=category_columns)
            predict_test(0, model)

    # print()
    # print(clf.get_all_params())
    # print()
    print('best_params:', best_params)

    f1w = 0
    for mdl in models:
        f1w += m.f1_score(y_valid, clf.predict(X_valid), average='weighted')
    f1w /= len(models)
    print(f'Weighted F1-score = {f1w:.6f}')

    model_params = clf.get_params()
    print('Параметры модели:', model_params)

    print_time(start_time)

    comment = {'stratified': stratified,
               'bestIteration': [clf.tree_count_ for clf in models],
               'clf.learning_rate': [clf.get_all_params()['learning_rate']
                                     for clf in models]}
    comment.update(models[0].get_params())

    with open(file_logs, mode='a') as log:
        log.write(f'{max_num};DS4;{als_factors};'
                  f'{", ".join(als_columns_male)};{url_male_als_factors};'
                  f'{f1w:.6f};{2 * f1w:.6f};'
                  f'{[file.name for file in name_emb_factors]};'
                  f'{model_columns};{category_columns};{comment}\n')
