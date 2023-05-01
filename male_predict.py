import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from matplotlib import rcParams

from print_time import print_time, print_msg
from mts_paths import *

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score as RAS
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier

__import__("warnings").filterwarnings("ignore")

SEED = 2023
MY_DATA_PATH = WORK_PATH

USE_CORES = os.cpu_count() - 1
PATH_FILES = os.path.dirname(os.path.abspath(__file__))
PATH_EXPORT = os.path.join(PATH_FILES, 'EXPORT')

if not os.path.exists(PATH_EXPORT):
    os.makedirs(PATH_EXPORT)
if not os.path.exists(os.path.join(PATH_EXPORT, 'predictions')):
    os.makedirs(os.path.join(PATH_EXPORT, 'predictions'))

rcParams.update({'font.size': 14})  # размер шрифта на графиках
pd.options.display.max_columns = 100
global_start_time = time.time()


def process_model(model=RandomForestClassifier(random_state=SEED),
                  params={'max_depth': [6]}, n_fold=4,
                  verbose=0, build_model=False):
    """
    Поиск лучшей модели
    :param model: модель для обучения и предсказаний
    :param params: параметры для модели
    :param n_fold: на сколько фолдов разбивать данные финальной модели
    :param verbose: = 1 - отображать процесс
    :param build_model: = True - строить модель и выгружать предсказания
    :return: параметры модели, feat_imp_df - датафрейм с фичами
    """
    # skf = StratifiedKFold(n_splits=n_fold, random_state=SEED, shuffle=True)
    skf = KFold(n_splits=n_fold, random_state=SEED, shuffle=True)
    gscv = model.grid_search(params, X_train, y_train, cv=skf,
                             # stratified=True,
                             refit=True)
    model.fit(X_train, y_train)
    best_ = gscv['params']
    print(gscv['cv_results'].keys())

    f1_train = np.array(gscv['cv_results']['train-AUC-mean']).max() * 2 - 1
    f1_valid = np.array(gscv['cv_results']['test-AUC-mean']).max() * 2 - 1

    print(f'folds={n_fold:2d}, '
          f'RAS_train={f1_train:0.7f}, '
          f'RAS_valid={f1_valid:0.7f}, '
          f'best_params={best_}')

    # построение лучшей модели
    if build_model:
        submit_prefix = 'cb_'
        submit_csv = f'{submit_prefix}submit.csv'
        file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
        start_time_cv = time.time()
        print('Обучение модели...')
        max_depth = best_['depth']

        feat_imp = model.feature_importances_
        feat_imp_df = pd.DataFrame({'features': X.columns.values,
                                    'importances': feat_imp})
        feat_imp_df.sort_values('importances', ascending=False, inplace=True)

        # params = {
        #     'n_estimators': 284,
        #     'max_depth': 4,
        #     'subsample': 0.6983901315995189,
        #     'l2_leaf_reg': 3.180242242411711,
        #     'random_strength': 1.2423130425640145,
        #     'eta': 0.04356020658096416,
        #     'min_data_in_leaf': 1,
        #     'grow_policy': 'Lossguide',
        #     'silent': True,
        #     'eval_metric': 'AUC:hints=skip_train~false'
        # }
        # model = CatBoostClassifier(cat_features=category_columns, **params)

        # Обучение модели
        model.fit(X, y)

        test['is_male'] = model.predict_proba(test)[:, 1]

        date_now = datetime.now()
        time_stamp = date_now.strftime('%y%m%d%H%M%S')

        submit_columns = ['user_id', 'is_male']
        submit = test[submit_columns]
        file_submit_csv = PREDICTIONS_DIR.joinpath(
            file_submit_csv.name.replace('.csv', f'_{time_stamp}.csv'))
        submit.to_csv(file_submit_csv, index=False)

        # сохранение результатов итерации в файл
        file_name = os.path.join(PATH_EXPORT, 'results.csv')
        if os.path.exists(file_name):
            file_df = pd.read_csv(file_name)
            file_df.time_stamp = pd.to_datetime(file_df.time_stamp,
                                                format='%y-%m-%d %H:%M:%S')
            file_df.time_stamp = file_df.time_stamp.dt.strftime(
                '%y-%m-%d %H:%M:%S')
            if 'comment' not in file_df.columns:
                file_df['comment'] = ''
        else:
            file_df = pd.DataFrame()
        time_stamp = date_now.strftime('%y-%m-%d %H:%M:%S')
        features_list = feat_imp_df.to_dict(orient='split')['data']
        comments_ = f'{data.data_comment} '
        comments_ += f'X={len(X_train)} rows, y={len(y_train)} rows'
        temp_df = pd.DataFrame({'time_stamp': time_stamp,
                                'mdl': submit_prefix[:2].upper(),
                                'max_depth': max_depth,
                                'folds': n_fold,
                                'f1_train': f1_train,
                                'f1_valid': f1_valid,
                                'best_params': [best_],
                                'features': [features_list],
                                # 'column_dummies': [processor_data.dummy],
                                'model_columns': [model_columns],
                                'category_columns': [category_columns],
                                'learn_exclude': [learn_exclude],
                                'comment': comments_
                                })

        file_df = file_df.append(temp_df)
        file_df.f1_train = file_df.f1_train.round(7)
        file_df.f1_valid = file_df.f1_valid.round(7)
        file_df.to_csv(file_name, index=False)
        file_df.name = 'results'
        # экспорт в эксель
        export_to_excel(file_df)
        print_time(start_time_cv)
        return feat_imp_df
    return [f1_train, f1_valid, n_fold, best_]


def find_depth(use_model, max_depth_values=range(5, 9), show_plot=True):
    print(use_model)
    # Подберем оптимальное значение глубины обучения дерева.
    scores = pd.DataFrame(columns=['max_depth', 'train_score', 'valid_score'])
    for max_depth in max_depth_values:
        print(f'max_depth = {max_depth}')
        find_model = use_model(random_state=SEED, silent=True,
                               cat_features=category_columns,
                               max_depth=max_depth,
                               early_stopping_rounds=30,
                               # class_weights=[1, imbalance],
                               auto_class_weights='Balanced',
                               loss_function='Logloss',
                               eval_metric='AUC:hints=skip_train~false')

        find_model.fit(X_train, y_train, cat_features=category_columns,
                       verbose=20)

        y_train_pred = find_model.predict_proba(X_train)[:, 1]
        y_valid_pred = find_model.predict_proba(X_valid)[:, 1]
        train_score = RAS(y_train, y_train_pred) * 2 - 1
        valid_score = RAS(y_valid, y_valid_pred) * 2 - 1

        print(f'\ttrain_score = {train_score:.6f}')
        print(f'\tvalid_score = {valid_score:.6f}\n')

        scores.loc[len(scores)] = [max_depth, train_score, valid_score]

    scores.max_depth = scores.max_depth.astype(int)
    scores_data = pd.melt(scores,
                          id_vars=['max_depth'],
                          value_vars=['train_score', 'valid_score'],
                          var_name='dataset_type',
                          value_name='score')
    if show_plot:
        # Визуализация
        plt.figure(figsize=(12, 7))
        sns.lineplot(x='max_depth', y='score', hue='dataset_type',
                     data=scores_data)
        plt.show()
    print(scores.sort_values('valid_score', ascending=False))
    print()
    print('Наилучший результат с параметрами:')
    print(scores.loc[scores.valid_score.idxmax()])
    print()


def export_to_excel(data_df: pd.DataFrame) -> None:
    """
    # экспорт датафрема в эксель
    Convert the dataframe to an XlsxWriter Excel object.
    Note that we turn off default header and skip one row to allow us
    to insert a user defined header.
    :param data_df: dataframe
    :return: None
    """
    name_data = data_df.name
    file_xls = os.path.join(PATH_EXPORT, f'{name_data}.xlsx')
    writer = pd.ExcelWriter(file_xls, engine='xlsxwriter')
    data_df.to_excel(writer, sheet_name=name_data, startrow=1,
                     header=False, index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[name_data]
    # Add a header format.
    font_name = 'Arial'
    header_format = workbook.add_format({
        'font_name': font_name,
        'font_size': 10,
        'bold': True,
        'text_wrap': True,
        'align': 'center',
        'valign': 'center',
        'border': 1})
    # Write the column headers with the defined format.
    worksheet.freeze_panes(1, 0)
    for col_num, value in enumerate(data_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    cell_format = workbook.add_format()
    cell_format.set_font_name(font_name)
    cell_format.set_font_size(12)
    nums_format = workbook.add_format({'num_format': '#0.0000000'})
    nums_format.set_font_name(font_name)
    nums_format.set_font_size(12)
    for num, value in enumerate(data_df.columns.values):
        if value == 'time_stamp':
            width = 19
        elif value in ('mdl', 'folds'):
            width = 8
        elif value in ('max_depth', 'f1_train', 'f1_valid',
                       'r2_train', 'r2_valid',
                       'ras_train', 'ras_valid'):
            width = 14
        else:
            width = 32
        if value in ('f1_train', 'f1_valid', 'r2_train', 'r2_valid',
                     'ras_train', 'ras_valid'):
            worksheet.set_column(num, num, width, nums_format)
        else:
            worksheet.set_column(num, num, width, cell_format)
    worksheet.autofilter(0, 0, len(data_df) - 1, len(data_df) - 1)
    writer.save()
    # End excel save


total_time = time.time()

SEED = 2023
MY_DATA_PATH = WORK_PATH

data = dict(data_comment='')

name_emb_factors = []
# количество признаков
als_factors = 97
# als_factors = 104
# als_factors = 100
# als_factors = 150
name_csv = f'url_usr_emb_factors_{als_factors}.csv'
url_usr_emb_factors = MY_DATA_PATH.joinpath(name_csv.replace('.csv',
                                                             '.feather'))
print(url_usr_emb_factors)
name_emb_factors.append(url_usr_emb_factors)

# количество признаков
region_als_factors = 20
name_csv = f'region_usr_emb_factors_{region_als_factors}.csv'
region_usr_emb_factors = MY_DATA_PATH.joinpath(name_csv.replace('.csv',
                                                                '.feather'))
print(region_usr_emb_factors)
name_emb_factors.append(region_usr_emb_factors)

# количество признаков
city_als_factors = 30
name_csv = f'city_usr_emb_factors_{city_als_factors}.csv'
city_usr_emb_factors = MY_DATA_PATH.joinpath(name_csv.replace('.csv',
                                                              '.feather'))
# print(city_usr_emb_factors)
# name_emb_factors.append(city_usr_emb_factors)

# количество признаков
url_male_als_factors = 100
# male_prs_1=0.730036, male_user_prs_1=0.730056, male_avg_prs_1=0.729514
als_columns_male = ('male_prs_1', 'male_user_prs_1', 'male_avg_prs_1')
# als_columns_male = ('male_prs_1', 'male_user_prs_1')
# als_columns_male = ('male_user_prs_1',)
als_columns_male = ('male_avg_prs_1',)
for als_column in als_columns_male:
    name_a = f'url_{als_column}_emb_factors_{url_male_als_factors}.feather'
    url_male_emb_factors = MY_DATA_PATH.joinpath(name_a)
    print(url_male_emb_factors)
    name_emb_factors.append(url_male_emb_factors)
#

# test = pd.read_feather(df_test_users)
test = pd.read_feather(file_submit)  # заглушка для поиска наилучшей модели

targets = pd.read_feather(file_users)
df = pd.read_feather(df_train_users)
df = targets.merge(df, how='left', on='user_id')
df.fillna(0, inplace=True)
# удалить колонку index
for col in ('index', 'url_host_count_distinct_y'):
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

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

for emb_file in name_emb_factors:
    usr_emb = pd.read_feather(emb_file)
    df = df.merge(usr_emb, how='left', on='user_id')

train = df.drop(['user_id', 'age', 'is_male'], axis=1)
target = df['is_male']
# test = test[model_columns]

X = train
y = target

txt = ('Размер ', ' пропусков ')
print(f'{txt[0]}трейна: {train.shape}{txt[1]}{train.isna().sum().sum()}')
print(f'{txt[0]}теста: {test.shape}{txt[1]}{test.isna().sum().sum()}')

# было test_size=0.25
X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                      test_size=0.25,
                                                      shuffle=True,
                                                      random_state=SEED,
                                                      # stratify=target
                                                      )
print()
print(f'{txt[0]}X_train: {X_train.shape}{txt[1]}'
      f'{X_train.isna().sum().sum()}')
print(f'{txt[0]}X_valid: {X_valid.shape}{txt[1]}'
      f'{X_valid.isna().sum().sum()}')

# X_train, X_valid, y_train, y_valid = X, X, y, y

imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f'Дисбаланс классов = {imbalance}')

# find_depth(CatBoostClassifier, max_depth_values=range(5, 11))
#    max_depth  train_score  valid_score
# 1          6     0.810699     0.729837
# 2          7     0.849588     0.728472
# 0          5     0.781493     0.727756
# 3          8     0.898785     0.727400


# настройки для первого приближения: поиск глубины деревьев
# и количества фолдов
models = []
for fold in range(4, 5):
    for depth in range(8, 12):
        param = {'max_depth': [depth]}
        # определение моделей
        mdl = CatBoostClassifier(random_state=SEED, silent=True,
                                 early_stopping_rounds=30,
                                 cat_features=category_columns,
                                 # class_weights=[1, imbalance],
                                 auto_class_weights='Balanced',
                                 loss_function='Logloss',
                                 eval_metric='AUC:hints=skip_train~false')
        result = process_model(mdl, params=param, n_fold=fold)
        models.append(result)

models.sort(key=lambda x: (-x[1], x[2]))
print()
for elem in models:
    print(elem)

# 0.8437483213620247	0.7188143510030487	4	{'depth':9}
# 0.7897304981980717	0.7184473580057866	7	{'depth':8}
# 0.9022726798856942	0.7181084715454793	4	{'depth':10}
# 0.7990665673162787	0.7180516979261511	4	{'depth':8}
# 0.7990665673162787	0.7180516979261511	4	{'depth':8}
# 0.7916883840262603	0.7180356585905916	6	{'depth':8}
# 0.7946830856084759	0.7179485011914679	5	{'depth':8}
# 0.9613212205327983	0.7175714159630415	4	{'depth':11}
# 0.7614934714711510	0.7167048254186077	7	{'depth':7}
# 0.7626360325289416	0.7163990952043642	6	{'depth':7}
# 0.7645856108005067	0.7162733150179625	5	{'depth':7}
# 0.7674535456165619	0.7158902915123679	4	{'depth':7}
# 0.7440358089529262	0.7138606145649216	5	{'depth':6}
# 0.7420613205206359	0.7138008118748469	7	{'depth':6}
# 0.7429614137737424	0.7136085677284543	6	{'depth':6}
# 0.7457020970649113	0.7134123164876374	4	{'depth':6}
# 0.7303552144207426	0.7099595095898499	4	{'depth':5}
# 0.7278921997079653	0.7099461418821309	7	{'depth':5}
# 0.7290969361259374	0.7097488518718413	5	{'depth':5}
# 0.7284264836386318	0.7096368478864266	6	{'depth':5}


X_train, y_train = X, y

imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f'Дисбаланс классов = {imbalance}')

# определение моделей
mdl = CatBoostClassifier(random_state=SEED, silent=True,
                         cat_features=category_columns,
                         # class_weights=[1, imbalance],
                         auto_class_weights='Balanced',
                         early_stopping_rounds=50,
                         loss_function='Logloss',
                         eval_metric='AUC:hints=skip_train~false')

# немного потюним и результат грузим на Kaggle
feat_imp_df_ = pd.DataFrame()

# CB,4,6,0.873188,0.8490922,"{'min_data_in_leaf': 2, 'depth': 4,
# 'iterations': 370, 'subsample': 0.7, 'random_strength': 1.25,
# 'l2_leaf_reg': 3.2, 'grow_policy': 'Lossguide'}" mean_total_spent

f_params = {
    'n_estimators': range(365, 381, 1),
    # 'n_estimators': [370],
    # 'max_depth': range(2, 7),
    'max_depth': [4],
    # 'learning_rate': [.01, .05],
    # 'learning_rate': np.linspace(0.03, 0.04, 11),
    # 'learning_rate': [.03],
    # 'min_data_in_leaf': range(2, 4, 1),
    'min_data_in_leaf': [2],
    'subsample': np.linspace(0.7, 0.8, 11),
    # 'subsample': [0.75],
    # 'random_strength': np.linspace(1.24, 1.26, 5),
    'random_strength': [1.255],
    # 'l2_leaf_reg': np.linspace(3.1, 3.2, 11),
    'l2_leaf_reg': [3.15],
    'grow_policy': ['Lossguide'],
}
# раскоментарить эту строку для расчета
# for n in range(6, 7):
#     mdl = CatBoostClassifier(random_state=SEED, silent=True,
#                              cat_features=category_columns,
#                              # class_weights={0: 1, 1: imbalance},
#                              auto_class_weights='Balanced',
#                              loss_function='Logloss',
#                              eval_metric='AUC:hints=skip_train~false')
#
#     _, feat_imp_df_ = process_model(mdl, params=f_params, n_fold=n,
#                                     verbose=20, build_model=True)
#     print(feat_imp_df_)

# model = CatBoostClassifier(silent=True, random_state=SEED,
#                            class_weights=[1, imbalance],
#                            eval_metric='F1')
# model.fit(X_train, y_train)
# evaluate_preds(model, X_train, X_valid, y_train, y_valid)

# params = {
#     # 'iterations': [5, 7, 10, 20, 30, 50, 100],
#     # 'max_depth': [3, 5, 7, 10],
#     'max_depth': range(5, 6),
#     'iterations': range(50, 201, 10),
#     # 'learning_rate': [.005, .01, .025, .05]
# }
# поставил общий дисбаланс попробовать это грузануть
# imbalance = y.value_counts()[0] / y.value_counts()[1]

# model = CatBoostClassifier(silent=True, random_state=SEED,
#                            class_weights=[1, imbalance],
#                            cat_features=category_columns,
#                            eval_metric='F1',
#                            early_stopping_rounds=50, )
#
# feat_imp_df_ = process_model(model, params=params, fold_single=5,
#                              verbose=20, build_model=True)
# print(feat_imp_df_)

# skf = StratifiedKFold(n_splits=5, random_state=SEED,
#                       shuffle=True)
# search_cv = model.grid_search(params, X_train, y_train, cv=skf,
#                               stratified=True, refit=True)
# for key, value in model.get_all_params().items():
#     print(f'{key} : {value}'.format(key, value))
#
# a = model.get_all_params()['iterations']
# b = model.get_all_params()['depth']

print_time(total_time)
