import pandas as pd
from pathlib import Path
from ast import literal_eval
from df_addons import df_to_excel

__import__("warnings").filterwarnings('ignore')

file_logs = Path(r'D:\python-txt\mts\gini_male.logs')
if not file_logs.is_file():
    with open(file_logs, mode='a') as log:
        log.write('num;DS;als_factors;als_column;url_male_als_factors;'
                  'ROC_AUC;GINI;name_emb_factors;model_columns;'
                  'category_columns;comment\n')
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
    # df.to_csv(file_logs, index=False, sep=';')
    max_num = df.num.max()
    print(max_num)
    print(df.columns)
    print(df)

df.sort_values(['GINI', 'als_factors', 'num'],
               ascending=[False, True, True], inplace=True)
df.rename(columns={'num': 'N', 'als_factors': 'als',
                   'url_male_als_factors': 'ncol', }, inplace=True)
# экспорт в эксель
file_xls = file_logs.with_suffix('.xlsx')
df_to_excel(df, file_xls)
