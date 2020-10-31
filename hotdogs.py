import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, \
    Normalizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import json
import requests
import math

df = pd.read_excel('DSC_2020_1/Train2.xlsx')
trainY = df['GrossSoldQuantity']
city_dic = {}


def get_city_pop(city, state):
    if city == "DUPONT":
        return 9503
    elif city in city_dic:
        return city_dic[city]
    else:
        temp = 'https://public.opendatasoft.com/api/records/1.0/search/?dataset=worldcitiespop&q=%s&sort=population&facet=region&refine.region=%s'
        cmd = temp % (city, state)
        res = requests.get(cmd)
        dct = json.loads(res.content)
        out = dct['records'][0]['fields']
        pop = out['population']
        city_dic[city] = pop

        return pop


def label_pop(row):
    return get_city_pop(row['City'], row['State'])


def get_day_status(day, month, year):
    k = day
    m = month
    C = year // 100
    Y = year - 2000
    W = (k + math.floor(2.6*m - 0.2) - 2*C + Y + Y//4 + C//4) % 7

    return W


def label_day_of_week(row):
    return get_day_status(row['Day'], row['Month'], row['Year'])


df['Population'] = df.apply(lambda row: label_pop(row), axis=1)
df['Day of Week'] = df.apply(lambda row: label_day_of_week(row), axis=1)

# Column Transformers (Unused)
# cts = [
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol', 'Carwash']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol', 'Carwash', 'Food Service']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol', 'Carwash', 'Food Service', 'City']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol', 'Carwash', 'Food Service', 'City', 'State']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (StandardScaler(), ['Population']),
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol', 'Carwash', 'Food Service', 'City', 'State']),
#         remainder='drop'
#     ),
#     make_column_transformer(
#         (StandardScaler(), ['Population']),
#         (OneHotEncoder(handle_unknown='ignore'),
#          ['StoreNumber', 'Day', 'Month', 'Year', '3HourBucket', 'Cash/Credit Site', 'EBT Site', 'Loyalty Site', 'ExtraMile Site', 'CoBrand', 'Alcohol', 'Carwash', 'Food Service', 'City', 'State', 'Day of Week']),
#         remainder='drop'
#     )
# ]

# Final column transformer
ct = make_column_transformer(
    (StandardScaler(), ['Population']),
    (OneHotEncoder(handle_unknown='ignore'),
     ['StoreNumber', 'Day', 'Month', '3HourBucket', 'Cash/Credit Site', 'ExtraMile Site', 'Carwash', 'City', 'Day of Week']),
    remainder='drop'
)

# Random Forest Regression (Unused)
# def get_cv_error(features):
#     pipeline = make_pipeline(
#         cts[(len(features)-1)],
#         RandomForestRegressor(n_estimators=10, random_state=42)
#     )
#     cv_errs = -cross_val_score(pipeline, X=df[features],
#                                y=trainY, scoring="neg_mean_squared_error", cv=10)

#     return cv_errs.mean()

# Random Forest Regressor (Unused)
# def get_cv_error(features):
#     pipeline = make_pipeline(
#         ct,
#         RandomForestRegressor(n_estimators=10, random_state=42)
#     )
#     cv_errs = -cross_val_score(pipeline, X=df[features],
#                                y=trainY, scoring="neg_mean_squared_error", cv=10)

#     return cv_errs.mean()


# KNeighborsRegressor for final variables
def get_cv_error(features):
    pipeline = make_pipeline(
        ct,
        KNeighborsRegressor(n_neighbors=70)
    )
    cv_errs = -cross_val_score(pipeline, X=df[features],
                               y=trainY, scoring="neg_mean_squared_error", cv=10)

    return cv_errs.mean()


print(get_cv_error(['StoreNumber', 'Day', 'Month', '3HourBucket', 'Cash/Credit Site',
                    'ExtraMile Site', 'Carwash', 'City', 'Population', 'Day of Week']))


# Linear Regressor (Unused)
# def get_cv_error(features):
#     pipeline = make_pipeline(
#         cts[(len(features)-1)],
#         LinearRegression()
#     )
#     cv_errs = -cross_val_score(pipeline, X=df[features], y=trainY, scoring="neg_mean_squared_error", cv=10)

#     return cv_errs.mean()

# Getting cv errors for various columns to test accuracy
# errs=pd.Series(dtype='float64')
# for features in [["StoreNumber"],
#                  ["StoreNumber", "Day"],
#                  ["StoreNumber", "Day, 'Month"],
#                  ["StoreNumber", "Day", "Month", "Year"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket"],
#                  ["StoreNumber", "Day", "Month", "Year",
#                      "3HourBucket", "Cash/Credit Site"],
#                  ["StoreNumber", "Day", "Month", "Year",
#                      "3HourBucket", "Cash/Credit Site", "EBT Site"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket",
#                      "Cash/Credit Site", "EBT Site", "Loyalty Site"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket",
#                      "Cash/Credit Site", "EBT Site", "Loyalty Site", "ExtraMile Site"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site",
#                      "EBT Site", "Loyalty Site", "ExtraMile Site", "CoBrand"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site",
#                      "EBT Site", "Loyalty Site", "ExtraMile Site", "CoBrand", "Alcohol"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site",
#                      "EBT Site", "Loyalty Site", "ExtraMile Site", "CoBrand", "Alcohol", "Carwash"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site", "EBT Site",
#                      "Loyalty Site", "ExtraMile Site", "CoBrand", "Alcohol", "Carwash", "Food Service"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site", "EBT Site",
#                      "Loyalty Site", "ExtraMile Site", "CoBrand", "Alcohol", "Carwash", "Food Service", "City"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site", "EBT Site",
#                      "Loyalty Site", "ExtraMile Site", "CoBrand", "Alcohol", "Carwash", "Food Service", "City", "State"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site", "EBT Site", "Loyalty Site",
#                      "ExtraMile Site", "CoBrand", "Alcohol", "Carwash", "Food Service", "City", "State", "Population"],
#                  ["StoreNumber", "Day", "Month", "Year", "3HourBucket", "Cash/Credit Site", "EBT Site", "Loyalty Site",
#                      "ExtraMile Site", "CoBrand", "Alcohol", "Carwash", "Food Service", "City", "State", "Population", "Day of Week"]]:
#     errs[str(features)]=get_cv_error(features)

# print(errs)
