import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)

df = pd.read_csv("wage_prediction/hitters-230817-080430.csv")
df.head()
df.shape
df.info()
df.describe().T

# Kolon isimleri büyük harf yapılıyor.
df.columns = [col.upper() for col in df.columns]


# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"length of cat_cols: {len(cat_cols)}")
    print(f"length of num_cols: {len(num_cols)}")
    print(f"length of cat_but_car: {len(cat_but_car)}")
    print(f"length of num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
target = "SALARY"  # Hedef değişken seçildi.
for col in cat_cols:
    print(col, df[col].value_counts())


def num_summary(dataframe, numerical_cols):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_cols].describe(quantiles).T)


for col in num_cols:
    num_summary(df, col)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################")


for col in cat_cols:
    cat_summary(df, col)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, target, col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n")


for col in num_cols:
    target_summary_with_num(df, target, col)


# Boş olan SALARY'ler KNNImputer ile dolduruluyor.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True, dtype=int)
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
df["SALARY_IMPUTED_KNN"] = dff[["SALARY"]]
df["SALARY"] = df["SALARY_IMPUTED_KNN"]
df.drop(["SALARY_IMPUTED_KNN"], axis=1, inplace=True)


# Aykırı değer tespiti ve limitlerle doldurulması.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))
for col in num_cols:
    replace_with_thresholds(df, col)
for col in num_cols:
    print(col, check_outlier(df, col))
# Aykırı değer kalmadı.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
for col in num_cols:
    num_summary(df, col)

for col in cat_cols:
    cat_summary(df, col)


# Label Encoder ile kategorik değişkenlerin dönüşümü
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "SALARY" not in col]
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)
df[num_cols].head()

# Model
y = df["SALARY"]
X = df.drop(["SALARY"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)  # 1990'lı olduğum için 90 :)
light_gbm_model = LGBMRegressor().fit(X_train, y_train)
light_gbm_params = [{"learning_rate": [0.001, 0.01, 0.1], "n_estimators": [300, 500, 700]}]
gs = GridSearchCV(light_gbm_model, param_grid=light_gbm_params, cv=5, n_jobs=-1).fit(X_train, y_train)
gs.best_params_  # {'learning_rate': 0.01, 'n_estimators': 500}

light_gbm_final_model = light_gbm_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train)
light_gbm_final_model.score(X_test, y_test)  # 0.6580390929226357

X_random = X_test.sample(1, random_state=90)
y_predicted = light_gbm_final_model.predict(X_random)  # 732.67040777
y.iloc[178]  # 1000.0


reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_model.score(X_train, y_train)  # 0.6002240351646269
y_predicted = reg_model.predict(X_random)  # 938.56975986
y.iloc[178]  # 1000.0


# LinearRegression modeli LGBMRegressor modelinden daha iyi sonuç verdi.

