import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import xgboost
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV,  train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from datetime import date, datetime
from sklearn.preprocessing import StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.3f' % x)
pd.set_option("display.width", 500)


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    num_cols_not_cat = ["BSMTFINSF2", "ENCLOSEDPORCH", "3SSNPORCH", "SCREENPORCH", "POOLAREA", "MISCVAL"]
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = [col for col in cat_cols if col not in num_cols_not_cat]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"length of cat_cols: {len(cat_cols)}")
    print(f"length of num_cols: {len(num_cols)}")
    print(f"length of cat_but_car: {len(cat_but_car)}")
    print(f"length of num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car, num_but_cat


def num_summary(dataframe):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    for col in num_cols:
        print(dataframe[col].describe(quantiles).T)


def cat_summary(dataframe):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("############################################################")


def target_summary_with_cat(dataframe):
    target = "SALEPRICE"
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    for col_name in cat_cols:
        print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col_name)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe):
    target = "SALEPRICE"
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    for col_name in num_cols:
        print(dataframe.groupby(target).agg({col_name: "mean"}), end="\n\n")


# Bu veri setinde fiyatı belirleyecek kimi önemli değerler aykırı değerin çok üstünde diye atılıyordu.
# Bu yüzden, q1'in değerini 0.004'e, q3'ün değerini 0.996'ya çektim.
def outlier_thresholds(dataframe, col_name, q1=0.004, q3=(1 - 0.004)):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * interquantile_range)
    low_limit = quartile1 - (1.5 * interquantile_range)
    return low_limit, up_limit


def check_outlier(dataframe):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    for col in num_cols:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        result = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None)
        print(col, result)


def replace_with_thresholds(dataframe):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    line = f"{datetime.now()} - Outlier values will be replaced by threshold."
    log_lines.append(line)
    for col in num_cols:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        dataframe.loc[(dataframe[col] < low_limit), col] = low_limit
        dataframe.loc[(dataframe[col] > up_limit), col] = up_limit
    line = f"{datetime.now()} - Outlier values are replaced by threshold."
    log_lines.append(line)
    return dataframe


# Boş olan nümerik değişkenler KNNImputer ile dolduruluyor.
def fill_empty_values_knn_imputer(dataframe, target="SALEPRICE"):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    dff = pd.get_dummies(dataframe[cat_cols + num_cols], drop_first=True, dtype=int)
    imputer = KNNImputer(n_neighbors=5)
    dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
    remaining_num_cols = [col for col in num_cols if target not in col]
    for col_name in remaining_num_cols:
        dataframe[f"{col_name}_KNN"] = dff[[col_name]]
        dataframe[col_name] = dataframe[f"{col_name}_KNN"]
        dataframe.drop([f"{col_name}_KNN"], axis=1, inplace=True)
    return dataframe


def fill_empty_values_with_mode(dataframe):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    for col_name in cat_cols:
        dataframe[col_name] = dataframe[col_name].fillna(dataframe[col_name].mode()[0])
    return dataframe


def report_and_fill_empty_inputs(dataframe):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    dataframe = report_and_fill_empty_num_cols(dataframe, num_cols)
    dataframe = report_and_fill_empty_cat_cols(dataframe, cat_cols)
    return dataframe


def report_and_fill_empty_num_cols(dataframe, num_cols, target="SALEPRICE"):
    remaining_num_cols = [col for col in num_cols if target not in col]
    empty_count = dataframe[remaining_num_cols].isnull().sum().sum()
    line = f"{datetime.now()} - Count of empty numerical values: {empty_count}"
    log_lines.append(line)
    print(line)
    if empty_count == 0:
        return dataframe
    dataframe = fill_empty_values_knn_imputer(dataframe)
    empty_count = dataframe[remaining_num_cols].isnull().sum().sum()
    line = f"{datetime.now()} - After filling empty numerical values via KNN Imputer, remaining count of empty numerical values: {empty_count}"
    log_lines.append(line)
    print(line)
    if empty_count == 0:
        return dataframe
    dataframe = dataframe.apply(lambda x: x.fillna(x.mean()) if "SALEPRICE" not in x & x.dtype != "O" else x, axis=0)
    empty_count = dataframe[remaining_num_cols].isnull().sum().sum()
    line = f"{datetime.now()} - After filling empty numerical values with mean values, remaining count of empty numerical values: {empty_count}"
    log_lines.append(line)
    print(line)
    return dataframe


def report_and_fill_empty_cat_cols(dataframe, cat_cols):
    empty_count = dataframe[cat_cols].isnull().sum().sum()
    line = f"{datetime.now()} - Count of empty categorical values: {empty_count}"
    log_lines.append(line)
    print(line)
    if empty_count == 0:
        return dataframe
    dataframe = fill_empty_values_with_mode(dataframe)
    empty_count = dataframe[cat_cols].isnull().sum().sum()
    line = f"{datetime.now()} - After filling empty categorical values with mode values, remaining empty categorical values: {empty_count}"
    log_lines.append(line)
    print(line)
    return dataframe


def rare_analyzer(dataframe):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    target = "SALEPRICE"
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df


def generate_new_features(dataframe):
    """
    Yeni özellikler üretip girdide verilen Dataframe'e ekleyip döner.
    Yeni özellikleri:
        1. Konutun satıldığı yılda hangi çeyrekte satıldığı
        2. Toplam veranda alanına göre kategori
        3. Evin yaşına göre kategori
        4. Eve yapılan eklemenin/yenilemenin üzerinden kaç yıl geçtiğine göre kategori
        5. Garaj alanına göre kategori
        6. Arsa alanına göre kategori
        7. İnşası bitirilmiş olan zemin kat alanına göre kategori
        8. İnşası bitirilmiş olan 1.kat alanına göre kategori
        9. İnşası bitirilmiş olan 2.kat alanına göre kategori
        10. Havuz alanına göre kategori
    :param dataframe: tipi : pandas.Dataframe
    :return: dataframe: tipi: pandas.Dataframe
    """
    dataframe.loc[dataframe["MOSOLD"] < 4, "NEW_QUARTER_SOLD"] = "Q1"
    dataframe.loc[((dataframe["MOSOLD"] >= 4) & (dataframe["MOSOLD"] < 7)), "NEW_QUARTER_SOLD"] = "Q2"
    dataframe.loc[((dataframe["MOSOLD"] >= 7) & (dataframe["MOSOLD"] < 10)), "NEW_QUARTER_SOLD"] = "Q3"
    dataframe.loc[dataframe["MOSOLD"] >= 10, "NEW_QUARTER_SOLD"] = "Q4"

    dataframe["TOTAL_PORCH_AREA"] = dataframe["OPENPORCHSF"] + dataframe["3SSNPORCH"] + dataframe["ENCLOSEDPORCH"] + \
                                    dataframe["SCREENPORCH"]
    dataframe.loc[dataframe["TOTAL_PORCH_AREA"] == 0, "NEW_TOTAL_PORCH_AREA_CAT"] = "NONE"
    dataframe.loc[((dataframe["TOTAL_PORCH_AREA"] > 0) & (
                dataframe["TOTAL_PORCH_AREA"] < 20)), "NEW_TOTAL_PORCH_AREA_CAT"] = "SMALL"
    dataframe.loc[((dataframe["TOTAL_PORCH_AREA"] >= 20) & (
                dataframe["TOTAL_PORCH_AREA"] < 30)), "NEW_TOTAL_PORCH_AREA_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["TOTAL_PORCH_AREA"] >= 30) & (
                dataframe["TOTAL_PORCH_AREA"] < 50)), "NEW_TOTAL_PORCH_AREA_CAT"] = "LARGE"
    dataframe.loc[((dataframe["TOTAL_PORCH_AREA"] >= 50) & (
                dataframe["TOTAL_PORCH_AREA"] < 80)), "NEW_TOTAL_PORCH_AREA_CAT"] = "XLARGE"
    dataframe.loc[((dataframe["TOTAL_PORCH_AREA"] >= 80) & (
                dataframe["TOTAL_PORCH_AREA"] < 130)), "NEW_TOTAL_PORCH_AREA_CAT"] = "LUXURY"
    dataframe.loc[(dataframe["TOTAL_PORCH_AREA"] >= 130), "NEW_TOTAL_PORCH_AREA_CAT"] = "XLUXURY"

    dataframe["HOUSE_AGE"] = (date.today().year - dataframe["YEARBUILT"])
    dataframe.loc[dataframe["HOUSE_AGE"] < 20, "NEW_HOUSE_AGE_CAT"] = "NEW"
    dataframe.loc[((dataframe["HOUSE_AGE"] >= 20) & (dataframe["HOUSE_AGE"] < 50)), "NEW_HOUSE_AGE_CAT"] = "MIDDLEAGED"
    dataframe.loc[((dataframe["HOUSE_AGE"] >= 50) & (dataframe["HOUSE_AGE"] < 65)), "NEW_HOUSE_AGE_CAT"] = "OLD"
    dataframe.loc[((dataframe["HOUSE_AGE"] >= 65) & (dataframe["HOUSE_AGE"] < 80)), "NEW_HOUSE_AGE_CAT"] = "VERYOLD"
    dataframe.loc[dataframe["HOUSE_AGE"] > 80, "NEW_HOUSE_AGE_CAT"] = "ANCIENT"

    dataframe["HOUSE_REMOD_AGE"] = (date.today().year - dataframe["YEARREMODADD"])
    dataframe.loc[dataframe["HOUSE_REMOD_AGE"] < 20, "NEW_HOUSE_REMOD_AGE_CAT"] = "NEW"
    dataframe.loc[((dataframe["HOUSE_REMOD_AGE"] >= 20) & (
                dataframe["HOUSE_REMOD_AGE"] < 50)), "NEW_HOUSE_REMOD_AGE_CAT"] = "MIDDLEAGED"
    dataframe.loc[
        ((dataframe["HOUSE_REMOD_AGE"] >= 50) & (dataframe["HOUSE_REMOD_AGE"] < 65)), "NEW_HOUSE_REMOD_AGE_CAT"] = "OLD"
    dataframe.loc[((dataframe["HOUSE_REMOD_AGE"] >= 65) & (
                dataframe["HOUSE_REMOD_AGE"] < 80)), "NEW_HOUSE_REMOD_AGE_CAT"] = "VERYOLD"
    dataframe.loc[dataframe["HOUSE_REMOD_AGE"] > 80, "NEW_HOUSE_REMOD_AGE_CAT"] = "ANCIENT"

    dataframe.loc[dataframe["GARAGEAREA"] == 0, "NEW_GARAGEAREA_CAT"] = "NONE"
    dataframe.loc[((dataframe["GARAGEAREA"] > 0) & (dataframe["GARAGEAREA"] < 240)), "NEW_GARAGEAREA_CAT"] = "VERYSMALL"
    dataframe.loc[((dataframe["GARAGEAREA"] >= 240) & (dataframe["GARAGEAREA"] < 384)), "NEW_GARAGEAREA_CAT"] = "SMALL"
    dataframe.loc[
        ((dataframe["GARAGEAREA"] >= 384) & (dataframe["GARAGEAREA"] < 480)), "NEW_GARAGEAREA_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["GARAGEAREA"] >= 480) & (dataframe["GARAGEAREA"] < 560)), "NEW_GARAGEAREA_CAT"] = "LARGE"
    dataframe.loc[((dataframe["GARAGEAREA"] >= 560) & (dataframe["GARAGEAREA"] < 720)), "NEW_GARAGEAREA_CAT"] = "XLARGE"
    dataframe.loc[(dataframe["GARAGEAREA"] >= 720), "NEW_GARAGEAREA_CAT"] = "LUXURY"

    dataframe.loc[((dataframe["LOTAREA"] >= 1340) & (dataframe["LOTAREA"] < 3500)), "NEW_LOTAREA_CAT"] = "VERYSMALL"
    dataframe.loc[((dataframe["LOTAREA"] >= 3500) & (dataframe["LOTAREA"] < 8000)), "NEW_LOTAREA_CAT"] = "SMALL"
    dataframe.loc[((dataframe["LOTAREA"] >= 8000) & (dataframe["LOTAREA"] < 10200)), "NEW_LOTAREA_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["LOTAREA"] >= 10200) & (dataframe["LOTAREA"] < 11050)), "NEW_LOTAREA_CAT"] = "LARGE"
    dataframe.loc[((dataframe["LOTAREA"] >= 11050) & (dataframe["LOTAREA"] < 14300)), "NEW_LOTAREA_CAT"] = "XLARGE"
    dataframe.loc[(dataframe["LOTAREA"] >= 14300), "NEW_LOTAREA_CAT"] = "LUXURY"

    dataframe["BSMT_FINSF"] = dataframe["BSMTFINSF1"] + dataframe["BSMTFINSF2"]
    dataframe.loc[dataframe["BSMT_FINSF"] == 0, "NEW_BSMT_FINSF_CAT"] = "NONE"
    dataframe.loc[((dataframe["BSMT_FINSF"] >= 0) & (dataframe["BSMT_FINSF"] < 218)), "NEW_BSMT_FINSF_CAT"] = "SMALL"
    dataframe.loc[
        ((dataframe["BSMT_FINSF"] >= 218) & (dataframe["BSMT_FINSF"] < 450)), "NEW_BSMT_FINSF_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["BSMT_FINSF"] >= 450) & (dataframe["BSMT_FINSF"] < 655)), "NEW_BSMT_FINSF_CAT"] = "LARGE"
    dataframe.loc[((dataframe["BSMT_FINSF"] >= 655) & (dataframe["BSMT_FINSF"] < 930)), "NEW_BSMT_FINSF_CAT"] = "XLARGE"
    dataframe.loc[(dataframe["BSMT_FINSF"] >= 930), "NEW_BSMT_FINSF_CAT"] = "LUXURY"

    dataframe.loc[
        ((dataframe["TOTALBSMTSF"] >= 29.5) & (dataframe["TOTALBSMTSF"] < 519)), "NEW_TOTALBSMTSF_CAT"] = "VERYSMALL"
    dataframe.loc[
        ((dataframe["TOTALBSMTSF"] >= 519) & (dataframe["TOTALBSMTSF"] < 910)), "NEW_TOTALBSMTSF_CAT"] = "SMALL"
    dataframe.loc[
        ((dataframe["TOTALBSMTSF"] >= 910) & (dataframe["TOTALBSMTSF"] < 1089)), "NEW_TOTALBSMTSF_CAT"] = "MODERATE"
    dataframe.loc[
        ((dataframe["TOTALBSMTSF"] >= 1089) & (dataframe["TOTALBSMTSF"] < 1300)), "NEW_TOTALBSMTSF_CAT"] = "LARGE"
    dataframe.loc[
        ((dataframe["TOTALBSMTSF"] >= 1300) & (dataframe["TOTALBSMTSF"] < 1550)), "NEW_TOTALBSMTSF_CAT"] = "XLARGE"
    dataframe.loc[(dataframe["TOTALBSMTSF"] >= 1550), "NEW_TOTALBSMTSF_CAT"] = "LUXURY"

    dataframe.loc[((dataframe["1STFLRSF"] >= 334) & (dataframe["1STFLRSF"] < 915)), "NEW_1STFLRSF_CAT"] = "SMALL"
    dataframe.loc[((dataframe["1STFLRSF"] >= 915) & (dataframe["1STFLRSF"] < 1200)), "NEW_1STFLRSF_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["1STFLRSF"] >= 1200) & (dataframe["1STFLRSF"] < 1314)), "NEW_1STFLRSF_CAT"] = "LARGE"
    dataframe.loc[((dataframe["1STFLRSF"] >= 1314) & (dataframe["1STFLRSF"] < 1500)), "NEW_1STFLRSF_CAT"] = "XLARGE"
    dataframe.loc[(dataframe["1STFLRSF"] >= 1500), "NEW_1STFLRSF_CAT"] = "LUXURY"

    dataframe.loc[dataframe["2NDFLRSF"] == 0, "NEW_2NDFLRSF_CAT"] = "NONE"
    dataframe.loc[((dataframe["2NDFLRSF"] >= 0) & (dataframe["2NDFLRSF"] < 455)), "NEW_2NDFLRSF_CAT"] = "SMALL"
    dataframe.loc[((dataframe["2NDFLRSF"] >= 455) & (dataframe["2NDFLRSF"] < 672)), "NEW_2NDFLRSF_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["2NDFLRSF"] >= 672) & (dataframe["2NDFLRSF"] < 796)), "NEW_2NDFLRSF_CAT"] = "LARGE"
    dataframe.loc[((dataframe["2NDFLRSF"] >= 796) & (dataframe["2NDFLRSF"] < 954)), "NEW_2NDFLRSF_CAT"] = "XLARGE"
    dataframe.loc[(dataframe["2NDFLRSF"] >= 954), "NEW_2NDFLRSF_CAT"] = "LUXURY"

    dataframe.loc[dataframe["POOLAREA"] == 0, "NEW_POOLAREA_CAT"] = "NONE"
    dataframe.loc[((dataframe["POOLAREA"] >= 0) & (dataframe["POOLAREA"] < 200)), "NEW_POOLAREA_CAT"] = "SMALL"
    dataframe.loc[((dataframe["POOLAREA"] >= 200) & (dataframe["POOLAREA"] < 480)), "NEW_POOLAREA_CAT"] = "MODERATE"
    dataframe.loc[((dataframe["POOLAREA"] >= 480) & (dataframe["POOLAREA"] < 550)), "NEW_POOLAREA_CAT"] = "LARGE"
    dataframe.loc[(dataframe["POOLAREA"] >= 550), "NEW_POOLAREA_CAT"] = "XLARGE"

    return dataframe


def one_hot_encoder(dataframe, drop_first=True):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe, car_th=25)
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first, dtype=int)
    return dataframe


def create_lgbm_regressor_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_lgbm_regressor_model_get_score has started."
    log_lines.append(line)
    print(line)
    # Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse) almayı unutmayınız.
    y_train_log = np.log(y_train)
    light_gbm_model = LGBMRegressor(verbose=-1).fit(X_train, y_train_log)
    # Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
    light_gbm_params = [{"learning_rate": [0.001, 0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                        "max_depth": [5, 10, 15, 20, 25, None]}]
    gs = GridSearchCV(light_gbm_model, param_grid=light_gbm_params, cv=5, n_jobs=-1).fit(X_train, y_train_log)
    line = f"{datetime.now()} - Best parameters for model LGBMRegressor: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    light_gbm_final_model = light_gbm_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train_log)
    y_pred_log = light_gbm_final_model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance(light_gbm_final_model, "LGBMRegressor", X_train, save=True)
    line = f"{datetime.now()} - create_lgbm_regressor_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return light_gbm_final_model, score


def create_gradient_boost_regression_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_gradient_boost_regression_model_get_score has started."
    log_lines.append(line)
    print(line)
    # Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse) almayı unutmayınız.
    y_train_log = np.log(y_train)

    grad_boost_reg_model = GradientBoostingRegressor().fit(X_train, y_train_log)
    # Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
    grad_boost_reg_params = {"learning_rate": [0.001, 0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                             "max_depth": [5, 10, 15, 20, 25, None], "subsample": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
    gs = GridSearchCV(grad_boost_reg_model, param_grid=grad_boost_reg_params, cv=5, n_jobs=-1).fit(X_train, y_train_log)
    line = f"{datetime.now()} - Best parameters for Gradient Boosting Regressor model: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    grad_boost_reg_final_model = grad_boost_reg_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train_log)
    y_pred_log = grad_boost_reg_model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance(grad_boost_reg_final_model, "GradientBoostingRegressor", X_train, save=True)
    line = f"{datetime.now()} - create_gradient_boost_regression_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return grad_boost_reg_final_model, score


def create_xgb_regressor_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_xgb_regressor_model_get_score has started."
    log_lines.append(line)
    print(line)
    # Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse) almayı unutmayınız.
    y_train_log = np.log(y_train)
    xgb_reg_model = xgboost.XGBRegressor().fit(X_train, y_train_log)
    # Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
    xgb_params = [{"learning_rate": [0.001, 0.005, 0.01, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                   "max_depth": [5, 10, 15, 20, 25, None]}]
    gs = GridSearchCV(xgb_reg_model, param_grid=xgb_params, cv=5, n_jobs=-1).fit(X_train, y_train_log)
    line = f"{datetime.now()} - Best parameters for XGBRegressor model: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    xgb_reg_final_model = xgb_reg_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train_log)
    y_pred_log = xgb_reg_final_model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance(xgb_reg_final_model, "XGBRegressor", X_train, save=True)
    line = f"{datetime.now()} - create_xgb_regressor_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return xgb_reg_final_model, score


def create_rf_regressor_model_get_score(X_train, y_train, X_test, y_test):
    # Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse) almayı unutmayınız.
    line = f"{datetime.now()} - create_rf_regressor_model_get_score has started."
    log_lines.append(line)
    print(line)
    y_train_log = np.log(y_train)
    rf_reg_model = RandomForestRegressor().fit(X_train, y_train_log)
    # Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
    rf_reg_params = {"max_features": [1.0, "sqrt", "log2", None], "min_samples_split": [2, 5, 10, 15, 20],
                     "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], "max_depth": [5, 10, 15, 20, 25, None]}
    gs = GridSearchCV(rf_reg_model, param_grid=rf_reg_params, cv=5, n_jobs=-1).fit(X_train, y_train_log)
    line = f"{datetime.now()} - Best parameters for RandomForestRegressor model: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    rf_reg_final_model = rf_reg_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train_log)
    y_pred_log = rf_reg_final_model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance(rf_reg_final_model, "RandomForestRegressor", X_train, save=True)
    line = f"{datetime.now()} - create_rf_regressor_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return rf_reg_final_model, score


def create_linear_reg_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_linear_reg_model_get_score has started."
    log_lines.append(line)
    print(line)
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    line = f"{datetime.now()} - Weights and bias for Linear Regression model: "
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - Weights: {linear_reg_model.coef_}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - Bias: {linear_reg_model.intercept_}"
    log_lines.append(line)
    print(line)
    y_pred = linear_reg_model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance_for_linear_model("LinearRegressor", linear_reg_model.coef_, X_train.columns, save=True)
    line = f"{datetime.now()} - create_linear_reg_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return linear_reg_model, score


def create_lasso_reg_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_lasso_reg_model_get_score has started."
    log_lines.append(line)
    print(line)
    lasso_reg_model = Lasso(max_iter=100000)
    lasso_reg_model_param = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    gs = GridSearchCV(lasso_reg_model, lasso_reg_model_param, cv=5, n_jobs=-1).fit(X_train, y_train)
    line = f"{datetime.now()} - Best parameters for Lasso Linear Regression model: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    lasso_reg_final_model = lasso_reg_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train)
    y_pred = lasso_reg_final_model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance_for_linear_model("LassoLinearRegressor", lasso_reg_final_model.coef_, X_train.columns, save=True)
    line = f"{datetime.now()} - create_lasso_reg_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return lasso_reg_final_model, score


def create_ridge_reg_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_ridge_reg_model_get_score has started."
    log_lines.append(line)
    print(line)
    ridge_reg_model = Ridge()
    ridge_reg_model_param = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    gs = GridSearchCV(ridge_reg_model, ridge_reg_model_param, cv=5, n_jobs=-1).fit(X_train, y_train)
    line = f"{datetime.now()} - Best parameters for Ridge Linear Regression model: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    ridge_reg_final_model = ridge_reg_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train)
    y_pred = ridge_reg_final_model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance_for_linear_model("RidgeLinearRegressor", ridge_reg_final_model.coef_, X_train.columns, save=True)
    line = f"{datetime.now()} - create_ridge_reg_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return ridge_reg_final_model, score


def create_elastic_net_reg_model_get_score(X_train, y_train, X_test, y_test):
    line = f"{datetime.now()} - create_elastic_net_reg_model_get_score has started."
    log_lines.append(line)
    print(line)
    elastic_net_reg_model = ElasticNet()
    elastic_net_reg_model_param = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                                   'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    gs = GridSearchCV(elastic_net_reg_model, elastic_net_reg_model_param, cv=5, n_jobs=-1).fit(X_train, y_train)
    line = f"{datetime.now()} - Best parameters for ElasticNet Linear Regression model: {gs.best_params_}"
    log_lines.append(line)
    print(line)
    elastic_net_reg_final_model = elastic_net_reg_model.set_params(**gs.best_params_, random_state=90).fit(X_train, y_train)
    y_pred = elastic_net_reg_final_model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    plot_importance_for_linear_model("ElasticNetLinearRegressor", elastic_net_reg_final_model.coef_, X_train.columns, save=True)
    line = f"{datetime.now()} - create_elastic_net_reg_model_get_score has finished."
    log_lines.append(line)
    print(line)
    return elastic_net_reg_final_model, score


def plot_importance(model, model_name, features, feature_count_to_plot=10, save=False):
    feature_importances = model.feature_importances_[0:len(features)]
    feature_imp = pd.DataFrame({"Value": feature_importances, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:feature_count_to_plot])

    plt.title("Features")
    plt.tight_layout()
    # plt.show()
    if save:
        plt.savefig(f"{model_name}_importances.png")


def plot_importance_for_linear_model(model_name, model_weights, feature_names, save=True, feature_count_to_plot=20):
    df_importance = pd.DataFrame({"Feature": feature_names, "Weight": model_weights})
    df_importance['Weight_Absolute_Value'] = df_importance['Weight'].abs()
    df_importance = df_importance.sort_values(by='Weight_Absolute_Value', ascending=False)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Weight_Absolute_Value", y="Feature", data=df_importance[0:feature_count_to_plot])

    plt.title("Features")
    plt.tight_layout()
    # plt.show()
    if save:
        plt.savefig(f"{model_name}_importances.png")


def data_prep():
    ############################################################
    # Görev 1: Keşifçi Veri Analizi
    ############################################################

    df_train = pd.read_csv("house-prices-advanced-regression-techniques/train.csv")
    df_train.shape
    df_test = pd.read_csv("house-prices-advanced-regression-techniques/test.csv")
    df_test.shape
    df_test.columns

    # Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
    dfs_to_concat = [df_train, df_test]
    df = pd.concat(dfs_to_concat)

    # Kolon isimleri büyük harf yapılıyor.
    df.columns = [col.upper() for col in df.columns]
    df.reset_index(inplace=True)

    df.head()
    df.shape
    df.info()
    df.describe().T
    df["SALEPRICE"].isnull().sum()

    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
    cat_but_car
    # cat_but_car listesinde NEIGHBORHOOD değişkeni geliyor.
    # Emlak değerlemesinde belki de en önemli kriterlerden birini dikkate almamış olacaktık.
    len(df["NEIGHBORHOOD"].value_counts())  # 25. Dolayısıyla car_th'a 25 değerini vermeliyiz.
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, car_th=25)

    # Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
    # Tip hatası olan bir değişken gözlemlemedim.

    # Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
    num_summary(df)
    cat_summary(df)

    # Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
    target_summary_with_cat(df)
    target_summary_with_num(df)

    # Adım 6: Aykırı gözlem var mı inceleyiniz.
    check_outlier(df)

    # Adım 7: Eksik gözlem var mı inceleyiniz.
    df.isnull().sum()
    df[num_cols].isnull().sum()
    df[cat_cols].isnull().sum()

    ############################################################
    # Görev 2: Feature Engineering
    ############################################################
    # Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
    check_outlier(df)  # Aykırı değer var.
    df = replace_with_thresholds(df)
    check_outlier(df)  # Aykırı değer kalmadı.

    df = report_and_fill_empty_inputs(df)

    # Adım 2: Rare Encoder uygulayınız.
    rare_analyzer(df)
    df = rare_encoder(df, 0.003)
    df.head()

    num_summary(df)

    # Adım 3: Yeni değişkenler oluşturunuz.
    df = generate_new_features(df)

    # Adım 4: Encoding işlemlerini gerçekleştiriniz.
    df = one_hot_encoder(df)
    return df


def get_tree_model_score_dict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)

    ############################################################
    # Görev 3: Model Kurma
    ############################################################
    # LightGBM Regressor ile
    light_gbm_final_model, lgbm_score = create_lgbm_regressor_model_get_score(X_train, y_train, X_test, y_test)

    # Linear Regressor ile
    grad_boost_final_reg_model, grad_boost_reg_score = create_gradient_boost_regression_model_get_score(X_train, y_train, X_test, y_test)

    # XGBoost Regressor ile
    xgboost_reg_final_model, xgboost_score = create_xgb_regressor_model_get_score(X_train, y_train, X_test, y_test)

    # Random Forest Regressor ile
    rf_reg_final_model, rf_reg_score = create_rf_regressor_model_get_score(X_train, y_train, X_test, y_test)

    line = f"{datetime.now()} - Tree Models and Their Scores"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(light_gbm_final_model).__name__}: {lgbm_score}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(grad_boost_final_reg_model).__name__}: {grad_boost_reg_score}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(xgboost_reg_final_model).__name__}: {xgboost_score}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(rf_reg_final_model).__name__}: {rf_reg_score}"
    log_lines.append(line)
    print(line)

    model_score_dict = {light_gbm_final_model: lgbm_score, grad_boost_final_reg_model: grad_boost_reg_score,
                        xgboost_reg_final_model: xgboost_score, rf_reg_final_model: rf_reg_score}
    return model_score_dict


def get_linear_model_score_dict(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=90)

    ############################################################
    # Görev 3: Model Kurma
    ############################################################
    # Linear Regressor ile
    linear_reg_model, linear_reg_score = create_linear_reg_model_get_score(X_train, y_train, X_test, y_test)

    # Ridge Linear Regressor ile
    ridge_final_reg_model, ridge_reg_score = create_ridge_reg_model_get_score(X_train, y_train, X_test, y_test)

    # Lasso Linear Regressor ile
    lasso_reg_final_model, lasso_reg_score = create_lasso_reg_model_get_score(X_train, y_train, X_test, y_test)

    # ElasticNet Linear Regressor ile
    elastic_net_reg_final_model, elastic_net_reg_score = create_elastic_net_reg_model_get_score(X_train, y_train, X_test, y_test)

    line = f"{datetime.now()} - Linear Models and Their Scores"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(linear_reg_model).__name__}: {linear_reg_score}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(ridge_final_reg_model).__name__}: {ridge_reg_score}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(lasso_reg_final_model).__name__}: {lasso_reg_score}"
    log_lines.append(line)
    print(line)
    line = f"{datetime.now()} - {type(elastic_net_reg_final_model).__name__}: {elastic_net_reg_score}"
    log_lines.append(line)
    print(line)

    model_score_dict = {linear_reg_model: linear_reg_score, ridge_final_reg_model: ridge_reg_score,
                        lasso_reg_final_model: lasso_reg_score, elastic_net_reg_final_model: elastic_net_reg_score}
    return model_score_dict


def house_price_predict(log_lines):
    df = data_prep()
    line = f"{datetime.now()} - house_price_predict has started."
    log_lines.append(line)
    print(line)

    # Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
    df_test = df[df["SALEPRICE"].isnull()]
    df_train = df[df["SALEPRICE"].notnull()]

    # Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.
    target = "SALEPRICE"
    y = df_train[target]
    X = df_train.drop([target, "ID"], axis=1)
    tree_model_score_dict = get_tree_model_score_dict(X, y)
    # tree_model_with_lowest_rmse_score = sort_models_by_rmse_get_first(tree_model_score_dict)

    linear_model_score_dict = get_linear_model_score_dict(X, y)
    # linear_model_with_lowest_rmse_score = sort_models_by_rmse_get_first(linear_model_score_dict)

    model_score_dict = tree_model_score_dict | linear_model_score_dict
    model_with_lowest_rmse_score = sort_models_by_rmse_get_first(model_score_dict)

    # Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturup sonucunuzu yükleyiniz.
    df_result = predict_with_model_get_result(df_test, model_with_lowest_rmse_score)
    df_result.to_csv("submission.csv", sep=",", index=False, columns=df_result.columns)
    line = f"{datetime.now()} - house_price_predict has finished."
    log_lines.append(line)
    print(line)


def predict_with_model_get_result(dataframe, model, target="SALEPRICE"):
    X = dataframe.drop([target, "ID"], axis=1)
    dataframe[target] = model.predict(X)
    tree_models = ["LGBMRegressor", "GradientBoostingRegressor", "XGBRegressor", "RandomForestRegressor"]
    if type(model).__name__ in tree_models:
        dataframe[target] = np.exp(dataframe[target])
    dataframe["ID"] = dataframe["ID"].astype(int)
    df_result = dataframe[["ID", "SALEPRICE"]]
    df_result.columns = ["Id", "SalePrice"]
    df_result["Id"] = df_result["Id"].astype(int)
    return df_result


def sort_models_by_rmse_get_first(model_score_dict):
    line = f"{datetime.now()} - Scores are being ordered. Final model will be decided according to the scores. " \
           "The minimum score is the best score, because scores are calculated via RMSE method."
    log_lines.append(line)
    print(line)

    keys = list(model_score_dict.keys())
    values = list(model_score_dict.values())
    sorted_value_index = np.argsort(values)
    sorted_model_score_dict = {keys[i]: values[i] for i in sorted_value_index}
    model_with_lowest_rmse_score = next(iter(sorted_model_score_dict))
    return model_with_lowest_rmse_score


def write_logs_to_file(log_lines):
    file_name = f"logs_{datetime.now().year}{datetime.now().month}{datetime.now().day}_{datetime.now().hour}{datetime.now().minute}.txt"
    with open(file_name, 'w') as f:
        for line in log_lines:
            f.write(f"{line}\n")


if __name__ == "__main__":
    # using now() to get current time
    log_lines = []
    dt_start = datetime.now()
    line = f"{dt_start} - House price prediction process is started. Linear and tree models will be examined."
    print(line)
    log_lines.append(line)
    house_price_predict(log_lines)
    dt_now = datetime.now()
    elapsed_time = dt_now - dt_start
    line = f"{datetime.now()} - Elapsed time: {str(elapsed_time)}"
    print(line)
    log_lines.append(line)
    line = f"{dt_now} - House price prediction process is ended."
    print(line)
    write_logs_to_file(log_lines)
