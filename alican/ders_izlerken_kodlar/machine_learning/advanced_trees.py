#######################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
#######################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install xgboost
# !pip install lightgbm
# !pip install catboost

pd.set_option("display.max_columns", None)
warnings.simplefilter(action="ignore", category=Warning)

df = pd.read_csv("datasets/diabetes.csv")
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)



####################################################
# Random Forests
####################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
# Bizim için önemli olan hyperparametreler:
# max_features
# max_depth
# min_samples_split
# n_estimators: Fit edilecek random ağaç sayısı

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7525803144224197
cv_results["test_f1"].mean()  # 0.6165191330554752
cv_results["test_roc_auc"].mean()  # 0.8238418803418803


# Optimize edilmesi gereken hyperparametreler
rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

# max_features değeri, veri setindeki bağımsız değişken sayısından fazla olmamalı. Yoksa hata alınıyor.

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_
'''
{'max_depth': None,
 'max_features': 5,
 'min_samples_split': 8,
 'n_estimators': 500}
'''

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.766848940533151
cv_results["test_f1"].mean()  # 0.6447777811143756
cv_results["test_roc_auc"].mean()  # 0.8274814814814816


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model, X=X, y=y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=cv,
                                               scoring=scoring)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)


plot_importance(rf_final, X)
val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="f1")



#########################################################
# GBM
#########################################################

gbm_model = GradientBoostingClassifier(random_state=17)

cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7591715474068416
cv_results["test_f1"].mean()  # 0.634235802826363
cv_results["test_roc_auc"].mean()  # 0.8260164220824597

gbm_model.get_params()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_
'''
{'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1000, 'subsample': 0.7}
'''

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7734912146676853
cv_results["test_f1"].mean()  # 0.6608461098755429
cv_results["test_roc_auc"].mean()  # 0.8351411600279526



########################################################
# XGBoost
########################################################
xgboost_model = XGBClassifier(random_state=17)

cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7409557762498938
cv_results["test_f1"].mean()  # 0.6231739342622644
cv_results["test_roc_auc"].mean()  # 0.7991180992313069

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, None],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [None, 0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

xgboost_best_grid.best_params_
'''
{'colsample_bytree': 0.7,
 'learning_rate': 0.1,
 'max_depth': 5,
 'n_estimators': 100}
'''

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7617689500042442
cv_results["test_f1"].mean()  # 0.6363347763347763
cv_results["test_roc_auc"].mean()  # 0.8183675751222921


########################################################
# LightGBM
########################################################
lgbm_model = LGBMClassifier(random_state=17)

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7474492827434004
cv_results["test_f1"].mean()  # 0.624110522144179
cv_results["test_roc_auc"].mean()  # 0.7990293501048218

lgbm_params = {"learning_rate": [0.1, 0.01],
              "n_estimators": [100, 300, 500, 1000],
              "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_
'''
{'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 300}
'''

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7617689500042442
cv_results["test_f1"].mean()  # 0.6363347763347763

cv_results["test_roc_auc"].mean()  # 0.8183675751222921

# Light GBM'de en önemli parametre n_estimators'dür.
# Light GBM ile çalışırken, bu değişkenin değerini artırıp çıkararak sıklıkla denenmelidi.


########################################################
# CatBoost
########################################################
cat_boost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(cat_boost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7735251676428148
cv_results["test_f1"].mean()  # 0.6502723851348231
cv_results["test_roc_auc"].mean()  # 0.8378923829489867


cat_boost_params = {"learning_rate": [0.1, 0.01],
              "iterations": [200, 500],
              "depth": [3, 6]}

cat_boost_best_grid = GridSearchCV(cat_boost_model, cat_boost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

cat_boost_best_grid.best_params_
'''
{'depth': 3, 'iterations': 500, 'learning_rate': 0.01}
'''

cat_boost_final = cat_boost_model.set_params(**cat_boost_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(cat_boost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7708768355827178
cv_results["test_f1"].mean()  # 0.6342431003125872
cv_results["test_roc_auc"].mean()  # 0.8421956673654787




############################################################
# Feature Importance
############################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])

    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(rf_final, X)  # Glucose, BMI, Age
plot_importance(gbm_final, X)  # Glucose, BMI, Age
plot_importance(xgboost_final, X)  # Glucose, Age, BMI
plot_importance(lgbm_final, X)  # BMI, DiabetesPedigreeFunction, Glucose
plot_importance(cat_boost_final, X)  # Glucose, BMI, Age



########################################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
########################################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # rf_random_params'ın kombinasyonlarından denenecek kombinasyon sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)

rf_random.best_params_
'''
{'n_estimators': 200,
 'min_samples_split': 13,
 'max_features': 5,
 'max_depth': 32}
'''

rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.7722010016127664
cv_results["test_f1"].mean()  # 0.6524947274947275
cv_results["test_roc_auc"].mean()  # 0.8283864430468204



########################################################
# Analyzing Model Complexity with Learning Curves (BONUS)
########################################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model, X=X, y=y,
                                               param_name=param_name,
                                               param_range=param_range,
                                               cv=cv,
                                               scoring=scoring)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score, label="Training Score", color="b")
    plt.plot(param_range, mean_test_score, label="Validation Score", color="g")
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]

rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])


