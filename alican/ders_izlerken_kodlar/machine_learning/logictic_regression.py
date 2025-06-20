######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

import numpy as np
import pandas as pd
import matplotlib
import plotly.io

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.93):
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


pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)


################################################
# Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape


#########################################################
# Target'ın Analizi (Bağımlı Değişkenin Analizi)
#########################################################

df["Outcome"].value_counts()
sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)


############################################################
# Feature'ların Analizi (Bağımsız Değişkenlerin Analizi)
############################################################

df.describe().T  # describe sadece sayısal değişkenleri getirir ve onların durumunu özetler.

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

df["Glucose"].hist(bins=20)
plt.xlabel("Glucose")
plt.show()


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)  # Peşpeşe göstereceğimiz grafiklerin birbirini ezmemesi için block=True yolluyoruz.


for col in df.columns:
    plot_numerical_col(df, col)


# Hedef değişkeni yukarıdaki listeden çıkarmak istiyoruz.
cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)



######################################################
# Target vs Features
######################################################

df.groupby("Outcome").agg({"Pregnancies": "mean"})


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(df, "Outcome", col)



#################################################################
# Data Preprocessing (Veri Ön İşleme)
#################################################################
df.shape
df.head()

df.isnull().sum()

df.describe().T

for col in cols:
    hasOutliers = check_outlier(df, col)
    print(col, hasOutliers)
    if hasOutliers:
        replace_with_thresholds(df, col)


# Verileri standart hale getiriyoruz ki modeller veriyi işlerken bir değişkeninn diğerinden üstün olduğunu düşünmesin.
# Bunun için RobustScaler kullanıyoruz, bunun nedeni RobustScaler'ın aykırı değerlere dayanıklı olması.
# Başka scaler'lar da kullanabilirdik, MinMaxSclaer vb. Burda tamamen tercihe ve veri setinin niteliğine kalıyor.
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


########################################################
# Model & Prediction
########################################################
# Logictic Regression kullanıyoruz.

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)
log_model.intercept_  # b
log_model.coef_  # w1, w2, w3, w4, ...

# model denklemi = b + (w1 * x1) + (w2 * x2) + (w3 * x3) + (w4 * x4) + ...

y_pred = log_model.predict(X)  # Bağımsız değişkenleri al ve tahmin et.
y_pred[0:10]  # İlk 10 tanesine örnek olarak bakalım.
y[0:10]


########################################################
# Model Evaluation (Model Başarı Değerlendirme)
########################################################

def plot_confussion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()


plot_confussion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Precission: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# ROC AUC score: 0.8394626865671642


########################################################
# Model Validation: Holdout (Model Doğrulama)
########################################################
# Yukarıda, elimizdeki veri setinin tümüyle model oluşturduk ve modeli yine aynı veri setiyle test ettik. Bu modelin doğrulanmaya ihtiyacı var.
# Bunun için, elimizdeki veri setini train ve test olacak şekilde ayırmalıyız. random_state=17, videodaki test ve train veri setlerinin birebir elimizde olması
# ve skorların uyuşması için gereklidir. Yoksa rastgele değerlerle denenebilir.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.77
# Precission: 0.79
# Recall: 0.53
# F1-score: 0.63

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title("ROC Curve")
plt.plot([0, 1], [0, 1], "r--")
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
# ROC AUC score: 0.8752034725990233



########################################################
# Model Validation: 10-Fold Cross Validation
########################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])  # cv=5 : Kaç katlı cross validation yapacağımız
cv_results = cross_validate(log_model, X, y, cv=10, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])  # cv=10 : Kaç katlı cross validation yapacağımız

cv_results["test_accuracy"].mean()  # Ortalama test accuracy'si
cv_results["test_precision"].mean()  # Ortalama test precision'ı
cv_results["test_recall"].mean()  # Ortalama test recall'u
cv_results["test_f1"].mean()  # Ortalama test f1 score'u
cv_results["test_roc_auc"].mean()  # Ortalama test ROC AUC score'u


# Dengeli bir veri setinde (0 = 0.45- 1 = 0.55 gibi) sadece accuracy iş görebilir ama dengeli değilse precission, recall, f1 score'a da bakmak gerekiyor.


#######################################################
# Prediction for A New Observation
#######################################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)  # array([1]) => Bu kişi diabettir şeklinde tahmin etti modelimiz.