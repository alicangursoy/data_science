############################################################
# KNN (K-nearest neighbors)
############################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option("display.max_columns", None)


################################################
# 1. Exploratory Data Analysis
################################################
df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


################################################
# 2. Data Preprocessing & Feature Engineering
################################################
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_scaled = StandardScaler().fit_transform(X)  # KNN yönteminde bağımlı değişkenleri standart hale getirmemiz gerekiyor. Böylece daha iyi sonuç elde ediyoruz.

X = pd.DataFrame(X_scaled, columns=X.columns)
X.head()


################################################
# 3. Modeling
################################################
knn_model = KNeighborsClassifier().fit(X, y)
random_user = X.sample(1, random_state=45)
knn_model.predict(random_user)


################################################
# 4. Model Evaluation
################################################
# Confussion Matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))  # accuracy = 0.83, precision = 0.79, recall = 0.70, f1-score = 0.74

# AUC
roc_auc_score(y, y_prob)  # 0.9017686567164179

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()  # 0.733112638994992
cv_results["test_f1"].mean()  # 0.5905780011534191
cv_results["test_roc_auc"].mean()  # 0.7805279524807827


# Cross_validate ile modeli valide ettiğimizde, daha önce bulduğumuz accuracy, f1 score ve roc auc score'ların çok düştüğünü görüyoruz.
# Bu skorları artırabilmek için ne yapabiliriz?
# 1. Veri boyutu artırılabilir.
# 2. Veri ön işleme ile aykırı değerler vs.ler önlenebilir.
# 3. Özellik mühendisliği ile yeni değişkenler türetilip onlar ile model oluşturulabilir.
# 4. İlgili algoritma için optimizasyonlar yapılabilir.


knn_model.get_params()
# 'n_neighbors': 5 => Bu parametre, gözlem biriminin kaç komşuluğuyla bu modelin kurulduğunu belirtir. Bu parametreyi değiştirerek modelin başarısını artırabiliriz.

# parametre vs. hyperparametre
# Parametre, verilerin ağırlıklarının parametresi. Bunu model kendi öğreniyor.
# Hyperparametre, modele dışardan verilebilen parametrelerdir. KNN için, n_neighbors değeri



############################################
# 5. Hyperparameter Optimization
############################################
knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs=-1, verbose=1).fit(X, y)
# n_jobs=-1 iken işlemcileri tam performans ile kullan demek oluyor.
# verbose=1 iken işlemlerin sonucunda rapor beklediğimizi belirtmiş oluyoruz.

knn_gs_best.best_params_  # {'n_neighbors': 17}


############################################
# 6. Final Model
############################################
# En iyi hyperparametre değerini yukarıda tespit ettik ('n_neighbors': 17). Buna göre modelimizi tekrar kuruyoruz.

# knn_gs_best.best_params_ dictionary'sini tek tek yazmak yerine başına 2 yıldız koyarak kendinin otomatik olarak almasını sağlıyoruz.
knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()  # n_neighbors = 5 iken 0.733112638994992 idi, n_neighbors = 17 iken 0.7669892199303965
cv_results["test_f1"].mean()  # n_neighbors = 5 iken 0.5905780011534191 idi, n_neighbors = 17 iken 0.6170909049720137
cv_results["test_roc_auc"].mean()  # n_neighbors = 5 iken 0.7805279524807827 idi, n_neighbors = 17 iken 0.8127938504542278

random_user = X.sample(1)
knn_final.predict(random_user)  # 0 geldi.

random_user = X.sample(1)
knn_final.predict(random_user)  # 1 geldi.