########################################################
# Sales Prediction with Linear Regression
########################################################
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.2f" % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

########################################################
# Simple Linear Regression with OLS Using Scikit-Learn
########################################################

df = pd.read_csv("datasets/advertising.csv")
df.head()
df.shape

X = df[["TV"]]
y = df[["sales"]]


#############################################
# Model
#############################################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*TV

# sabit (b - bias(intercept))
b = reg_model.intercept_[0]  # array döndüğü için 0.elemanı alıyoruz.

# TV'nin katsayısı (w - weight(coefficient))
w = reg_model.coef_[0][0]


#############################################
# Tahmin
#############################################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
y_hat = b + w * 150

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
y_hat = b + w * 500

df.describe().T
# TV'nin max değeri 296.40 olmasına rağmen
# # "500 için satış kaç olurdu?" sorumuzun cevabını LinearRegression sayesinde tahminleyebiliyoruz.


################################################
# Modelin Görselleştirilmesi
################################################
g = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s": 9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


################################################
# Tahmin Başarısı
################################################
# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 10.51
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
reg_model.score(X, y)
# 0.61


########################################
# Multiple Linear Regression
########################################

df = pd.read_csv("datasets/advertising.csv")

X = df.drop("sales", axis=1)
y = df[["sales"]]


########################################
# Model
########################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape
y_train.shape
X_test.shape
y_test.shape

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# sabit (b - bias(intercept))
reg_model.intercept_[0]

# bağımlı değişkenlerin katsayıları (TV, radio, newspaper) (w - weight(coefficients))
reg_model.coef_[0]


############################
# Tahmin
############################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# [0.0468431 , 0.17854434, 0.00258619]

# Sales = 2.90 + TV * 0.04 + radio * 0.17 + newspaper * 0.002
y_hat = 2.90 + (30 * 0.0468431) + (10 * 0.17854434) + (40 * 0.00258619)

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)



################################################
# Tahmin Başarısını Değerlendirme
################################################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.736902590147092

# Train RKARE
reg_model.score(X_train, y_train)
# 0.8959372632325174


# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.4113417558581565

# Test RKARE
reg_model.score(X_test, y_test)
# 0.8927605914615387


# 10 katlı CV (Cross Validation) RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))
# 1.6913531708051799


# 5 katlı CV (Cross Validation) RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# 1.7175247278732086



###################################################
# Simple Linear Regression with Gradient Descent from Scratch
###################################################

# Cost function (MSE)
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0  # sum of squared error
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m  # mean squared error
    return mse


# update weights: Bu fonksiyon sadece 1 iterasyon içeriyor, birkaç kez çağrılmalı.
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]

    new_b = b - (learning_rate * (1 / m) * b_deriv_sum)
    new_w = w - (learning_rate * (1 / m) * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                    cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter:{:d}   b:{:.2f}     w:{:.4f}    mse:{:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/advertising.csv")

X = df["radio"]
y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

cost_history, b, w = train(y, initial_b, initial_w, X, learning_rate, num_iters)