import math
import numpy as np
import pandas as pd

columns = ["x", "y"] # x: Deneyim Yılı, y: Maaş
data = [[5, 600], [7, 900], [3, 550], [3, 500], [2, 400], [7, 950], [3, 540], [10, 1200],
        [6, 900], [4, 550], [8, 1100], [1, 460], [1, 400], [9, 1000], [1, 380]]
df = pd.DataFrame(data=data, columns=columns)

# 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz.Bias=275,Weight=90(y’=b+wx)
# y' = 275 + (90 * x)

# 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
b = 275
w = 90
df["y_hat"] = b + w * df["x"]

# 3-Modelin başarısını ölçmek için MSE,RMSE,MAE skorlarını hesaplayınız.
# MSE, RMSE, MAE
MSE = 0
RMSE = 0
MAE = 0
for index, row in df.iterrows():
    MSE += pow((row["y_hat"] - row["y"]), 2)
    MAE += abs(row["y_hat"] - row["y"])

MAE = MAE / df.shape[0]
MSE = MSE / df.shape[0]
RMSE = math.sqrt(MSE)



