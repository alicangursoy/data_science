import math
import numpy as np
import pandas as pd

# Görev 1:

#####################################################################################################
# | Gerçek Değer | Model Olasılık Tahmini(1 sınıfına ait olma olasılığı) | Eşik Değeri 0.5 iken Değer
#####################################################################################################
# | 1 | 0.7  | 1 |
# | 1 | 0.8  | 1 |
# | 1 | 0.65 | 1 |
# | 1 | 0.9  | 1 |
# | 1 | 0.45 | 0 |
# | 1 | 0.5  | 1 |
# | 0 | 0.55 | 1 |
# | 0 | 0.35 | 0 |
# | 0 | 0.4  | 0 |
# | 0 | 0.25 | 0 |


#                     | Model Churn(1) | Model Non-Churn(0) |
# ----------------------------------------------------------|
# Gerçek Churn(1)     |       5        |          1         | 6
# ----------------------------------------------------------|
# Gerçek Non-Churn(0) |       1        |          3         | 4
# ----------------------------------------------------------|
#                     |       6        |          4         |
# ----------------------------------------------------------|


# Accuracy = (TP + TN) / (TP + FN + TN + FP)
# Accuracy = (5 + 3) / 10 = 8 / 10 = 0.80

# Recall = TP / (TP + FN)
# Recall = 5 / 6 = 0.83

# Precision = TP / (TP + FP)
# Precision = 5 / 6 = 0.83

# F1 score = 2 * Precision * Recall / (Precision + Recall)
# F1 score = 2 * 0.83 * 0.83 / (0.83 + 0.83) = 0.83


# Görev 2
#                     | Model Fraud(1) | Model Non-Fraud(0) |
# ----------------------------------------------------------------|
# Gerçek Fraud(1)     |       5        |          5         | 10  |
# ----------------------------------------------------------------|
# Gerçek Non-Fraud(0) |      90        |        900         | 990 |
# --------------------------------------------------------------- |
#                     |      95        |        905         |
# ----------------------------------------------------------|

# Accuracy = (TP + TN) / (TP + FN + TN + FP)
# Accuracy = (5 + 900) / 1000 = 905 / 1000 = 0.905

# Recall = TP / (TP + FN)
# Recall = 5 / 10 = 0.50

# Precision = TP / (TP + FP)
# Precision = 5 / 95 = 0.053

# F1 score = 2 * Precision * Recall / (Precision + Recall)
# F1 score = 2 * 0.053 * 0.50 / (0.053 + 0.50) = 0.096

# Doğruluk (accuracy) değeri %90.5 gibi çok yüksek bir değer olsa da, doğruluk değerinde yalnızca doğru tahminler
# göz önünde bulundurulur. Halbuki, yanlış tahminler de doğru tahminler kadar etkilidir. Özellikle, bir bankada,
# fraud olmayan işlemler fraud'muş gibi işaretlenir ve müşterilere bu şekilde dönüşler yapılırsa, müşteriler
# duygusal varlıklar olduğu için etkilenip bankayla çalışmayı bırakacaktır.
# Ayrıca, bu örnek modelde fraud işlemlerin sadece %50'si yakalanabilmiş ve bu da büyük para kayıplarına neden olabilir.
# Dolayısıyla bir model değerlendirilirken sadece Accuracy değeri değil F1 score'un da göz önünde dolayısıyla Recall ve Precision
# değerlerinin de göz önünde bulundurulmalıdır.

# Veri bilimi ekibi modeli gözden geçirerek;
# recall, precision ve F1 score'larını yukarılara çekecek şekilde modeli optimize etmelidir.
