###################################################
# End-to-End Diabetes Machine Learning Pipeline III
###################################################

import joblib
import pandas as pd

df = pd.read_csv("datasets/diabetes.csv")
random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")
new_model.predict(random_user)
# Hata aldı. Nedeni, 8.satırda okuttuğumuz veri setinin, modelin işleyebileceği tarzda olmaması.
# Bunu çözmek için, 8.satırda okunan veri setini, modelin işleyebileceği tarza çevirmeliyiz.

from diabetes_pipeline import diabetes_data_prep

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=45)
new_model = joblib.load("voting_clf.pkl")
new_model.predict(random_user)  # Sonuç: 1 => diatbet hastası