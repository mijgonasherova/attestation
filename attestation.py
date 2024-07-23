import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI': lst})
data.head()
# print(data)

# 1 вариант:
onehotencoder = OneHotEncoder()
encoder_data = onehotencoder.fit_transform(data[['whoAmI']]).toarray()
encoder_data = pd.DataFrame(encoder_data, columns=onehotencoder.categories_[0])
res_one_hot = pd.concat([data, encoder_data], axis=1)
print(res_one_hot)

# 2 вариант(без наименований категорий):
# onehotencoder = OneHotEncoder()
# encoder_data = pd.DataFrame(onehotencoder.fit_transform(data[['whoAmI']]).toarray())
# res_one_hot = pd.concat([data, encoder_data], axis=1)
# print(res_one_hot)
