# RALP
Reconnaissance automatique du langage de programmation

| Langage       |
| :------------ |
| css       |
| go     |
| html        |
| java        |
| js        |
| python        |
| ruby        |



## Comment l'utiliser
```python
import os

from sklearn.externals import joblib

clf = joblib.load('modelTrain.pkl')

# Defined your test directory
DIR_TEST_PATH = "test/"

data = []
for file in os.listdir(DIR_TEST_PATH):
    with open(DIR_TEST_PATH + file, 'r') as f:
        data.append(f.read())
        f.close()

predict = clf.predict(data)

print(predict)
```
