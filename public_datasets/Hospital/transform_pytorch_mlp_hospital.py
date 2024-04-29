import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from hummingbird.ml import convert, load
from torch import save, load

# Hummingbird Sklearn -> PyTorch 模型

# hospital

#表路径
path1 = "./LengthOfStay.csv"
#读取csv表
data = pd.read_csv(path1)
print(data)

#获取类型
numerical_names = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse',
	'respiration', 'secondarydiagnosisnonicd9']
categorical_names = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence',
	'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo']

#替换缺失值
data.dropna(inplace=True)      #删除NaN

#获取分类label
y = np.array(data.loc[:, 'lengthofstay'])
#10 numerical, 12 categorical
numerical = np.array(data.loc[:, numerical_names])
categorical = np.array(data.loc[:, categorical_names])

#standard scaling & one-hot encoding
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder(handle_unknown='ignore')
one_hot_model = enc.fit(categorical)

#获取训练数据
X = np.hstack((scaler.transform(numerical) , enc.transform(categorical).toarray()))
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

# 训练过程
mlp = MLPClassifier()                                                   #多重感知机模型
mlp.fit(X_train, y_train)                                               #训练

y_prob = mlp.predict_proba(X_test)[:, 1]                                #预测结果为1的概率
y_pred = mlp.predict(X_test)                                            #预测结果
fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob, pos_label=1)    #真阳率、伪阳率、阈值
auc = metrics.auc(fpr, tpr)                                             #AUC
score = metrics.accuracy_score(y_test, y_pred)                          #模型准确率
print('MLP模型准确率:{}\nAUC得分:{}'.format(score, auc))

#保存模型
scaler_path = './hospital_standard_scale_model.pkl'
enc_path = './hospital_one_hot_encoder.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)

test_input = X_test[0:1]
model = convert(mlp, 'pytorch', test_input=test_input)

# save model
# model.save('./hospital_mlp_pytorch_model')
save(model, './hospital_mlp_pytorch.pth')