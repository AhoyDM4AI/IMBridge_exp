import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn

from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType
from hummingbird.ml import convert, load
# hospital

#表路径
path1 = "./LengthOfStay.csv"
#读取csv表
data = pd.read_csv(path1)
print(data)
#替换缺失值
data.dropna(inplace=True)      #删除NaN

#获取类型
numerical = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse',
	'respiration', 'secondarydiagnosisnonicd9']
categorical = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence',
	'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 'malnutrition', 'hemo']
input_columns = numerical + categorical

X = data.loc[:, input_columns]
#获取分类label
y = np.array(data.loc[:, 'lengthofstay'])
# ONNX pipeline
type_map = {
    "int64": Int64TensorType([None, 1]),
    "float32": FloatTensorType([None, 1]),
    "float64": FloatTensorType([None, 1]),
    "object": StringTensorType([None, 1])
}
init_types = [(elem, type_map[X[elem].dtype.name]) for elem in input_columns]

numerical_preprocessor = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
    ],
    verbose=True
)
normal_preprocessor = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown='ignore')),
    ],
    verbose=True
)
preprocessor = ColumnTransformer(
    [
        ("numerical", numerical_preprocessor, numerical),
        ("categorical", normal_preprocessor, categorical),
    ],
    verbose=True
)

# 获取训练数据
# X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

# 训练过程
preprocessor.fit_transform(X)                                           #预处理器
X_train_pre = preprocessor.transform(X_train)                           #处理测试集
X_test_pre = preprocessor.transform(X_test)

mlp = MLPClassifier()                                                   #多重感知机模型
mlp.fit(X_train_pre, y_train)                                           #训练

y_prob = mlp.predict_proba(X_test_pre)[:, 1]                            #预测结果为1的概率
y_pred = mlp.predict(X_test_pre)                                        #预测结果
fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob, pos_label=1)    #真阳率、伪阳率、阈值
auc = metrics.auc(fpr, tpr)                                             #AUC
score = metrics.accuracy_score(y_test, y_pred)                          #模型准确率
print('MLP模型准确率:{}\nAUC得分:{}'.format(score, auc))

# 转换模型
model = Pipeline(steps=[
    ('precprocessor', preprocessor),
    ('classifier', mlp)
])
model_onnx = convert_sklearn(model, initial_types = init_types)

# 保存模型
model_path = './hospital_mlp_pipeline.onnx'
with open(model_path, 'wb') as f:
    f.write(model_onnx.SerializeToString())

