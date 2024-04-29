import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn

from onnxconverter_common import FloatTensorType, Int64TensorType, StringTensorType

# 抽取字符串并转换为FLOAT32，选用
def cols_str2float(input_columns, X):
    # 输入为 列名 + pandas dataframe
    fc = []
    for col in input_columns:
        xt = X[col].dtype.name
        if xt == 'object':
            fc.append(col)
    X[fc] = X[fc].apply(lambda x:
        x.str.replace("\'", "").astype(np.float32)
    )
    return X

# Credit Card

#表路径
path1 = "./creditcard.csv"

#读取csv表
data = pd.read_csv(path1)

data.dropna(inplace=True)      #删除NaN

#获取分类label
y = np.array(data.loc[:, 'Class'])
#28 numerical
features = []
for i in range(1, 29):
	features.append('V{}'.format(i))
features.append('Amount')
numerical = features
input_columns = features
#Time 计算相隔时间
#Amount 权重系数

X = data.loc[:, input_columns]
# X = cols_str2float(input_columns, X)
# print(X.loc[:10, categorical])

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
preprocessor = ColumnTransformer(
    [
        ("numerical", numerical_preprocessor, numerical)
    ],
    verbose=True
)

# 获取训练数据
# X = X.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

# 训练过程
preprocessor.fit_transform(X)                                                       #预处理器
X_train_pre = preprocessor.transform(X_train)                                         #处理测试集
X_test_pre = preprocessor.transform(X_test)

lr = LogisticRegression(solver='liblinear')                                         #逻辑回归模型
lr.fit(X_train_pre, y_train)                                                        #训练

y_prob = lr.predict_proba(X_test_pre)[:, 1]                                         #预测结果为1的概率
y_pred = lr.predict(X_test_pre)                                                     #预测结果
fpr_lr, tpr_lr, threshold_lr = metrics.roc_curve(y_test, y_prob)                    #真阳率、伪阳率、阈值
auc_lr = metrics.auc(fpr_lr,tpr_lr)                                                 #AUC
score_lr = metrics.accuracy_score(y_test, y_pred)                                   #模型准确率
print('LR模型准确率:{}\nAUC得分:{}'.format(score_lr, auc_lr))

# 转换模型
model = Pipeline(steps=[
    ('precprocessor', preprocessor),
    ('classifier', lr)
])
model_onnx = convert_sklearn(model, initial_types = init_types)

# 保存模型
# model_path = './creditcard_nostr.onnx'
model_path = './creditcard_pipeline.onnx'
with open(model_path, 'wb') as f:
    f.write(model_onnx.SerializeToString())
