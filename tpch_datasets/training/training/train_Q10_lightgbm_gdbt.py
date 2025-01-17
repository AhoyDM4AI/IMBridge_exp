import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

# 定义全局变量
n_epochs = 10
batch_size = 200

# 读取数据
path1 = "./lineitem.tbl"
path2 = "./customer.tbl"
path3 = "./orders.tbl"
path4 = "./nation.tbl"
lineitem = pd.read_table(path1, names=['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 
                                       'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 
                                       'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], sep='|', index_col=False, nrows=1000000) # nrows=1000000
#print(lineitem.info())
customer = pd.read_table(path2, names=['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 
                                       'c_comment'], sep='|', index_col=False, nrows=500000) # nrows=500000
#print(customer.info())
orders = pd.read_table(path3, names=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 
                                     'o_clerk', 'o_shippriority', 'o_comment'], sep='|', index_col=False)
#print(orders.info())
nation = pd.read_table(path4, names=['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], sep='|', index_col=False)
#print(nation.info())

# join
#data = pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey')
data = pd.merge(pd.merge(pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey'), 
                         lineitem, how = 'inner', left_on = 'o_orderkey', right_on = 'l_orderkey'), 
                nation, how = 'inner', left_on = 'c_nationkey', right_on = 'n_nationkey')
#print(data.info())
select_cols = ['c_custkey', 'c_acctbal', 'o_orderkey', 'o_orderstatus', 'o_totalprice', 'o_orderpriority', 'o_clerk', 
               'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipinstruct', 'l_shipmode',
               'n_nationkey', 'n_regionkey']
data = data[select_cols]
print(data.info())
# 获取特征和数据预处理

numerical_names = ['c_acctbal', 'o_totalprice', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax'] # 
categorical_names = ['o_orderstatus', 'o_orderpriority', 'l_linestatus', 'l_shipinstruct', 'l_shipmode', 'n_nationkey', 'n_regionkey'] # 
features = numerical_names + categorical_names
numerical = np.array(data.loc[:, numerical_names])
categorical = np.array(data.loc[:, categorical_names])
label = np.array(data.loc[:, 'l_returnflag'])


# 标准化和独热编码
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder()
one_hot_model = enc.fit(categorical)

X_numerical = scaler.transform(numerical)
X_categorical = enc.transform(categorical).toarray()
X = np.hstack((X_numerical, X_categorical))

# 'A' - > 0, 'N' -> 1, 'R' -> 2 
le = LabelEncoder()
y = le.fit_transform(label) 
lb = LabelBinarizer()
lb.fit(label)
print(y)
print(le.inverse_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))
train_data = lgb.Dataset(data=X_train, label=y_train)
test_data = lgb.Dataset(data=X_test,label=y_test)


# 训练参数
param = {'num_leaves':30, 'objective':'multiclass', 'num_class':3, 'metric': 'multi_logloss', 'learning_rate':0.05}
num_round = 10

#训练过程
clf = lgb.train(param, train_data, num_round, valid_sets=[test_data])

y_pred = clf.predict(X_test)
#print(y_pred)
y_test_val = le.inverse_transform(y_test)
print(y_test_val)
y_pred_val = lb.inverse_transform(y_pred)
print(y_pred_val)
pos = 0
for i in range(len(y_pred)):
  if y_pred_val[i] == y_test_val[i]:
    pos += 1
auc = pos / len(y_pred_val)
print("Accurary:", auc)


# 保存模型
scaler_path = './Q10_standard_scale_model.pkl'
enc_path = './Q10_one_hot_encoder.pkl'
le_path = './Q10_label_encoder.pkl'
lb_path = './Q10_label_binarizer.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
with open(lb_path, 'wb') as f:
    pickle.dump(lb, f)
    
clf.save_model('./Q10_lgb_gbdt_model.txt')


'''
classifier = make_pipeline(
  ColumnTransformer([
    ('num', StandardScaler(), numerical_names),
    ('cat', OneHotEncoder(), categorical_names)]
  ),
  lgb.LGBMClassifier(parameters={'num_leaves':31, 'num_trees':100, 'objective':'binary'}, learning_rate=0.05))

oof_pred = cross_val_predict(classifier, 
  X_train, 
  y_train, 
  cv=5,
  method="predict_proba")

print("Cross validation AUC {:.4f}".format(roc_auc_score(train[target_column], oof_pred[:,1])))
'''