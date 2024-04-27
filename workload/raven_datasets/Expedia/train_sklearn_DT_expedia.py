import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

#expedia

#表路径
path1 = "./S_listings.csv"
path2 = "./R1_hotels.csv"
path3 = "./R2_searches.csv"

#读取csv表
S_listings = pd.read_csv(path1)
R1_hotels = pd.read_csv(path2)
R2_searches = pd.read_csv(path3)

#连接3张表
data = pd.merge(pd.merge(S_listings, R1_hotels, how = 'inner'), R2_searches, how = 'inner')
#print(data.isnull().any())    #检测缺失值
data.dropna(inplace=True)      #删除NaN

#获取分类label
y = np.array(data.loc[:, 'promotion_flag'])
#8 numerical, 20 categorical
numerical = np.array(data.loc[:, ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                                  'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']])
categorical = np.array(data.loc[:, ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                                    'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id',
                                    'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                                    'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']])

#standard scaling & one-hot encoding
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder()
one_hot_model = enc.fit(categorical)

#获取训练数据
X = np.hstack((scaler.transform(numerical) , enc.transform(categorical).toarray()))
X_train, X_test, y_train, y_test = train_test_split(X, y)
print('训练集维度:{}\n测试集维度:{}'.format(X_train.shape, X_test.shape))

#训练过程
dt = DecisionTreeClassifier()                                           #决策树模型
dt.fit(X_train, y_train)                                                #训练

y_prob = dt.predict_proba(X_test)[:, 1]                                 #预测结果为1的概率
y_pred = dt.predict(X_test)                                             #预测结果
fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob)                 #真阳率、伪阳率、阈值
auc = metrics.auc(fpr, tpr)                                             #AUC
score = metrics.accuracy_score(y_test, y_pred)                          #模型准确率
print('模型准确率:{}\nAUC得分:{}'.format(score, auc))

#保存模型
scaler_path = './expedia_standard_scale_model.pkl'
enc_path = './expedia_one_hot_encoder.pkl'
file_path = './expedia_dt_model.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)
with open(file_path, 'wb') as f:
    pickle.dump(dt, f)
