import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#表路径
path1 = "/home/test/experiments/test_raven/Expedia/S_listings.csv"
path2 = "/home/test/experiments/test_raven/Expedia/R1_hotels.csv"
path3 = "/home/test/experiments/test_raven/Expedia/R2_searches.csv"
#读取csv表
S_listings = pd.read_csv(path1)
R1_hotels = pd.read_csv(path2)
R2_searches = pd.read_csv(path3)
#连接3张表
data = pd.merge(pd.merge(S_listings, R1_hotels, how = 'inner'), R2_searches, how = 'inner').loc[0:1, :]
#8 numerical, 20 categorical
numerical = np.array(data.loc[:, ['prop_location_score1', 'prop_location_score2', 'prop_log_historical_price', 'price_usd',
                                  'orig_destination_distance', 'prop_review_score', 'avg_bookings_usd', 'stdev_bookings_usd']])
categorical = np.array(data.loc[:, ['position', 'prop_country_id', 'prop_starrating', 'prop_brand_bool', 'count_clicks',
                                    'count_bookings', 'year', 'month', 'weekofyear', 'time', 'site_id', 'visitor_location_country_id',
                                    'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                                    'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 'random_bool']])

#standard scaling & one-hot encoding & LogisticRegression Model
scaler_path = '/home/test/experiments/test_raven/Expedia/expedia_standard_scale_model.pkl'
enc_path = '/home/test/experiments/test_raven/Expedia/expedia_one_hot_encoder.pkl'
model_path = '/home/test/experiments/test_raven/Expedia/expedia_lr_model.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(enc_path, 'rb') as f:
    enc = pickle.load(f)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

#获i取训练数据并预测
X = np.hstack((scaler.transform(numerical) , enc.transform(categorical).toarray()))
print('需预测数据的维度:{}'.format(X.shape))
y = model.predict(X)
print(y)
