import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pickle

# 读取数据
path1 = "./lineitem.tbl"
path2 = "./customer.tbl"
path3 = "./orders.tbl"
path4 = "./nation.tbl"
path5 = "./supplier.tbl"

lineitem = pd.read_table(path1, names=['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 
                                       'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 
                                       'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], sep='|', index_col=False, nrows=1000) # nrows=1000000
#print(lineitem.info())
customer = pd.read_table(path2, names=['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 
                                       'c_comment'], sep='|', index_col=False, nrows=500) # nrows=500000
#print(customer.info())
orders = pd.read_table(path3, names=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 
                                     'o_clerk', 'o_shippriority', 'o_comment'], sep='|', index_col=False)
#print(orders.info())

nation = pd.read_table(path4, names=['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], sep='|', index_col=False)
#print(nation.info())
supplier = pd.read_table(path5, names=['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'], sep='|', index_col=False)
#print(nation.info())

# join
#data = pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey')
data = pd.merge(pd.merge(pd.merge(pd.merge(customer, orders, how = 'inner', left_on = 'c_custkey', right_on = 'o_custkey'), 
                                  lineitem, how = 'inner', left_on = 'o_orderkey', right_on = 'l_orderkey'), 
                         nation, how = 'inner', left_on = 'c_nationkey', right_on = 'n_nationkey'), 
                supplier, how = 'inner', left_on = 'l_suppkey', right_on = 's_suppkey')
#print(data.info())
select_cols = ['c_custkey', 'c_acctbal', 'o_orderkey', 'o_orderstatus', 'o_totalprice', 'o_orderpriority', 'o_clerk', 
               'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipinstruct', 'l_shipmode',
               'n_nationkey', 'n_regionkey', 's_suppkey', 's_acctbal']
data = data[select_cols]
print(data.info())


# 获取特征和数据预处理
numerical_names = ['c_acctbal', 'o_totalprice', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 's_acctbal'] # 
categorical_names = ['o_orderstatus', 'l_returnflag', 'l_linestatus', 'l_shipinstruct', 'l_shipmode', 'n_nationkey', 'n_regionkey'] # 
features = numerical_names + categorical_names
numerical = np.array(data.loc[:, numerical_names])
categorical = np.array(data.loc[:, categorical_names])
label = np.array(data.loc[:, 'o_orderpriority'])

# 标准化和独热编码
scaler = StandardScaler()
standard_scale_model = scaler.fit(numerical)
enc = OneHotEncoder()
one_hot_model = enc.fit(categorical)

X_numerical = scaler.transform(numerical)
X_categorical = enc.transform(categorical).toarray()
X = np.hstack((X_numerical, X_categorical))

# '1-URGENT'->0, '2-HIGH'->1, '3-MEDIUM'->2, '4-NOT SPECIFIED'->3, '5-LOW'->4 
#le = LabelEncoder()
#y = le.fit_transform(label) 
lb = LabelBinarizer()
y = lb.fit_transform(label)
print(y)
print(lb.inverse_transform(y))

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义全局变量
n_epochs = 10
batch_size = 200

# PyTorch数据集
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

# 创建数据加载器
train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义PyTorch模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# 初始化模型、损失函数和优化器
model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
for epoch in range(n_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')

# 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不计算梯度，减少内存消耗和加速计算
    test_loss = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss}')

# 推理（假设 new_data 是已经预处理好的新数据）
new_data = data[0:10000]  # 仅作为示例，实际应用中应替换为真实的新数据
new_numerical = np.array(new_data.loc[:, numerical_names])
new_categorical = np.array(new_data.loc[:, categorical_names])
new_X_numerical = scaler.transform(new_numerical)
new_X_categorical = enc.transform(new_categorical).toarray()
new_X = np.hstack((new_X_numerical, new_X_categorical))
new_X = torch.tensor(new_X, dtype=torch.float32)
new_y = new_data.loc[:, 'o_orderpriority']

model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不计算梯度
    predictions = model(new_X).numpy()
    print('Predictions:', predictions)  # 输出预测结果
    answer = lb.inverse_transform(predictions)
    print('Answers:', answer) # 输出对应结果
    print('Auc:', accuracy_score(new_y, answer)) # 输出正确率

# 保存模型
torch.save(model, 'Q5_pytorch_mlp.model')  # 保存模型本身
#torch.save(model.state_dict(), 'Q5_pytorch_mlp.pth')  # 保存模型参数

# 保存标签编码器
lb_path = './Q5_label_binarizer.pkl'
with open(lb_path, 'wb') as f:
    pickle.dump(lb, f)

# 保存标准化和独热编码器
scaler_path = './Q5_standard_scale_model.pkl'
enc_path = './Q5_one_hot_encoder.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(standard_scale_model, f)
with open(enc_path, 'wb') as f:
    pickle.dump(one_hot_model, f)

