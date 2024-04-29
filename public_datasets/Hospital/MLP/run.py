import os
import numpy
import torch
import random
from utils import Config, CLF_Model, REG_Model
from clf_model.MLP_clf import MLP

# 随机数种子确定
seed = 1129
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


# 分类
if __name__ == '__main__':
	# 读数据
    config = Config(
        data_path="../LengthOfStay.csv",
        name="Hospital",
        batch_size=128,
        learning_rate=0.000005,
        epoch=200
    )
    # 搭模型
    clf = MLP(
        input_n=len(config.input_col),
        output_n=len(config.output_class),
        num_layer=4,
        layer_list=[10,10,10,10],
        dropout=0.5
    )
    print(clf)
    # 训练模型并评价模型
    model = CLF_Model(clf, config)
    model.run()
