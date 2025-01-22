import torch
import pickle

# 从.pth文件中加载数据
data = torch.load('node_features.pth')

# 将数据存储为.pkl文件
with open('node_features.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Data saved as pkl file.")