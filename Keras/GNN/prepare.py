import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

zip_file = keras.utils.get_file(
    fname  ="cora.tgz",
    origin ="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=  True ,
)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")

citations = pd.read_csv(#的论文 ID 引用的论文 ID
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)#引文数据集

print("Citations shape:", citations.shape)#(5429, 2)
print(type(citations))
print(citations.values[0])

citations.sample(frac=1).head()

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(  #论文（2708个）、术语（1433）、主题(7个)
    os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
)#主题数据集

print("Papers shape:", papers.shape)  #(2708, 1435)
print(papers.sample(5).T)
print(papers.subject.value_counts())  #统计每个主题的论文数量

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

#图中的每个节点代表一篇论文，节点的颜色与其主题相对应
plt.figure(figsize=(10, 10))
colors = papers["subject"].tolist()
cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
subjects = list(papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"])
nx.draw_spring(cora_graph, node_size=15, node_color=subjects)

print("papers.groupby",type( papers.groupby("subject") ))

train_data, test_data = [], []
for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training.
    random_selection = np.random.rand(len(group_data.index)) <= 0.5#生成随机数字
    train_data.append(group_data[random_selection])#添加到训练集
    test_data.append(group_data[~random_selection])#添加到测试集

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

#应该是根据主题数据集拆分出来的结果
print("Train data shape:", train_data.shape) # (1346, 1435)
print("Test data shape:", test_data.shape)   # (1362, 1435)
print(type(train_data))
print(type(test_data ))
train_data.to_csv('train_data.csv')#写
test_data.to_csv('test_data.csv')#写

edges = citations[["source", "target"]].to_numpy().T
print(type(edges))
print(edges.shape)
np.savez('data',edges=edges)
