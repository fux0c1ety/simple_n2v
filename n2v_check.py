from n2v import Node2Vec
import networkx as nx
from torch_geometric.utils.convert import from_networkx
import torch


G = nx.complete_graph(5)
tensor = from_networkx(G)
model = Node2Vec(tensor.edge_index, embedding_dim=2, walk_length=3, context_size=2)
loader = model.loader()
counter = 0
for i in loader:
    print(i)
    counter+=1
    if counter>5:
        break

optim = torch.optim.Adam(model.parameters(), lr=0.01)
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos, neg in loader:
            optimizer.zero_grad()
            loss = model.loss(pos, neg)
            loss.backward()
            optimizer.step()
            total_loss+=loss
    return total_loss/len(loader)

for epoch in range(100):
    loss = train(model, loader, optim)
    print(loss)
