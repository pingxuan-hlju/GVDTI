from numpy import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
FloatTensor = torch.FloatTensor
from tool import laplacians,max_min_normalize
import numpy as np


class Attention_GCN(nn.Module):
    def __init__(self):
        super(Attention_GCN, self).__init__()
        self.H = Parameter(torch.from_numpy(random.random((1, 300))).float())
        self.W0 = Parameter(torch.from_numpy(random.random((708, 200))).float())
        self.W1 = Parameter(torch.from_numpy(random.random((200, 100))).float())
        self.W2 = Parameter(torch.from_numpy(random.random((100, 200))).float())
        self.W3 = Parameter(torch.from_numpy(random.random((200, 708))).float())

        self.att_layer = nn.Sequential(nn.Linear(708, 300, bias=True), nn.Tanh())
        self.encoder = nn.Sequential(
            nn.Softmax(dim=-1)
        )
        self.decoder = nn.Sequential(
            nn.Sigmoid()
        )
    def forward(self, X1, A1):
        M = zeros((A1.shape[0], 1), dtype=float64)
        A2 = A1/1
        X2 = X1/1
        for i in range(A1.shape[0]):
            K = self.att_layer(X1[i])
            K = torch.unsqueeze(K, dim=1).type(torch.FloatTensor)
            K = self.H.mm(K)
            M[i] = K.detach().numpy()
        M = torch.from_numpy(M)
        M = torch.squeeze(M, dim=1).type(torch.FloatTensor)
        score = F.softmax(M, dim=0)

        for i in range(X1.shape[0]):
            X2[i] = X1[i] * score[i]
        Z1 = self.encoder(A2.mm(X2).mm(self.W0))
        Z = self.decoder(A2.mm(Z1).mm(self.W1))
        Z2 = self.encoder(A2.mm(Z).mm(self.W2))
        X_p = self.decoder(A1.mm(Z2).mm(self.W3))
        return X_p, Z

Ar = loadtxt("max_ordata/Similarity_Matrix_Drugs.txt")
Xr = loadtxt("max_ordata/mat_drugDP.txt")  #drug-disease+protein   protein-disease+drug
# Ar = loadtxt("max_ordata/Similarity_Matrix_proteins.txt")
# Xr = loadtxt("max_ordata/mat_proteinDD.txt")  #drug-disease+protein   protein-disease+drug
Ar1 = torch.from_numpy(laplacians(Ar)).float()
Xr1 = torch.from_numpy(max_min_normalize(Xr)).float()

attention_GCN = Attention_GCN()
optimizer = torch.optim.Adam(attention_GCN.parameters(), lr=0.3)
loss_func = nn.MSELoss()

for epoch in range(150):  #260
    decoded, encoded = attention_GCN(Xr1,Ar1)
    loss = loss_func(decoded, Xr1)  # 计算损失函数

    print('epoch {}, loss is {}'.format(epoch + 1, loss.item()))
    print("###########################")
    optimizer.zero_grad()               # 梯度清零
    loss.backward()                     # 反向传播
    optimizer.step()                    # 梯度优化

savetxt("maxdrug_feature.txt", encoded.detach().numpy())
#savetxt("maxprotein_feature.txt", encoded.detach().numpy())

middrug = np.loadtxt("maxdrug_feature.txt")
midprotein = np.loadtxt("maxprotein_feature.txt")
dr_p = np.loadtxt("max_ordata/mat_drug_protein.txt")
X_t = np.loadtxt("maxdata/X_train1.txt")
#Y_t = np.loadtxt("mindata/Y_train1.txt")
X_te = np.loadtxt("maxdata/X_test1.txt")
#Y_te = np.loadtxt("mindata/Y_test1.txt")
#gcn_mid = torch.from_numpy(np.hstack((middrug, midprotein))).float()

def load_data2(train_index,middrug,midprotein):

    x = np.zeros((0, 200))
    for j in range(train_index.shape[0]):
        x_A = int(train_index[j][0])
        y_A = int(train_index[j][1])
        drugp_simdrug = np.hstack((middrug[x_A], midprotein[y_A]))
        x = np.vstack((x,drugp_simdrug))
    #print('x.size()', x.shape)
    return x

train = load_data2(X_t, middrug, midprotein)
np.savetxt("maxgcn_train.txt", train)

test = load_data2(X_te, middrug, midprotein)
np.savetxt("maxgcn_test.txt", test)
