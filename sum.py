import numpy as np
import torch
import torch.nn as nn
from torch.autograd import no_grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

flag_train1 = True
flag_test1 = True
use_cuda = False

if use_cuda and torch.cuda.is_available():
    print("Using GPU")
    cuda = torch.device("cuda")
else:
    print("Using CPU")
    cuda = torch.device("cpu")

#############################划分数据集#########################################
print("###############")
dr_p = np.loadtxt("max_ordata/mat_drug_protein.txt")
X_t = np.loadtxt("maxdata/X_train1.txt")
Y_t = np.loadtxt("maxdata/Y_train1.txt")
X_te = np.loadtxt("maxdata/X_test1.txt")
Y_te = np.loadtxt("maxdata/Y_test1.txt")

vae_train1 = np.loadtxt("maxvae_train1.txt")
vae_test1 = np.loadtxt("maxvae_test1.txt")
cnn_train1 = np.loadtxt("maxcnn_train1.txt")
cnn_test1 = np.loadtxt("maxcnn_test1.txt")
gcn_train1 = np.loadtxt("maxgcn_train1.txt")
gcn_test1 = np.loadtxt("maxgcn_test1.txt")


ratio = {"gcn":0.3,"cnn":0.3,"vae":0.4}
sum_train1 = np.hstack((ratio["gcn"]*gcn_train1, ratio["cnn"]*cnn_train1, ratio["vae"]*vae_train1))
print("sum",sum_train1.shape)
sum_test1 = np.hstack((ratio["gcn"]*gcn_test1, ratio["cnn"]*cnn_test1, ratio["vae"]*vae_test1))
print("sum",sum_test1.shape)

auto_train = torch.Tensor(sum_train1).float().to(cuda)
auto_train_label = torch.Tensor(Y_t).long().to(cuda)

auto_test = torch.Tensor(sum_test1).float().to(cuda)
auto_test_label = torch.Tensor(Y_te).long().to(cuda)

print("##分类器数据封装##", auto_train.shape, auto_train_label.shape )

train_dataset = Data.TensorDataset(auto_train, auto_train_label)
auto_train_data = Data.DataLoader( dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True )

test_dataset = Data.TensorDataset(auto_test, auto_test_label)
auto_test_data = Data.DataLoader( dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True )

class classfier(nn.Module):
    def __init__(self):
        super(classfier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4200, 2000),
            nn.PReLU(),  # nn.PReLU(), 
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


classLoss_function = torch.nn.CrossEntropyLoss()
cla = classfier().to(cuda)  # 实例化网络
cla_optimizer = torch.optim.Adam(cla.parameters(), lr=0.0001)

if flag_train1:
    loss_list = []
    acc_list = []
    cla.train()
    for epoch in range(20):
        cnn_loss = 0.
        train_correct = 0.

        print("epoch", epoch, '...')
        for x, y in auto_train_data:
            output = cla(x)
            loss = classLoss_function(output, y)

            cla_optimizer.zero_grad()
            loss.backward()
            cla_optimizer.step()

            cnn_loss += loss.item()
            ###计算准确率#####
            _, pred = output.max(1)
            num_correct = (pred == auto_train_label).sum().item()
            train_correct += num_correct
        loss_list.append(cnn_loss / len(auto_train_data))
        acc_list.append(train_correct / len(auto_train))

        fig, axis = plt.subplots(2, 1)
        axis[0].plot(loss_list, label="loss")
        axis[1].plot(acc_list, label="acc")
        axis[1].set_ylim(0, 1.3)
        axis[0].legend()
        axis[1].legend()

        print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'.format(epoch, cnn_loss / len(auto_train_data),
                                                                        train_correct / len(auto_train)))


if flag_test1:
    print("开始测试...")
    test_correct = 0.
    o = np.zeros((0, 2))
    cla.eval()
    with no_grad():

        for x, label in auto_test_data:
            out = cla(x)
            test_out = F.softmax(out, dim=1)
            # 计算准确率
            _, pred_y = test_out.max(1)
            num_correct = (pred_y == label).sum().item()
            test_correct += num_correct
            o = np.vstack((o, test_out.detach().cpu().numpy()))

    print("测试完成", '预测正确数量:{}, Test Auc: {:.6f}'.format(test_correct, test_correct / len(auto_test)))



