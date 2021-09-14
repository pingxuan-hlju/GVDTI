import numpy as np
import torch
import torch.nn as nn
from torch.autograd import no_grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data as Data

"药物蛋白，药物相似，蛋白相似,蛋白疾病, 药物疾病 maxdata 2*7823"

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 1), stride=2, padding=1),
            #nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.MaxPool2d(2,2),  #(1,6)
            nn.Conv2d(8, 16, kernel_size=(2, 2), stride=2, padding=1),
            #nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier1 = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(16 * 1 * 489, 4000),
            nn.PReLU(),
            # nn.Dropout(),
            nn.Linear(4000, 2000),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(16*1*489, 1024),
            nn.PReLU(),
            nn.Linear(1024, 2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        encode = self.encoder(x)
        #print("encode: ", encode.shape)
        out1=encode.view(encode.size(0),16*1*489)
        out2 = self.classifier1(out1)
        out=self.classifier(out1)
        return out, out2

class MengData(Dataset):
    def __init__(self, train_index, drug_sim, protein_sim, drug_p, drug_d, protein_d, device):
        super(Dataset, self).__init__()
        self.train_index = train_index
        self.drug_sim = drug_sim
        self.protein_d = protein_d
        self.protein_sim = protein_sim
        self.drug_p = torch.from_numpy(drug_p).long().to(cuda)
        self.drug_d = drug_d
        self.protein_d = protein_d
        self.drpd = torch.from_numpy(np.hstack((drug_p, drug_sim, drug_d))).float().to(cuda)
        self.pdrd = torch.from_numpy(np.hstack((protein_sim, drug_p.T, protein_d))).float().to(cuda)

    def __getitem__(self, index):
        x_A = int(self.train_index[index][0])  # 21
        y_A = int(self.train_index[index][1])  # 394

        drugp_simdrug = self.drpd[x_A]
        simpr_drugp = self.pdrd[y_A]

        features = torch.stack((drugp_simdrug, simpr_drugp)).view(1, 2, 7823)
        label = self.drug_p[x_A, y_A]
        # features = torch.FloatTensor(features)
        # label = torch.LongTensor(np.array(label)).to(cuda)
        return features, label

    def __len__(self):
        return len(self.train_index)


def load_data(train_index, drug_sim, protein_sim, BATCH_SIZE, drug_p, drug_d, protein_d):
    torch_dataset = MengData(train_index, drug_sim, protein_sim, drug_p, drug_d, protein_d, cuda)
    data_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    return data_loader

Simila_protein = np.loadtxt("max_ordata/Similarity_Matrix_Proteins.txt")
Simila_drug = np.loadtxt("max_ordata/Similarity_Matrix_Drugs.txt")
protein_d = np.loadtxt("max_ordata/mat_protein_disease.txt")
drug_d = np.loadtxt("max_ordata/mat_drug_disease.txt")
dr_p = np.loadtxt("max_ordata/mat_drug_protein.txt")

#############################划分数据集#########################################
X_t = np.loadtxt("maxdata/X_train1.txt")
Y_t = np.loadtxt("maxdata/Y_train1.txt")
X_te = np.loadtxt("maxdata/X_test1.txt")
Y_te = np.loadtxt("maxdata/Y_test1.txt")

#############################embed matrix#########################################
flag_train = True
flag_test = True
use_cuda = False

if use_cuda and torch.cuda.is_available():
    print("Using GPU")
    cuda = torch.device("cuda")
else:
    print("Using CPU")
    cuda = torch.device("cpu")


print("配置Dataloader...")
train_loader = load_data(X_t, Simila_drug, Simila_protein, 1, dr_p, drug_d, protein_d)
test_loader = load_data(X_te, Simila_drug, Simila_protein, 1, dr_p, drug_d, protein_d)
print('train_loader', train_loader)
print('test_loader', test_loader)

#####################train#################################################
Loss_function = torch.nn.CrossEntropyLoss()
cnn = Cnn().to(cuda)  # 实例化网络
cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
if flag_train:
    loss_list = []
    acc_list = []
    cnn.train()
    for epoch in range(50):
        cnn_loss = 0.
        train_correct = 0.
        cnn_train = np.zeros((0, 2000))
        print("epoch", epoch, '...')
        for x, y in train_loader:
            #print(x.shape)
            output, encode = cnn(x)
            loss = Loss_function(output, y)

            cnn_optimizer.zero_grad()
            loss.backward()
            cnn_optimizer.step()
            cnn_train = np.vstack((cnn_train, encode.detach().cpu().numpy()))
            cnn_loss += loss.item()
            ###计算准确率#####
            _, pred = output.max(1)
            # print(output)
            num_correct = (pred == y).sum().item()
            train_correct += num_correct

        loss_list.append(cnn_loss / len(train_loader))
        acc_list.append(train_correct / len(X_t))

        fig, axis = plt.subplots(2, 1)
        axis[0].plot(loss_list, label="loss")
        axis[1].plot(acc_list, label="acc")
        axis[1].set_ylim(0, 1.3)
        axis[0].legend()
        axis[1].legend()
        plt.savefig("lossmaxcnn.jpg")

        print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'.format(epoch, cnn_loss / len(train_loader),
                                                                        train_correct / len(X_t)))

    np.savetxt("maxcnn_train.txt", cnn_train)
    print("cnn_train.shape", cnn_train.shape)

####################################--test--############################################################
if flag_test:
    print("开始测试...")
    test_correct = 0.
    o = np.zeros((0, 2))
    cnn_test = np.zeros((0, 2000))
    cnn.eval()
    with no_grad():

        for x, label in test_loader:
            print("##########")
            out,encode = cnn(x)
            test_out = F.softmax(out, dim=1)

            cnn_test = np.vstack((cnn_test, encode.detach().cpu().numpy()))
            # 计算准确率
            _, pred_y = test_out.max(1)
            num_correct = (pred_y == label).sum().item()
            test_correct += num_correct
            o = np.vstack((o, test_out.detach().cpu().numpy()))

    print("测试完成", '预测正确数量:{}, Test Auc: {:.6f}'.format(test_correct, test_correct / len(X_te)))
    np.savetxt("cnn_outmax.txt", o)
    np.savetxt("maxcnn_test.txt", cnn_test)
    print("cnntest.shape", cnn_test.shape)

