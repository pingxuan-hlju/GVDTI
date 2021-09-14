import numpy as np
import torch
import torch.nn as nn
from torch.autograd import no_grad, Variable
from torch.utils.data import Dataset
import torch.utils.data as Data


"药物蛋白，药物相似，蛋白相似,蛋白疾病, 药物疾病 2*7823"

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 1), stride=2, padding=1),  # 2
            nn.PReLU(),
            nn.MaxPool2d(2,2),  # 2
            nn.Conv2d(8, 16, kernel_size=(1, 2), stride=2, padding=1),  # 2
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*1*489, 2000)
        )
        self.de_fc = nn.Linear(2000, 16*1*489)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=(2, 3), stride=2, padding=0),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(1, 2), stride=2, padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=(2,2), stride=2, padding=0),
            nn.PReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(2, 3), stride=2, padding=1),
        )

    def reparametrize(self,mu,logvar):
        std=logvar.mul(0.5).exp_()
        eps=torch.FloatTensor(std.size()).normal_()
        eps=Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        encode = self.encoder(x)
        #print("encode: ",encode.shape)       #encode:  torch.Size([1, 16, 1, 489])
        tencode = encode.view(encode.size(0), -1)

        mu = self.fc(tencode)
        logavar = self.fc(tencode)
        encode = self.reparametrize(mu, logavar)
        #print("encode: ", encode.shape)          #encode:  torch.Size([1, 2000])
        de_encode = self.de_fc(encode)
        de_encode = de_encode.reshape(1, 16, 1, 489)
        decode = self.decoder(de_encode)
        #print("decode: ",decode.shape)      #decode:  torch.Size([1, 1, 2, 7823])
        return encode, decode, mu, logavar

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


# 相似性 蛋白 药物 # 蛋白疾病 # 药物疾病 药物蛋白
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
use_cuda = True

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

########################VAE的损失函数##############################################################
reconstruction_function=nn.MSELoss(size_average=False)
def loss_function(recon_x,x,mu,logvar):#生成图，原图，均值，方差
    BCE=reconstruction_function(recon_x,x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD=torch.sum(KLD_element).mul_(-0.5)
    return BCE+KLD
###############################cnn-----train#######################################################
#autoLoss_function = loss_function()
auto = conv_autoencoder().to(cuda)  # 实例化网络
auto_optimizer = torch.optim.Adam(auto.parameters(), lr=0.001)
if flag_train:
    auto.train()
    for epoch in range(50):
        cnn_loss = 0.
        auto_train = np.zeros((0,2000))
        print("train epoch", epoch, '...')
        for x, y in train_loader:
            print("x", x.shape)
            encode, decode, mu,logvar = auto(x)     #encode torch.Size([1, 200])
            loss = loss_function(decode, x, mu, logvar)
            #print("loss",loss)
            auto_optimizer.zero_grad()
            loss.backward()
            auto_optimizer.step()

            auto_train = np.vstack((auto_train, encode.detach().cpu().numpy()))
            cnn_loss += loss.item()

        print('Epoch: {}, Train Loss: {:.8f}'.format(epoch, cnn_loss / len(train_loader)))
    np.savetxt("vae_train.txt", auto_train)
    print("vae_train.shape", auto_train.shape)

#########################test############################################################
if flag_test:
    print("开始测试...")
    auto_test = np.zeros((0, 2000))
    auto.eval()
    with no_grad():
        for x in test_loader:
            print("#########")
            encode, decode, mu, logvar = auto(x)
            auto_test = np.vstack((auto_test, encode.detach().cpu().numpy()))

    print("测试完成")
    np.savetxt("vae_test.txt", auto_test)
    print("vae_test.shape", auto_test.shape)

