import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import torch.nn.functional as F
import funcs
import data_loader
from sklearn.model_selection import StratifiedKFold
import pandas as pd

funcs.setup_seed(1)

b_size, n_hidden = 64, 128
lr, wd = 1e-4, 1e-4
num_epoches = 200
# 使用GPU加速
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

# 输入id并映射到整数
Drug_id, Protein_id = data_loader.Get_id()
n_drugs, n_proteins = len(Drug_id), len(Protein_id)
dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
print("number of Drug: ", n_drugs)
print("number of Protein ", n_proteins)

Dr_finger = data_loader.Get_finger()
P_seq = data_loader.Get_seq()

def Trans_feature(Feature):
    for i in Feature:
        Feature[i] = torch.as_tensor(torch.from_numpy(Feature[i]), dtype=torch.float32).to(device)
    return Feature

Dr_finger = Trans_feature(Dr_finger)
P_seq = Trans_feature(P_seq)

class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(dim, dim*16), nn.ReLU(), nn.Linear(dim*16, dim))

    def forward(self, x):
        avg = self.avg_pool(x).squeeze(2)
        max = self.max_pool(x).squeeze(2)
        att1 = self.fc(avg)
        att2 = self.fc(max)
        att = F.softmax((att1+att2), dim=1)
        out = att.unsqueeze(2).expand(x.size())
        out = x * out
        return out

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.ECFP4 = nn.Sequential(nn.Linear(in_features=1024, out_features=n_hidden), nn.ReLU())
        self.FCFP4 = nn.Sequential(nn.Linear(in_features=1024, out_features=n_hidden), nn.ReLU())
        self.Pubchem = nn.Sequential(nn.Linear(in_features=881, out_features=n_hidden), nn.ReLU())
        self.RDK = nn.Sequential(nn.Linear(in_features=2048, out_features=n_hidden), nn.ReLU())
        self.MACCS = nn.Sequential(nn.Linear(in_features=166, out_features=n_hidden), nn.ReLU())
        self.KSCTriad = nn.Sequential(nn.Linear(in_features=343 * 4, out_features=n_hidden), nn.ReLU())
        self.PAAC = nn.Sequential(nn.Linear(in_features=22, out_features=n_hidden), nn.ReLU())
        self.CTD = nn.Sequential(nn.Linear(in_features=273, out_features=n_hidden), nn.ReLU())
        self.CKSAAP = nn.Sequential(nn.Linear(in_features=1600, out_features=n_hidden), nn.ReLU())
        self.TPC = nn.Sequential(nn.Linear(in_features=8000, out_features=n_hidden), nn.ReLU())

    def forward(self, Drug_feature, Protein_feature, x_dr, x_p):
        ecfp4, fcfp4 = Drug_feature['ecfp4'][x_dr], Drug_feature['fcfp4'][x_dr]
        maccs, pubchem = Drug_feature['maccs'][x_dr], Drug_feature['pubchem'][x_dr]
        rdk = Drug_feature['rdk'][x_dr]
        ksctriad, cksaap = Protein_feature['KSCTriad'][x_p], Protein_feature['CKSAAP'][x_p]
        paac, tpc = Protein_feature['PAAC'][x_p], Protein_feature['TPC'][x_p]
        ctd = Protein_feature['CTD'][x_p]
        ecfp4, fcfp4 = self.ECFP4(ecfp4), self.FCFP4(fcfp4)
        maccs, pubchem= self.MACCS(maccs), self.Pubchem(pubchem)
        rdk = self.RDK(rdk)
        ksctriad, cksaap = self.KSCTriad(ksctriad), self.CKSAAP(cksaap)
        paac, tpc = self.PAAC(paac), self.TPC(tpc)
        ctd = self.CTD(ctd)
        finger = torch.cat((ecfp4, fcfp4, pubchem, maccs, rdk), dim=1)
        seq = torch.cat((ksctriad, cksaap, tpc, paac, ctd), dim=1)
        return finger, seq

encoder_layer = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=1, dim_feedforward=128, dropout=0.1, batch_first=True)

class DNNNet(nn.Module):
    def __init__(self):
        super(DNNNet, self).__init__()
        self.feat_embedding = Embedding()
        self.dr_channel_attention = ChannelAttention(dim=5)
        self.p_channel_attention = ChannelAttention(dim=5)
        self.dr_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.p_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.connected_layer1 = nn.Sequential(nn.Linear(in_features=n_hidden * 10, out_features=n_hidden * 2), nn.BatchNorm1d(num_features=n_hidden * 2), nn.ReLU())
        self.connected_layer2 = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=n_hidden), nn.BatchNorm1d(num_features=n_hidden), nn.ReLU())
        self.connected_layer3 = nn.Sequential(nn.Linear(in_features=n_hidden, out_features=64), nn.BatchNorm1d(num_features=64), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(in_features=64, out_features=1), nn.Sigmoid())
    # 前向传播
    def forward(self, Dr_finger, P_seq, x_dr, x_p):
        dr_att, p_att = self.feat_embedding(Dr_finger, P_seq, x_dr, x_p)
        dr_att, p_att = self.dr_channel_attention(dr_att), self.p_channel_attention(p_att)
        dr_att, p_att = self.dr_attention(dr_att), self.p_attention(p_att)
        dr_att, p_att = dr_att.transpose(0, 1), p_att.transpose(0, 1)
        new_drug = torch.cat((dr_att[0], dr_att[1], dr_att[2], dr_att[3], dr_att[4]), dim=1)
        new_protein = torch.cat((p_att[0], p_att[1], p_att[2], p_att[3], p_att[4]), dim=1)
        h_dr_p = torch.cat((new_drug, new_protein), dim=1)
        out = self.connected_layer1(h_dr_p)
        out = self.connected_layer2(out)
        out = self.connected_layer3(out)
        out = self.output(out)
        return out


P = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/DTI_8020.csv", dtype=str, delimiter=",", skiprows=1)
N = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/negative samples/neg_DTI-net_8020.csv", dtype=str, delimiter=",", skiprows=1)
X, Y = funcs.Get_sample(P, N, dr_id_map, p_id_map)

skf = StratifiedKFold(n_splits=10, shuffle=True)

test_drug = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/Drug_1409.csv", dtype=str, delimiter=",", skiprows=1)
test_protein = np.loadtxt("D:/Users/czx/PycharmProjects/1-1HGDTI-code/DrugBank dataset/DTI-benchmark_set/Protein_1648.csv", dtype=str, delimiter=",", skiprows=1)
test_X = []
test_Y = []
for i in range(len(test_drug)):
    for j in range(len(test_protein)):
        test_X.append([dr_id_map[test_drug[i]], p_id_map[test_protein[j]]])
        test_Y.append([0])
test_X, test_Y = np.array(test_X), np.array(test_Y)

print("number of DTI: ", len(P))
print("number of Negative DTI ", len(N))

all_output_scores1, all_output_scores2, all_output_scores3, all_output_scores4, \
all_output_scores5, all_output_scores6, all_output_scores7, all_output_scores8, \
all_output_scores9, all_output_scores10 = [],[],[],[],[],[],[],[],[],[]

this_fold = 0
for train_index, dev_index in skf.split(X, Y):
    this_fold = this_fold + 1
    print("Fold: ", this_fold)
    X_train, X_dev = X[train_index], X[dev_index]
    Y_train, Y_dev = Y[train_index], Y[dev_index]
    train_loader = funcs.get_train_loader(X_train, Y_train, b_size)
    dev_loader = funcs.get_train_loader(X_dev, Y_dev, b_size)
    test_loader = funcs.get_test_loader(test_X, test_Y, b_size=len(test_protein))
    losses = nn.BCELoss()
    model = DNNNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=40, verbose=False)
    best_auc, best_epoch, best_extra = 0, 0, 0
    best_test = []
    for epoch in range(num_epoches):
        train_loss = 0
        train_scores, train_scores_label, train_labels = [], [], []
        for step, (batch_x, batch_y) in enumerate(train_loader):
            model.train()
            b_x = batch_x.long().to(device)
            b_y = torch.squeeze(batch_y.float().to(device), dim=1)
            b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
            b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
            output = model(Dr_finger, P_seq, b_x_dr, b_x_p)
            score = torch.squeeze(output, dim=1)
            loss = losses(score, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()  # 反向传播
            train_loss += loss.item()  # 记录误差
            scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
            train_scores = np.concatenate((train_scores, scores))
            train_labels = np.concatenate((train_labels, label))
        train_scores_label = funcs.computer_label(train_scores, 0.5)
        train_avloss = train_loss / len(train_loader)
        train_acc = skm.accuracy_score(train_labels, train_scores_label)
        train_auc = skm.roc_auc_score(train_labels, train_scores)
        dev_scores, dev_labels = [], []
        test_scores, test_scores_label, test_labels = [], [], []

        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(dev_loader):
                model.eval()
                b_x = batch_x.long().to(device)
                b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                output = model(Dr_finger, P_seq, b_x_dr, b_x_p)
                score = torch.squeeze(output, dim=1)
                scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                dev_scores = np.concatenate((dev_scores, scores))
                dev_labels = np.concatenate((dev_labels, label))
            dev_auc = skm.roc_auc_score(dev_labels, dev_scores)
            scheduler.step(dev_auc)
            if dev_auc >= best_auc:
                best_auc = dev_auc
                best_epoch = epoch
                torch.save(model, "test_model" + str(this_fold) + ".pt")
        print('epoch:{},Train Loss: {:.4f},Train Acc: {:.4f},Train Auc: {:.4f},Dev Auc: {:.4f}'.format(epoch,
                                                                                                       train_avloss,
                                                                                                       train_acc,
                                                                                                       train_auc,
                                                                                                       dev_auc))

    my_model = torch.load("test_model" + str(this_fold) + ".pt")
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            my_model.eval()
            b_x = batch_x.long().to(device)
            b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
            b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
            output = model(Dr_finger, P_seq, b_x_dr, b_x_p)
            score = torch.squeeze(output, dim=1)
            scores = score.cpu().detach().numpy()
            globals()['all_output_scores' + str(this_fold)].append(scores)
    globals()['all_output_scores' + str(this_fold)] = np.array(globals()['all_output_scores' + str(this_fold)])

all_output_scores = all_output_scores1 + all_output_scores2 + all_output_scores3 + all_output_scores4 + all_output_scores5 + all_output_scores6 + all_output_scores7 + all_output_scores8 + all_output_scores9 + all_output_scores10
all_output_scores = all_output_scores / 10
all_output_pandas = pd.DataFrame(all_output_scores)
all_output_pandas.to_csv("All_scores_10fold.csv", index=False)


