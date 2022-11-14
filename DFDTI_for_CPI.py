import torch
import numpy as np
import sklearn.metrics as skm
import torch.nn as nn
import torch.nn.functional as F
import funcs
import data_loader_CPI

funcs.setup_seed(1)

types = ["random","new_drug","new_protein","new_drug_protein"]
data_types = ["CPI_dataset/"]

b_size, n_hidden = 64, 128
lr, wd = 1e-4, 1e-4
num_epoches = 200
# 使用GPU加速
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")

# 输入id并映射到整数
Drug_id, Protein_id = data_loader_CPI.Get_CPI_id()
n_drugs, n_proteins = len(Drug_id), len(Protein_id)
dr_id_map, p_id_map = funcs.id_map(Drug_id), funcs.id_map(Protein_id)
print("number of Drug: ", n_drugs)
print("number of Protein ", n_proteins)

Dr_finger = data_loader_CPI.Get_CPI_finger()
P_seq = data_loader_CPI.Get_CPI_seq()

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

for data_type in data_types:
    for type in types:
        output_score = np.zeros(shape=(4, 5))
        for i in range(5):
            seed_type = "seed" + str(i + 1)
            train_P = np.loadtxt(data_type + seed_type + "/" + type + "/train_P.csv", dtype=str, delimiter=",",skiprows=1)
            dev_P = np.loadtxt(data_type + seed_type + "/" + type + "/dev_P.csv", dtype=str, delimiter=",", skiprows=1)
            test_P = np.loadtxt(data_type + seed_type + "/" + type + "/test_P.csv", dtype=str, delimiter=",", skiprows=1)
            train_N = np.loadtxt(data_type + seed_type + "/" + type + "/train_N.csv", dtype=str, delimiter=",", skiprows=1)
            dev_N = np.loadtxt(data_type + seed_type + "/" + type + "/dev_N.csv", dtype=str, delimiter=",", skiprows=1)
            test_N = np.loadtxt(data_type + seed_type + "/" + type + "/test_N.csv", dtype=str, delimiter=",", skiprows=1)
            print("number of DTI: ", len(train_P), len(dev_P), len(test_P))
            print("number of Negative DTI ", len(train_N), len(dev_N), len(test_N))
            train_X, train_Y = funcs.Get_sample(train_P, train_N, dr_id_map, p_id_map)
            dev_X, dev_Y = funcs.Get_sample(dev_P, dev_N, dr_id_map, p_id_map)
            test_X, test_Y = funcs.Get_sample(test_P, test_N, dr_id_map, p_id_map)
            train_loader = funcs.get_train_loader(train_X, train_Y, b_size)
            dev_loader = funcs.get_test_loader(dev_X, dev_Y, b_size)
            test_loader = funcs.get_test_loader(test_X, test_Y, b_size)
            best_auc, best_epoch, best_extra = 0, 0, 0
            best_test = []
            losses = nn.BCELoss()
            model = DNNNet().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.2, patience=20, verbose=False)
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
                    # 将预测的所有标签值放入train_scores，将原本的所有标签值放入train_labels
                train_scores_label = funcs.computer_label(train_scores, 0.5)
                train_avloss = train_loss / len(train_loader)
                train_acc = skm.accuracy_score(train_labels, train_scores_label)
                train_auc = skm.roc_auc_score(train_labels, train_scores)

                # 验证集进行验证，根据验证集AUC进行学习率衰减
                dev_scores, dev_labels = [], []
                test_scores, test_scores_label, test_labels = [], [], []
                extra_scores, extra_scores_label, extra_labels = [], [], []
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
                    # 测试集
                    for step, (batch_x, batch_y) in enumerate(test_loader):
                        model.eval()
                        b_x = batch_x.long().to(device)
                        b_y = torch.squeeze(batch_y.float().to(device), dim=1)
                        b_x_dr = torch.reshape(b_x[:, 0], (len(b_x), 1))
                        b_x_p = torch.reshape(b_x[:, 1], (len(b_x), 1))
                        output = model(Dr_finger, P_seq, b_x_dr, b_x_p)
                        score = torch.squeeze(output, dim=1)
                        scores, label = score.cpu().detach().numpy(), b_y.cpu().detach().numpy()
                        test_scores = np.concatenate((test_scores, scores))
                        test_labels = np.concatenate((test_labels, label))
                    test_scores_label = funcs.computer_label(test_scores, 0.5)
                    test_acc = skm.accuracy_score(test_labels, test_scores_label)
                    test_auc = skm.roc_auc_score(test_labels, test_scores)
                    test_aupr = skm.average_precision_score(test_labels, test_scores)
                    # 额外测试集

                print(
                    'epoch:{},Train Loss: {:.4f},Train Acc: {:.4f},Train Auc: {:.4f},Dev Auc: {:.4f}, Test Acc: {:.4f},Test Auc: {:.4f},TestAUPR: {:.4f}'
                        .format(epoch, train_avloss, train_acc, train_auc, dev_auc, test_acc, test_auc, test_aupr))
                if dev_auc >= best_auc:
                    best_auc = dev_auc
                    best_epoch = epoch
                    best_test = [format(test_acc, '.4f'), format(test_auc, '.4f'), format(test_aupr, '.4f')]
            print("best_dev_AUC:", best_auc)
            print("best_epoch", best_epoch)
            print("test_out", best_test)
            output_score[0][i], output_score[1][i], output_score[2][i], output_score[3][i] = best_auc, best_test[0], \
                                                                                             best_test[1], best_test[2]

        print(output_score)
        mean_acc, mean_auc, mean_mcc = np.nanmean(output_score[1]), np.nanmean(output_score[2]), np.nanmean(
            output_score[3])
        std_acc, std_auc, std_mcc = np.nanstd(output_score[1]), np.nanstd(output_score[2]), np.nanstd(output_score[3])
        print(mean_acc, mean_auc, mean_mcc)
        print(std_acc, std_auc, std_mcc)

