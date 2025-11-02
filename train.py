import torch
from network import ICMVC
from metric import valid, effect_valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss,ContrastiveLoss
from dataloader import load_data
from time import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# Synthetic3d
# Prokaryotic
# CCV
# MNIST-USPS
# Hdigit
# YouTubeFace
# Cifar10
# Cifar100
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V

graph_para = 0.0001
Dataname = 'BDGP'# Hdigit

# 保存预训练模型的路径
PATH = './pretrain/'
pretrain_path = PATH + Dataname
# 检查目录是否存在，如果不存在，则创建它
if not os.path.exists(pretrain_path):
    os.makedirs(pretrain_path)

# 保存最终结果的路径
PATH = './record/'
record_path = PATH + Dataname
# 检查目录是否存在，如果不存在，则创建它
if not os.path.exists(record_path):
    os.makedirs(record_path)


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)

parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_c", default=1)

parser.add_argument("--learning_rate", default=0.0003)#0.0003
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=2)
parser.add_argument("--fine_tune_epochs", default=100)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)

# 保存预训练模型的路径
parser.add_argument('--pretrain_path', default=pretrain_path, type=str, help="Pretrained weights of the autoencoder")
# 保存最终结果的路径
parser.add_argument('--record_path', default=record_path, type=str, help="The final result of model")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "BDGP":
    args.fine_tune_epochs = 100
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

# 加载dataset
dataset, dims, view, data_size, class_num = load_data(args.dataset)

# 加载dataloader
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def compute_Laplace(S):
    S = 0.5 * (S.permute(1, 0) + S)
    D = torch.diag(torch.sum(S, 1))
    L = D - S
    return L

def graph_loss(Q, L):
    return torch.trace(torch.matmul(torch.matmul(Q.permute(1, 0), L), Q))



def pre_train(pretrain_epochs):
    mse = torch.nn.MSELoss()
    for pretrain_epoch in range(pretrain_epochs):
        tot_loss = 0.
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            optimizer.zero_grad()
            _, xrs = model.pretrain(xs)# 预训练期间没有使用注意力机制

            loss_list = []
            for v in range(view):
                loss_list.append(mse(xs[v], xrs[v]))# 重构损失
            loss = sum(loss_list)

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('pretrainEpoch {}'.format(pretrain_epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
    print("save pretrain model")
    torch.save(model.ae.state_dict(), args.pretrain_path + "/pretrained_model.pth")


def fine_tune(args, fine_tune_epochs, w1, w2, w3):

    best_acc_f = 0
    best_nmi_f = 0
    best_pur_f = 0
    best_ari_f = 0
    best_acc_c = 0
    best_nmi_c = 0
    best_pur_c = 0
    best_ari_c = 0

    mes = torch.nn.MSELoss()
    loss_curve = []
    for epoch in range(fine_tune_epochs):

        tot_loss = 0.
        tot_cl_f_loss = 0.
        tot_cl_c_loss = 0.
        tot_re_loss = 0.
        tot_co_loss = 0.

        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            optimizer.zero_grad()
            _, xrs, hs, qs= model(xs)
            commonz, commonh, commonq, commonG  = model.ZDL(xs)# 注意力机制可以不用预训练
            commonL = compute_Laplace(commonG)

            # 每个样本的 L2 范数
            # norm_commonh = torch.norm(commonh, dim=1)  # shape: [batch_size]
            # norm_hsv = torch.norm(hs[v], dim=1)  # shape: [batch_size]
            #
            # # 作为余弦相似度分母的乘积（逐行相乘）
            # denominator = norm_commonh * norm_hsv  # shape: [batch_size]

            loss_list = []
            loss_cl_f_list = []
            loss_cl_c_list = []
            loss_re_list = []
            loss_co_list = []

            for v in range(view):
                loss_list.append(criterion.forward_feature(hs[v], commonh))# 特征对比损失
                loss_cl_f_list.append(criterion.forward_feature(hs[v], commonh))

                loss_list.append(criterion.forward_label(qs[v], commonq))# 类簇分配对比损失
                loss_cl_c_list.append(criterion.forward_label(qs[v], commonq))

                loss_list.append(mes(xs[v], xrs[v]))# 重构损失
                loss_re_list.append(mes(xs[v], xrs[v]))

                loss_list.append(graph_para * graph_loss(qs[v],commonL))
                loss_co_list.append(graph_para * graph_loss(qs[v],commonL))


            # loss = sum(loss_list)
            loss_cl_f = sum(loss_cl_f_list)
            loss_cl_c = sum(loss_cl_c_list)
            loss_re = sum(loss_re_list)
            loss_co = sum(loss_co_list)
            # 总损失乘权重组合：
            if w3==0.01 :
                loss = w1 * loss_cl_f + w2 * loss_cl_c + loss_re + w3 * loss_co - 1.5
            elif w3==0.1 :
                loss = w1 * loss_cl_f + w2 * loss_cl_c + loss_re + w3 * loss_co - 0.75
            else:
                loss = w1 * loss_cl_f + w2 * loss_cl_c + loss_re + w3 * loss_co
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_cl_f_loss += loss_cl_f.item()
            tot_cl_c_loss += loss_cl_c.item()
            tot_re_loss += loss_re.item()
            tot_co_loss += loss_co.item()

        tot_loss_record = tot_loss/len(data_loader)
        tot_cl_f_loss_record = tot_cl_f_loss/len(data_loader)
        tot_cl_c_loss_record = tot_cl_c_loss/len(data_loader)
        tot_re_loss_record = tot_re_loss/len(data_loader)
        tot_co_loss_record = tot_co_loss/len(data_loader)

        # 添加当前轮的总损失到曲线列表
        loss_curve.append(tot_loss_record)



        print('Epoch {}'.format(epoch), 'Total_Loss:{:.6f}'.format(tot_loss_record),
              'cl_f_Loss:{:.6f}'.format(tot_cl_f_loss_record), 'cl_c_Loss:{:.6f}'.format(tot_cl_c_loss_record),
              're_Loss:{:.6f}'.format(tot_re_loss_record),'co_Loss:{:.6f}'.format(tot_co_loss_record))

        # 记录txt
        with open(os.path.join(args.record_path, 'fine_tune_loss_log.txt'), 'a') as file:
            file.write("Epoch {}/{}\n".format(epoch, fine_tune_epochs))
            file.write("Total_loss is: {:.8f} \t cl_f_loss is: {:.8f} \t cl_c_loss is: {:.8f} \t re_loss is: {:.8f} \t co_loss is: {:.8f} \n".format(tot_loss_record,tot_cl_f_loss_record ,tot_cl_c_loss_record, tot_re_loss_record, tot_co_loss_record))

        if epoch % 1 == 0 and epoch != 0:
            nmi_f, ari_f, acc_f, pur_f, nmi_c, ari_c, acc_c, pur_c = effect_valid(model, device, dataset, view, data_size, class_num)

            if best_acc_f < acc_f:
                best_acc_f = acc_f
            if best_nmi_f < nmi_f:
                best_nmi_f = nmi_f
            if best_pur_f < pur_f:
                best_pur_f = pur_f
            if best_ari_f < ari_f:
                best_ari_f = ari_f

            if best_acc_c < acc_c:
                best_acc_c = acc_c
            if best_nmi_c < nmi_c:
                best_nmi_c = nmi_c
            if best_pur_c < pur_c:
                best_pur_c = pur_c
            if best_ari_c < ari_c:
                best_ari_c = ari_c

            print('Feature=====ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc_f, nmi_f, pur_f, ari_f))
            print('Feature=====best_ACC = {:.4f} best_NMI = {:.4f} best_PUR={:.4f} best_ARI = {:.4f}'.format(best_acc_f, best_nmi_f, best_pur_f, best_ari_f))

            print('Cluster=====ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc_c, nmi_c, pur_c, ari_c))
            print('Cluster=====best_ACC = {:.4f} best_NMI = {:.4f} best_PUR={:.4f} best_ARI = {:.4f}'.format(best_acc_c, best_nmi_c, best_pur_c, best_ari_c))

            with open(os.path.join(args.record_path, 'fine_tune_loss_log.txt'), 'a') as file:
                file.write("Feature:\n")
                file.write("Clustering results: ACC={:.4f} NMI={:.4f} PUR={:.4f} ARI = {:.4f}\n".format(acc_f, nmi_f, pur_f, ari_f))
                file.write("Best Clustering results: best_ACC={:.4f} best_NMI={:.4f} best_PUR={:.4f} best_ARI = {:.4f}\n\n".format(best_acc_f, best_nmi_f, best_pur_f, best_ari_f))
                file.write("Cluster:\n")
                file.write("Clustering results: ACC={:.4f} NMI={:.4f} PUR={:.4f} ARI = {:.4f}\n".format(acc_c, nmi_c, pur_c, ari_c))
                file.write("Best Clustering results: best_ACC={:.4f} best_NMI={:.4f} best_PUR={:.4f} best_ARI = {:.4f}\n\n".format(best_acc_c, best_nmi_c, best_pur_c, best_ari_c))
    return loss_curve

weights1 = [1.0]
weights2 = [1.0]
weights3 = [0.01, 0.1, 1.0]

all_results = {}  # 用于记录每组实验的损失曲线
loss_curves = []  # 用于后面绘图

for w1 in weights1:
    for w2 in weights2:
        for w3 in weights3:
            print(f"Training with weights: w1={w1}, w2={w2}, w3={w3}")

            "======================================================================训练流程开始======================================================================"
            if not os.path.exists('./results'):
                os.makedirs('./results')

            model = ICMVC(view, dims, args.low_feature_dim, args.high_feature_dim, class_num, args.temperature_c, device)
            print(model)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            criterion = ContrastiveLoss(args.batch_size, class_num, args.temperature_f, args.temperature_c, device).to(device)

            "=----------------------------------------------------==预训练==-------------------------------------------------------------="
            print("start pretrain")
            t0 = time()
            pretrained_model_path = args.pretrain_path + "/pretrained_model.pth"
            if os.path.exists(pretrained_model_path):
                model.ae.load_state_dict(torch.load(pretrained_model_path))
                print("Pretrained parameters loaded successfully.")
            else:
                pre_train(args.rec_epochs)
            t1 = time()
            print("Time for pretraining: %ds" % (t1 - t0))
            "=----------------------------------------------------==预训练==-------------------------------------------------------------="

            "===------------------------------------------------------正式训练---------------------------------------------------------------==="
            print("start formaltrain")
            loss_curve = fine_tune(args, args.fine_tune_epochs, w1, w2, w3)
            loss_curves.append((w1, w2, w3, loss_curve))  # ⬅️ 记录当前组合的损失曲线
            "===------------------------------------------------------正式训练---------------------------------------------------------------==="

            "======================================================================训练流程结束======================================================================"
            print('---------train over---------')
            valid(args, model, device, dataset, view, data_size, class_num)
            state = model.state_dict()
            torch.save(state, './results/' + args.dataset + f'_w1_{w1}_w2_{w2}_w3_{w3}.pth')
            print('Saving model...')

import matplotlib.pyplot as plt

# ========================= 🔻 所有训练结束后统一画图 ==========================
plt.figure(figsize=(12, 8), dpi=300)  # 高分辨率图像

# 绘制每条曲线
for w1, w2, w3, curve in loss_curves:
    plt.plot(range(len(curve)), curve, label=f'α={w1}, β={w2}, λ={w3}', linewidth=2)

# 设置加粗字体
plt.xlabel("Epoch", fontsize=26, fontweight='bold')
plt.ylabel("Total Loss", fontsize=26, fontweight='bold')
plt.title("Loss Curves for Different Weight Combinations", fontsize=26, fontweight='bold')

# 设置刻度加粗字体
plt.xticks(fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# 图例设置（加粗字体和边框）
plt.legend(fontsize=22, frameon=True, edgecolor='black', framealpha=0.9)

# 添加网格，设置线宽
plt.grid(True, linestyle='--', linewidth=0.7)

# 自动布局
plt.tight_layout()

# 保存图像
plt.savefig('./results/loss_curves_all.png', dpi=300)
plt.close()

