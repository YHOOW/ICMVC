import torch
import torch.nn as nn
import math

class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to('cuda')
        S_2 = all_one - S_1

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f

        sim1 = torch.multiply(sim, S_2)

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class BaseLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_c, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_c = temperature_c
        self.device = device

    def forward_feature(self, h_i, h_j):

        N =self.batch_size

        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature_f

        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)

        nominator = torch.exp(positives)
        denominator = (mask.bool()) * torch.exp(similarity_matrix)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)  # 计算 q_i 的全局分布
        p_i /= p_i.sum()  # 归一化
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()  # 计算信息熵

        p_j = q_j.sum(0).view(-1)  # 计算 q_j 的全局分布
        p_j /= p_j.sum()  # 归一化
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()  # 计算信息熵

        entropy = ne_i + ne_j  # 两个视图的信息熵总和


        q_i = q_i.t()  # 转置，使其符合计算方式
        q_j = q_j.t()

        N = self.class_num  # 类别数的两倍

        similarity_matrix = torch.matmul(q_i, q_j.T) / self.temperature_c

        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N)).to(self.device)  # 建立一个全是1的掩码矩阵
        mask = mask.fill_diagonal_(0)  # 让对角线元素为0

        nominator = torch.exp(positives)
        denominator = (mask.bool()) * torch.exp(similarity_matrix)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N

        return loss + entropy

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_c, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_c = temperature_c
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()

        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()

        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()

        N = 2 * self.class_num

        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_c

        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + entropy
