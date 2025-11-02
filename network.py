from torch import nn
from torch.nn.functional import normalize
import torch

class MultiViewAE(nn.Module):
    def __init__(self, views, input_size, low_feature_dim):
        super(MultiViewAE, self).__init__()

        self.view_encoders = nn.ModuleList()
        self.view_decoders = nn.ModuleList()

        for view in range(views):
            encoder = nn.Sequential(
                nn.Linear(input_size[view], 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, 2000),
                nn.ReLU(),
                nn.Linear(2000, low_feature_dim),
            )
            self.view_encoders.append(encoder)


            decoder = nn.Sequential(
                nn.Linear(low_feature_dim, 2000),
                nn.ReLU(),
                nn.Linear(2000, 500),
                nn.ReLU(),
                nn.Linear(500, 500),
                nn.ReLU(),
                nn.Linear(500, input_size[view])
            )
            self.view_decoders.append(decoder)

    def forward(self, inputs):
        zs = []
        xrs = []

        for view in range(len(inputs)):

            z = self.view_encoders[view](inputs[view])
            zs.append(z)

            x = self.view_decoders[view](z)
            xrs.append(x)

        return zs, xrs

class ICMVC(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, class_num,temperature_c, device):
        super(ICMVC, self).__init__()

        self.common_size = common_size = low_feature_dim * view

        self.ae = MultiViewAE(view, input_size, low_feature_dim)

        # feature
        self.Specific_view_feature = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.Common_view_feature = nn.Sequential(
            nn.Linear(common_size, high_feature_dim),
        )

        # cluster
        self.Specific_view_cluster = nn.Sequential(
            nn.Linear(low_feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.Common_view_cluster = nn.Sequential(
            nn.Linear(common_size, class_num),
            nn.Softmax(dim=1)
        )

        self.view = view
        self.similarity = nn.CosineSimilarity(dim=2)
        self.temperature_c = temperature_c

    def pretrain(self, xs):

        zs, xrs = self.ae(xs)

        return zs, xrs

    def forward(self, xs):
        hs = []
        qs = []

        zs, xrs = self.ae(xs)

        for v in range(self.view):
            h = normalize(self.Specific_view_feature(zs[v]), dim=1)
            q = self.Specific_view_cluster(zs[v])
            hs.append(h)
            qs.append(q)

        return zs, xrs, hs, qs

    def ZDL(self, xs):
        zs, _ = self.ae(xs)
        commonz = torch.cat(zs, dim=1)

        commonh = normalize(self.Common_view_feature(commonz), dim=1)

        commonq = self.Common_view_cluster(commonz)

        # commonG = torch.exp(self.similarity(commonh.unsqueeze(1), commonh.unsqueeze(0)))
        commonG = self.similarity(commonh.unsqueeze(1), commonh.unsqueeze(0))
        commonG = nn.ReLU()(commonG)

        return commonz, commonh, commonq, commonG

    def forward_test(self, xs):
        zs, _ = self.ae(xs)
        commonz = torch.cat(zs, dim=1)

        commonh = normalize(self.Common_view_feature(commonz), dim=1)
        commonq = self.Common_view_cluster(commonz)
        pred = torch.argmax(commonq, dim=1)

        return commonz, commonh, commonq, pred

