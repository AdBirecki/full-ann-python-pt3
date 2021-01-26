import torch
import torch.nn as nn
import torch.functional as F


class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()

        embLayers = [nn.Embedding(ni, nf) for ni, nf in emb_szs]
        self.embeds = nn.ModuleList(embLayers).cuda()

        self.emb_drop = nn.Dropout(p).cuda()
        self.bn_cont = nn.BatchNorm1d(n_cont).cuda()

        layerlist = []
        n_emb = sum([nf for ni, nf in emb_szs])
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i).cuda())
            layerlist.append(nn.ReLU(inplace=True).cuda())
            layerlist.append(nn.BatchNorm1d(i).cuda())
            layerlist.append(nn.Dropout(p).cuda())
            n_in = i

        layerlist.append(nn.Linear(layers[-1], out_sz).cuda(0))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embeds):
            embedding = e(x_cat[:, i]).cuda()
            embeddings.append(embedding)
        
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)

        return x

