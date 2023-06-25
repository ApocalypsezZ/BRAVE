# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from core.utils import accuracy
from .metric_model import MetricModel


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ProtoLayer(nn.Module):
    def __init__(self):
        super(ProtoLayer, self).__init__()
        # res-12: 640, conv-64: 1600
        # self.transformerencoderlayer = nn.TransformerEncoderLayer(d_model=640, nhead=8, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.transformerencoderlayer, num_layers=2)

    def forward(
        self,
        query_feat,
        support_feat,
        way_num,
        shot_num,
        query_num,
        mode="euclidean",
    ):
        t, wq, c = query_feat.size()
        _, ws, _ = support_feat.size()

        # t, wq, c  -> t, wq, c
        query_feat = query_feat.reshape(t, way_num * query_num, c)

        # t, w, c -> t, w, shot_num, c
        support_feat = support_feat.reshape(t, way_num, shot_num, c)

        # use transformer encoder
        # all_tokens = torch.cat([query_feat, support_feat.reshape(t, way_num * shot_num, c)], dim=1)
        # all_tokens = self.transformer_encoder(all_tokens)
        # query_feat = all_tokens[:, :way_num * query_num, :] # t, wq, c
        # support_feat = all_tokens[:, way_num * query_num:, :].reshape(t, way_num, shot_num, c)  # t, w, shot_num, c

        # get proto
        proto_feat = torch.mean(support_feat, dim=2)    # t, w, c

        # get distance
        return {
            # t, wq, 1, c - t, 1, w, c -> t, wq, w
            "euclidean": lambda x, y: -torch.sum(torch.pow(x.unsqueeze(2) - y.unsqueeze(1), 2),dim=3,),
            # t, wq, c - t, c, w -> t, wq, w
            "cos_sim": lambda x, y: torch.matmul(F.normalize(x, p=2, dim=-1),torch.transpose(F.normalize(y, p=2, dim=-1), -1, -2)),
            # t, wq, c - t, c, w -> t, wq, w
            "dot": lambda x, y: torch.matmul(x, torch.transpose(y, -1, -2)),
        }[mode](query_feat, proto_feat)


class ProtoNet(MetricModel):
    def __init__(self, **kwargs):
        super(ProtoNet, self).__init__(**kwargs)
        self.proto_layer = ProtoLayer()
        self.loss_func = nn.CrossEntropyLoss()
        # num_embeddings: codebook size, embedding_dim: codebook dim
        self._vq_vae = VectorQuantizerEMA(num_embeddings=8192, embedding_dim=640, commitment_cost=0.25, decay=0.99)
        # self._vq_vae_Transpose = VectorQuantizerEMA(num_embeddings=8192, embedding_dim=25, commitment_cost=0.25, decay=0.99)
        self.avgpool = nn.AvgPool2d(5, stride=1)


    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)

        # vq-vae
        vq_loss, feat, _, _ = self._vq_vae(feat)

        # vq-vae.Transpose
        # feat = feat.permute(0, 2, 3, 1)
        # feat = feat.reshape(feat.size(0), -1, feat.size(-1))
        # feat = feat.reshape(feat.size(0), feat.size(1), 32, 20)
        # vq_loss, feat, _, _ = self._vq_vae_Transpose(feat)
        # feat = feat.reshape(feat.size(0), feat.size(1), -1)
        # feat = feat.permute(0, 2, 1)
        # feat = feat.reshape(feat.size(0), feat.size(1), 5, 5)

        # recon loss
        # x_recon = self._decoder(feat)
        # recon_loss = F.mse_loss(x_recon, images)

        # pooling
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)


        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )

        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        images, global_targets = batch
        images = images.to(self.device)
        episode_size = images.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        emb = self.emb_func(images) # t, c, h, w

        # vq-vae
        vq_loss, emb, _, _ = self._vq_vae(emb)

        # vq-vae.Transpose
        # emb = emb.permute(0, 2, 3, 1)
        # emb = emb.reshape(emb.size(0), -1, emb.size(-1))
        # emb = emb.reshape(emb.size(0), emb.size(1), 32, 20)
        # vq_loss, emb, _, _ = self._vq_vae_Transpose(emb)
        # emb = emb.reshape(emb.size(0), emb.size(1), -1)
        # emb = emb.permute(0, 2, 1)
        # emb = emb.reshape(emb.size(0), emb.size(1), 5, 5)

        # recon loss
        # x_recon = self._decoder(emb)
        # recon_loss = F.mse_loss(x_recon, images)

        # pooling
        emb = self.avgpool(emb) # t, c, 1, 1
        emb = emb.view(emb.size(0), -1) # t, c

        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            emb, mode=1
        )

        # reshape: t, wq, w -> twq, w
        output = self.proto_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        ).reshape(episode_size * self.way_num * self.query_num, self.way_num)

        # query_target.shape: b, wq
        # query_target.reshape(-1).shape: bwq
        loss = self.loss_func(output, query_target.reshape(-1)) + vq_loss/50
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
