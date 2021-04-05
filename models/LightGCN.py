import os
import math
import time
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel

class LightGCN(BaseModel):
    def __init__(self, model_conf, num_users, num_items, device):
        super(LightGCN, self).__init__()
        self.data_name = model_conf.data_name
        self.num_users = num_users
        self.num_items = num_items

        self.emb_dim = model_conf.emb_dim
        self.num_layers = model_conf.num_layers
        self.node_dropout = model_conf.node_dropout

        self.split = model_conf.split
        self.num_folds = model_conf.num_folds
        self.reg = model_conf.reg
        self.batch_size = model_conf.batch_size

        self.Graph = None
        self.data_loader = None

        self.path = model_conf.graph_dir
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.device = device

        self.build_graph()

    def build_graph(self):
        self.user_embedding = nn.Embedding(self.num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.emb_dim)

        # weight initalization
        nn.init.normal_(self.user_embedding.weight, 0, 0.01) # (Tensor, mean, std)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        self.user_embedding_pred = None
        self.item_embedding_pred = None

        self.to(self.device)

    def forward(self, user, pos, neg):
        u_embedding, i_embedding = self.lightgcn_embedding(self.Graph)

        user_latent = F.embedding(user, u_embedding)
        positive_latent = F.embedding(pos, i_embedding)

        positive_score = torch.mul(user_latent, positive_latent).sum(1)

        if neg is not None:
            negative_latent = F.embedding(neg, i_embedding)
            negative_score = torch.mul(user_latent, negative_latent).sum(1)
            return positive_score, negative_score
        else:
            return positive_score

    def train_one_epoch(self, dataset, optimizer, batch_size, verbose):
        train_matrix = dataset.train_matrix

        if self.Graph == None:
            self.Graph = self.getSparseGraph(train_matrix)

        if self.data_loader == None:
            self.data_loader = PairwiseGenerator(train_matrix,
                                                 num_negatives=1,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 device=self.device)

        loss = 0.0
        for idx, batch_data in enumerate(self.data_loader):
            optimizer.zero_grad()
            batch_user, batch_pos, batch_neg = batch_data

            pos_output, neg_output = self.forward(batch_user, batch_pos, batch_neg)
            userEmb = self.user_embedding(batch_user)
            posEmb = self.item_embedding(batch_pos)
            negEmb = self.item_embedding(batch_neg)

            batch_loss = torch.mean(F.softplus(neg_output - pos_output))
            reg_loss = (1 / 2) * (userEmb.norm(2).pow(2) + posEmb.norm(2).pow(2) + negEmb.norm(2).pow(2)) / float(len(batch_user))

            batch_loss = batch_loss + self.reg * reg_loss

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss

        return loss

    def predict(self, eval_users, eval_pos, test_batch_size):
        num_eval_users = len(eval_users)
        num_batches = int(np.ceil(num_eval_users / test_batch_size))
        pred_matrix = np.zeros(eval_pos.shape)
        perm = list(range(num_eval_users))
        with torch.no_grad():
            for index in range(num_batches):
                if (index + 1) * test_batch_size >= num_eval_users:
                    batch_idx = perm[index * test_batch_size:]
                else:
                    batch_idx = perm[index * test_batch_size: (index + 1) * test_batch_size]

                batch_users = eval_users[batch_idx]
                batch_users_torch = torch.LongTensor(batch_users).to(self.device)
                pred_matrix[batch_users] = self.predict_batch_users(batch_users_torch).detach().cpu().numpy()

        pred_matrix[eval_pos.nonzero()] = float('-inf')

        return pred_matrix





    def before_evaluate(self):
        self.user_embedding_pred, self.item_embedding_pred = self.lightgcn_embedding(self.Graph)

    # ---
    def lightgcn_embedding(self, graph):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)

        embs = [all_emb]

        if self.node_dropout > 0:
            if self.training:
                g_droped = self.__dropout(graph, self.node_dropout)
            else:
                g_droped = graph
        else:
            g_droped = graph

        ego_emb = all_emb
        for k in range(self.num_layers):
            if self.split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], ego_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        lightgcn_out = torch.mean(embs, dim=1)
        # print(lightgcn_out, lightgcn_out.shape, self.num_users, self.num_items)
        users, items = torch.split(lightgcn_out, [self.num_users, self.num_items]) # num_users ~ num_users + num_items

        return users, items

    def make_train_matrix(self):
        train_matrix_arr = self.dataset.train_matrix.toarray()
        self.train_matrix = sp.csr_matrix(train_matrix_arr)

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.num_users + self.num_items) // self.num_folds
        for i_fold in range(self.num_folds):
            start = i_fold * fold_len
            if i_fold == self.num_folds - 1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self, rating_matrix):
        n_users, n_items = rating_matrix.shape
        print("loading adjacency matrix")

        filename = f'{self.data_name}_s_pre_adj_mat.npz'
        try:
            pre_adj_mat = sp.load_npz(os.path.join(self.path, filename))
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except:
            print("generating adjacency matrix")
            s = time.time()
            adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = rating_matrix.tolil()
            adj_mat[:n_users, n_users:] = R
            adj_mat[n_users:, :n_users] = R.T
            adj_mat = adj_mat.todok()

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time.time()
            # print(f"costing {end - s}s, saved norm_mat...")
            sp.save_npz(os.path.join(self.path, filename), norm_adj)

        if self.split == True:
            Graph = self._split_A_hat(norm_adj)
            print("Do split matrix")
        else:
            Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            Graph = Graph.coalesce().to(self.device)
            print("Do not split the matrix")

        return Graph

    def predict_batch_users(self, user_ids):
        user_embeddings = F.embedding(user_ids, self.user_embedding_pred)
        item_embeddings = self.item_embedding_pred
        return np.matmul(user_embeddings, item_embeddings.T)

# positive, negative sampling class
class PairwiseGenerator:
    def __init__(self, input_matrix, num_negatives=1, batch_size=64, shuffle=True, device=None):
        super().__init__()
        self.input_matrix = input_matrix
        self.num_negatives = num_negatives
        self.num_users, self.num_items = input_matrix.shape

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self._construct()

    def _construct(self):
        self.pos_dict = {}
        for u in range(self.num_users):
            u_items = self.input_matrix[u].indices

            self.pos_dict[u] = u_items.tolist()

    def __len__(self):
        return int(np.ceil(self.num_users / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(self.num_users)
        else:
            perm = np.arange(self.num_users)

        for b, st in enumerate(range(0, len(perm), self.batch_size)):
            batch_pos = []
            batch_neg = []

            ed = min(st + self.batch_size, len(perm))
            batch_users = perm[st:ed]
            for i, u in enumerate(batch_users):

                posForUser = self.pos_dict[u]
                if len(posForUser) == 0:
                    continue
                posindex = np.random.randint(0, len(posForUser))
                positem = posForUser[posindex]
                while True:
                    negitem = np.random.randint(0, self.num_items)
                    if negitem in posForUser:
                        continue
                    else:
                        break
                batch_pos.append(positem)
                batch_neg.append(negitem)

            batch_users = torch.tensor(batch_users, dtype=torch.long, device=self.device)
            batch_pos = torch.tensor(batch_pos, dtype=torch.long, device=self.device)
            batch_neg = torch.tensor(batch_neg, dtype=torch.long, device=self.device)
            yield batch_users, batch_pos, batch_neg