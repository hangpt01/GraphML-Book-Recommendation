import torch
import torch.nn as nn
import numpy as np
import time
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss
from util.evaluation import early_stopping

class GCCF(GraphRecommender):
    def __init__(self, conf, training_set, test_set, knowledge_set, **kwargs):
        super(GCCF, self).__init__(conf, training_set, test_set, knowledge_set, **kwargs)
        args = OptionConf(self.config['GCCF'])
        self.kwargs = kwargs
        self.n_layers = int(args['-n_layer'])
        self.model = GCCF_Encoder(self.data, self.emb_size, self.n_layers)
        self.lr_decay  = float(kwargs['lr_decay'])
        self.early_stopping_steps = int(kwargs['early_stopping_steps'])
        self.reg = float(kwargs['reg'])
        self.maxEpoch = int(kwargs['max_epoch'])
        self.lRate = float(kwargs['lrate'])
        self.wdecay = float(kwargs['weight_decay'])

    def train(self, load_pretrained):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate, weight_decay=self.wdecay)

        lst_train_losses = []
        recall_list = []

        for epoch in range(self.maxEpoch):
            train_losses = []
            s_train = time.time()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                # Calculate loss
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                reg_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb)
                batch_loss = rec_loss + reg_loss
                train_losses.append(batch_loss.item())

                # Backward propagation and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0:
                    print(f'Epoch {epoch + 1}, Batch {n}, Rec Loss: {rec_loss.item()}, Reg Loss: {reg_loss.item()}')

            # End of epoch
            e_train = time.time()
            print(f'Epoch {epoch + 1} completed in {e_train - s_train:.2f}s')

            with torch.no_grad():
                self.user_emb, self.item_emb = model()
                cur_recall = self.evaluate_model()
                recall_list.append(cur_recall)
                best_recall, should_stop = early_stopping(recall_list, self.early_stopping_steps)
                if should_stop:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            lst_train_losses.append(np.mean(train_losses))

        self.save()

    def evaluate_model(self):
        # Here you can implement the evaluation function based on recall or other metrics
        # For now, just returning a placeholder recall value
        return np.random.rand()  # Replace with actual recall calculation

    def save(self):
        print("Saving the model...")
        self.save_model(self.model)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class GCCF_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(GCCF_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.n_users, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.n_items, self.emb_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            ego_embeddings = torch.relu(ego_embeddings)  # Non-linear activation in GCCF
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.n_users, self.data.n_items])
        return user_all_embeddings, item_all_embeddings
