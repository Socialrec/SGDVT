import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE_inter,  InfoNCE_intra, InfoNCE_inter_intra
from data.social import Relation
from data.augmentor import GraphAugmentor
import torch.nn.functional as F
torch.set_printoptions(profile="full")

class SGDVT(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(SGDVT, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SGDVT'])
        self.n_layers = int(args['-n_layer'])
        downthre = int(args['-downthre'])
        upthre  = int(args['-upthre'])
        self.c1_rate = float(args['-lambda1'])
        self.c2_rate = float(args['-lambda2'])
        aug_type = int(args['-augtype'])
        drop_rate = float(args['-droprate'])
        tau = float(args['-tau'])
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.model = SGDVT_Encoder(self.data, self.emb_size, self.n_layers, self.social_data, drop_rate, tau, aug_type, downthre , upthre)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            droped_social1 = model.graph_reconstruction()
            droped_social2 = model.graph_reconstruction()
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb ,emb_list  = model()

                can_emb = emb_list[2]
                context_emb = emb_list[3]
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
               
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                c1_loss = self.c1_rate * model.cal_c1_loss([user_idx,pos_idx], droped_social1 , droped_social2)
                c2_loss = self.c2_rate * model.cal_c2_loss(context_emb,can_emb,user_idx,pos_idx)

                #batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb)  #可复现
                #batch_loss =  rec_loss + c1_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) 
                #batch_loss =  rec_loss + c2_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb)  
                batch_loss =  rec_loss + c1_loss + c2_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) 
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0 and n>0:
                    #print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'c1_loss', c1_loss.item(),'c2_loss', c2_loss.item())
                    # print('mean_sim:', model.mean_sim , 'pruning:', model.pruning, 'filter_num:',  model.filter_num )
                    # print('min_value:', model.min_value , 'min_row_index', model.min_row_index, 'min_col_index:',  model.min_col_index)
                    # print('max_value:', model.max_value , 'max_row_index', model.max_row_index, 'max_col_index:',  model.max_col_index)
                    # print('value-241:', model.value_241 )
                    
            with torch.no_grad():
                self.user_emb, self.item_emb,_  = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb,_ = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

class SGDVT_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers , social_data, drop_rate ,tau , aug_type, downthre , upthre):
        super(SGDVT_Encoder, self).__init__()
        self.data = data
        self.social_data = social_data
        self.latent_size = emb_size
        self.layers = n_layers
        self.drop_rate = drop_rate
        self.tau = tau
        self.aug_type = aug_type
        self.norm_adj = data.norm_adj
        self.social_adj_O = self.social_data.get_social_mat()
        print("<<<<<<<<<<<<<<<<social original number>>>>>>>>>>>>>>>>>>>")
        print(self.social_adj_O.nnz)

        self.social_adj_A = data.interaction_mat.dot(data.interaction_mat.T)
        self.social_adj_A.setdiag(0)

        self.social_adj_A.data[self.social_adj_A.data< downthre]=0  
        self.social_adj_A.data[self.social_adj_A.data> upthre ]=0  
        self.social_adj_A.eliminate_zeros()

        print("<<<<<<<<<<<<<<<<social augmented number>>>>>>>>>>>>>>>>>>>")
        print(self.social_adj_A.nnz)

        self.social_OA = self.social_adj_O + self.social_adj_A
        self.social_OA.data[self.social_OA.data > 1] = 1
        print("<<<<<<<<<<<<<<<<social original&augmented number>>>>>>>>>>>>>>>>>>>")
        print(self.social_OA.nnz)


        #self.case_study(self.social_OA, self.data.interaction_mat,241,1181)

        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.socialGraph = TorchGraphInterface.convert_sparse_mat_to_tensor(self.social_OA).cuda()
        self.Graph_Comb_u = Graph_Comb_user_EAGCN(self.latent_size)
        self.mean_sim = 0
        self.pruning = 0
        self.filter_num = 0
        # self.min_value = 0
        # self.min_row_index = 0
        # self.min_col_index = 0
        # self.value_241 = 0
        # self.max_value = 0
        # self.max_row_index = 0
        # self.max_col_index = 0
    

    def get_column_indices_csr(self,sparse_matrix, row_index):
        start = sparse_matrix.indptr[row_index]
        end = sparse_matrix.indptr[row_index + 1]
        return sparse_matrix.indices[start:end]

    def case_study(self,social,interaction,u1,u2):
        print("*************************************************")
        u1_list = self.get_column_indices_csr(social,u1)
        print(u1_list)
        u2_list = self.get_column_indices_csr(social,u2)
        print(u2_list)
        valid_num = 0
        for i in u1_list:
            for j in u2_list:
                if(interaction[i,j]!=0):
                    valid_num = valid_num + 1
        print("u1_list_len:",len(u1_list),"u2_list_len",len(u2_list))            
        print("valid_num:",valid_num,"valid_ratio",valid_num/(len(u1_list)*len(u2_list)))
        print("*************************************************")

    def sp_cos_sim(self, a, b, social_adj ,eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n)) 
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        social_indice = social_adj.coalesce().indices()
        social_shape  = social_adj.coalesce().shape

        L = social_indice.shape[1]  # self.social_indice.shape: torch.Size([2, 119728])
        sims = torch.zeros(L, dtype=a.dtype).cuda()

        a_batch = torch.index_select(a_norm, 0, social_indice[0, :])
        b_batch = torch.index_select(b_norm, 0, social_indice[1, :])
        dot_prods = torch.mul(a_batch, b_batch).sum(1)
        sims[:] = dot_prods

        return torch.sparse.FloatTensor(social_indice, sims, size=social_shape).coalesce()

    def get_sim_adj(self , u_emb , social_adj):

        social_shape  = social_adj.coalesce().shape
        u_emb = torch.sparse.mm(social_adj, u_emb)
        sim_adj = self.sp_cos_sim(u_emb, u_emb , social_adj)

        # sparse_indices = sim_adj.coalesce()._indices()
        # sparse_values  = sim_adj.coalesce()._values()
        # min_value, min_index = torch.min(sparse_values, dim=0)
        # min_row_index, min_col_index = sparse_indices[:, min_index]
        # self.min_value =  min_value.item()
        # self.min_row_index = min_row_index.item()
        # self.min_col_index = min_col_index.item()

        # self.value_241 = sim_adj[241,1181]

        # max_value, max_index = torch.max(sparse_values, dim=0)
        # max_row_index, max_col_index = sparse_indices[:, max_index]
        # self.max_value =  max_value.item()
        # self.max_row_index = max_row_index.item()
        # self.max_col_index = max_col_index.item()

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)  # sim = ( sim + 1 ) /2
        mean_sim = torch.mean(sim_value)
        #print(mean_sim)

        #flickr best
        # pruning = 0.8
        # if ( mean_sim < 0.8):
        #      pruning = 0.75

        #pruning = 0.75

        #ciao best
        pruning = 0.8
    
        #yelp_cat best
        # pruning = 0
        # if ( mean_sim > 0.7 ):
        #      pruning = 0.75

        #pruning = 0.7

        self.mean_sim = mean_sim
        self.pruning = pruning
        self.filter_num = (sim_value < pruning).sum().item()

        # torch.where(condition，a，b) 输入参数condition：条件限制，如果满足条件，则选择a，否则选择b作为输出
        pruned_sim_value = torch.where(sim_value < pruning, torch.zeros_like(sim_value),sim_value) if pruning > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, social_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value, social_shape).cuda().coalesce()

        return normal_sim_adj

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def graph_reconstruction(self):
        if self.aug_type==0 or 1:
            dropped_adj = self.random_graph_augment()
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def random_graph_augment(self):
        dropped_mat = None
        if self.aug_type == 0:
            dropped_mat = GraphAugmentor.node_dropout(self.social_OA, self.drop_rate)
        elif self.aug_type == 1 or self.aug_type == 2:
            dropped_mat = GraphAugmentor.edge_dropout(self.social_OA, self.drop_rate)
        return TorchGraphInterface.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def forward(self,perturbed_adj=None):
        if (perturbed_adj is None) :
            S = self.get_sim_adj(self.embedding_dict['user_emb'] , self.socialGraph)
        else :
            S = self.get_sim_adj(self.embedding_dict['user_emb'] , perturbed_adj)        
        user_emb_social_set = []
        users_emb_1 = self.embedding_dict['user_emb']
        for layer in range(3):
            user_emb_temp_1= torch.sparse.mm(S, users_emb_1)
            user_emb_social_set.append(user_emb_temp_1)
            users_emb_1 = user_emb_temp_1
        user_sview_emb = torch.stack(user_emb_social_set, dim=1)
        user_sview_emb = torch.mean(user_sview_emb, dim=1)

        ego_embeddings_v1 = torch.cat([self.embedding_dict['user_emb'] + user_sview_emb, self.embedding_dict['item_emb']], 0)
        all_embeddings_v1 = [ego_embeddings_v1]
        for k in range(self.layers):
            ego_embeddings_v1 = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings_v1)
            random_noise = torch.rand_like(ego_embeddings_v1).cuda()
            ego_embeddings_v1 += torch.sign(ego_embeddings_v1) * F.normalize(random_noise, dim=-1) * 0.1
            all_embeddings_v1.append(ego_embeddings_v1)

        all_embeddings = torch.stack(all_embeddings_v1, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings_v1 = all_embeddings[:self.data.user_num]
        item_all_embeddings_v1 = all_embeddings[self.data.user_num:]

        user_all_embeddings = self.Graph_Comb_u(user_all_embeddings_v1 , user_sview_emb)
        item_all_embeddings = item_all_embeddings_v1

        return user_all_embeddings, item_all_embeddings, all_embeddings_v1

    def cal_c1_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_view_1, item_view_1,_ = self.forward(perturbed_mat1)
        user_view_2, item_view_2,_ = self.forward(perturbed_mat2)

        user_c1_loss = InfoNCE_inter(user_view_1[u_idx], user_view_2[u_idx], self.tau)  
        return user_c1_loss 

        # user_c1_loss = InfoNCE_inter(user_view_1[u_idx], user_view_2[u_idx], self.tau)  
        # item_c1_loss = InfoNCE_inter(item_view_1[i_idx], item_view_2[i_idx], self.tau)  
        # return (user_c1_loss+ item_c1_1oss)/2 

    def cal_c2_loss(self, neighbor_embedding, center_embedding, user, item):
        neighbor_user_embedding, neighbor_item_embedding = torch.split(neighbor_embedding, [self.data.user_num, self.data.item_num])
        center_user_embedding, center_item_embedding = torch.split(center_embedding, [self.data.user_num, self.data.item_num])

        # 用户侧 注意锚点
        cent_user_embedding = center_user_embedding[user]  # l1
        neigh_item_embedding = neighbor_item_embedding[item]

        neigh_item_embedding = F.normalize(neigh_item_embedding)
        cent_user_embedding = F.normalize(cent_user_embedding)
        center_user_embedding = F.normalize(center_user_embedding)

        pos_score_user = torch.mul(neigh_item_embedding, cent_user_embedding).sum(dim=1)
        ttl_score_user = torch.matmul(neigh_item_embedding, center_user_embedding.transpose(0, 1))

        pos_score_user = torch.exp(pos_score_user / self.tau)
        ttl_score_user = torch.exp(ttl_score_user / self.tau).sum(dim=1)

        # 项目作为锚点
        neigh_user_embedding = neighbor_user_embedding[user]  # l2
        cent_item_embedding = center_item_embedding[item]

        neigh_user_embedding = F.normalize(neigh_user_embedding)
        cent_item_embedding = F.normalize(cent_item_embedding)
        center_item_embedding = F.normalize(center_item_embedding)

        pos_score_item = torch.mul(neigh_user_embedding, cent_item_embedding).sum(dim=1)
        ttl_score_item = torch.matmul(neigh_user_embedding, center_item_embedding.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.tau)
        ttl_score_item = torch.exp(ttl_score_item / self.tau).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        return (ssl_loss_user + ssl_loss_item)

class Graph_Comb_user_EAGCN(nn.Module):
    def __init__(self, embed_dim):
        super(Graph_Comb_user_EAGCN, self).__init__()
        self.gate1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, y):
        gu = torch.sigmoid(self.gate1(x)+self.gate2(y))
        output = torch.mul(gu,y)+torch.mul((1-gu),x)
        return output



