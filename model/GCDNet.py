apath = r'..\@aspect\d5\Digital_Music_5_sentiment_train_aspect2id.txt'
user_ys_path = '../data/Digital_Music_5/user_ys.pickle'
item_ys_path = '../data/Digital_Music_5/item_ys.pickle'
sentiment_feat_path = r'..\BERT\sentiment.pkl'
dataset_name = 'Digital_Music_5'
dataset_name_path = '../data/Digital_Music_5/Digital_Music_5.json'
aspect_feat_path = r'..\BERT\aspect.pkl'

import argparse
import time

import dgl.function as fn
import numpy as np

from util import *
import torch
import torch.nn.functional as F
import dgl
from load_data import *
from util import *
import random
import ast
from tqdm import tqdm, trange
import json
from abc import ABC

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    dgl.random.seed(seed)
    torch.use_deterministic_algorithms(True)

seed_everything(1234)




class Data(object):

    def __init__(self, dataset_name, dataset_path, device, review_fea_size):
        self._device = device
        self._review_fea_size = review_fea_size

        sent_train_data, sent_valid_data, sent_test_data, _, _, dataset_info = load_sentiment_data(dataset_path)

        self.remove_users = []

        self._num_user = dataset_info['user_size']
        self._num_item = dataset_info['item_size']

        review_feat_path = f'../checkpoint/{dataset_name}/BERT-Whitening/bert-base-uncased_sentence_vectors_dim_{review_fea_size}.pkl'
        self.train_review_feat = torch.load(review_feat_path)

        # def shuffle_dict(dictionary):
        #     keys = list(dictionary.keys())
        #     values = list(dictionary.values())
        #     random.shuffle(keys)
        #     random.shuffle(values)
        #     shuffled_dict = {k: v for k, v in zip(keys, values)}
        #     return shuffled_dict
        #
        # self.train_review_feat=shuffle_dict(self.train_review_feat)
        self.review_feat_updated = {}  # 97282是单向
        # 将train_review_feat放在review_feat_updated中，review_feat_updated序号包含用户项目，双向的
        for key, value in self.train_review_feat.items():  # train_review_feat一个字典，键是元组(5273, 3707)，值存储这个评论的64特征
            self.review_feat_updated[(key[0], key[1] + self._num_user)] = value
            self.review_feat_updated[(key[1] + self._num_user, key[0])] = value

        def process_sent_data(info):
            user_id = info['user_id'].to_list()
            # 让物品的id从max user id开始，相当于将用户和物品节点视为一类节点；
            item_id = [int(i) + self._num_user for i in info['item_id'].to_list()]
            rating = info['rating'].to_list()

            return user_id, item_id, rating

        self.train_datas = process_sent_data(sent_train_data)
        self.valid_datas = process_sent_data(sent_valid_data)
        self.test_datas = process_sent_data(sent_test_data)
        self.possible_rating_values = np.unique(self.train_datas[2])

        # print(f'user number: {self._num_user}, item number: {self._num_item}')

        self.user_item_rating = {}

        def _generate_train_pair_value(data: tuple):
            user_id, item_id, rating = np.array(data[0], dtype=np.int64), np.array(data[1], dtype=np.int64), \
                                       np.array(data[2], dtype=np.int64)

            # for u in self.remove_users:
            #     idx = np.where(user_id != u)
            #     user_id, item_id, rating = user_id[idx], item_id[idx], rating[idx]

            rating_pairs = (user_id, item_id)
            rating_pairs_rev = (item_id, user_id)
            rating_pairs = np.concatenate([rating_pairs, rating_pairs_rev], axis=1)  ## 双向 ！！！！

            rating_values = np.concatenate([rating, rating],
                                           axis=0)

            for i in range(len(rating)):
                uid, iid = user_id[i], item_id[i]
                if uid not in self.user_item_rating:
                    self.user_item_rating[uid] = []
                self.user_item_rating[uid].append((iid, rating[i]))

            return rating_pairs, rating_values

        def _generate_test_pair_value(data: tuple):
            user_id, item_id, rating = data[0], data[1], data[2]

            rating_pairs = (np.array(user_id, dtype=np.int64),
                            np.array(item_id, dtype=np.int64))

            rating_values = np.array(rating, dtype=np.float32)

            return rating_pairs, rating_values

        print('Generating train/valid/test data.\n')
        self.train_rating_pairs, self.train_rating_values = _generate_train_pair_value(self.train_datas)  # 双向
        self.valid_rating_pairs, self.valid_rating_values = _generate_test_pair_value(self.valid_datas)
        self.test_rating_pairs, self.test_rating_values = _generate_test_pair_value(self.test_datas)

        # generate train_review_pairs
        self.train_review_pairs = []
        for idx in range(len(self.train_rating_values)):
            u, i = self.train_rating_pairs[0][idx], self.train_rating_pairs[1][idx]
            review = self.review_feat_updated[(u, i)].numpy()
            self.train_review_pairs.append(review)

        self.train_review_pairs = np.array(self.train_review_pairs)  # 双向

        print('Generating train graph.\n')
        self.train_enc_graph = self._generate_enc_graph(self.train_rating_pairs, self.train_rating_values)

    # init结束

    def _generate_enc_graph(self, rating_pairs, rating_values):  # 产生编码
        # user_item_r = np.zeros((self._num_user + self._num_item, self._num_item + self._num_user),
        #                        dtype=np.float32)  # 把评分放到矩阵中
        # for i in range(len(rating_values)):
        #     user_item_r[[rating_pairs[0][i], rating_pairs[1][i]]] = rating_values[i]  # @ss
        #     ##@ss if user_item_r[[rating_pairs[0][i], rating_pairs[1][i]]].any()!=0:
        #     #     user_item_r[[rating_pairs[0][i], rating_pairs[1][i]]]
        record_size = rating_pairs[0].shape[0]
        review_feat_list = [self.review_feat_updated[(rating_pairs[0][x], rating_pairs[1][x])] for x in
                            range(record_size)]  # 双向
        review_feat_list = torch.stack(review_feat_list).to(torch.float32)

        # 构建用于RED的图 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        rating_row, rating_col = self.train_rating_pairs
        graph_dict_RED = {}
        for rating in self.possible_rating_values:
            ridx = np.where(self.train_rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]

            graph_dict_RED[str(rating)] = dgl.graph((rrow, rcol), num_nodes=self._num_user + self._num_item)
            graph_dict_RED[str(rating)].edata['review_feat'] = review_feat_list[ridx]

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        c = []
        for r_1 in self.possible_rating_values:
            c.append(graph_dict_RED[str(r_1)].in_degrees())
            graph_dict_RED[str(r_1)].ndata['ci_r'] = _calc_norm(graph_dict_RED[str(r_1)].in_degrees(), 0.5)

        c_sum = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 0.5)
        c_sum_mean = _calc_norm(torch.sum(torch.stack(c, dim=0), dim=0), 1)

        for r_1 in self.possible_rating_values:
            graph_dict_RED[str(r_1)].ndata['ci'] = c_sum
            graph_dict_RED[str(r_1)].ndata['ci_mean'] = c_sum_mean
        self.graph_dict_RED = graph_dict_RED
        # 构建用于RED的图 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        graph_data = {}
        # @ss>
        # apath = r'X:\workspace\datasets\Amazon\@aspect\office\ASTs2id.txt'
        with open(apath, 'r') as f:
            data = f.read()
        aspect_sentiment = {}
        # tmp=[]
        for line in data.split("\n"):
            if line == "": continue
            line = line.split("####")
            id, aspect = line[0], line[1]
            if aspect == '{}': continue
            id = ast.literal_eval(id)  # [int(x) for x in id[1:-1].split(",")]
            aspect = ast.literal_eval(aspect)
            # tmp+=list(aspect.keys())
            if tuple(id) not in aspect_sentiment:
                aspect_sentiment[tuple(id)] = aspect
            else:
                # 合并aspect                print("重复交互")
                # aspect_sentiment[tuple(id)]
                for a, sjDict in aspect.items():  # {2: {2: 0, 3: 0}}
                    for s, j in sjDict.items():
                        if a in aspect_sentiment[tuple(id)]:  # {2: {2: 0, 3: 0}}
                            if s not in aspect_sentiment[tuple(id)][a]:
                                aspect_sentiment[tuple(id)][a][s] = j
                            else:
                                pass
                        else:
                            aspect_sentiment[tuple(id)][a] = sjDict
                            continue
        all_aspect = []
        for ll in [list(x.keys()) for x in list(aspect_sentiment.values())]:
            all_aspect += ll

        # 将aspect_sentiment的id转为减少后的id
        # import pickle
        # with open(user_ys_path, 'rb') as handle:
        #     user_ys = pickle.load(handle)
        # with open(item_ys_path, 'rb') as handle:
        #     item_ys = pickle.load(handle)
        # aspect_sentiment_tmp = {}
        # for key, value in aspect_sentiment.items():
        #     aspect_sentiment_tmp[user_ys[key[0]], item_ys[key[1]]] = value
        # aspect_sentiment = aspect_sentiment_tmp

        sentiment_feat = torch.load(sentiment_feat_path)

        num_nodes_dict = {"user": self._num_user, "item": self._num_item,
                          "aspect": len(set(all_aspect)), "review": len(self.train_datas[0])}
        # for i in  len(self.train_datas[0]):
        #     u,i,s=self.train_datas[0][i],self.train_datas[1][i],self.train_datas[2][i]
        rrow = self.train_datas[0]
        rcol = [x - self._num_user for x in self.train_datas[1]]
        graph_data[("user", "review", "item")] = (rrow, rcol)
        graph_data[("item", "review_r", "user")] = (rcol, rrow)
        ##>> 二阶
        import pickle
        u_i = {}
        i_u = {}
        for i in range(len(rrow)):
            if rrow[i] not in u_i:
                u_i[rrow[i]] = [rcol[i]]
            else:
                u_i[rrow[i]] += [rcol[i]]
            if rcol[i] not in i_u:
                i_u[rcol[i]] = [rrow[i]]
            else:
                i_u[rcol[i]] += [rrow[i]]
        urow, ucol, urc = [], [], []  ## u-u
        # uiud={}
        # uiud_num = {}
        # for u1 in tqdm(u_i.keys()): ###############################
        #     for u12 in u_i[u1]:
        #         for u2 in i_u[u12]:
        #             if u1 == u2:
        #                 continue
        #             if u1 not in uiud.keys():
        #                 uiud[u1] = {u2:[u12]}
        #                 uiud_num[u1] = {u2:1}
        #             else:
        #                 if u2 not in uiud[u1].keys():
        #                     uiud[u1][u2] = [u12]
        #                     uiud_num[u1][u2] = 1
        #                 else:
        #                     uiud[u1][u2]+=[u12]
        #                     uiud_num[u1][u2] +=1
        # with open('uiud.pkl', 'wb') as f:
        #     pickle.dump(uiud, f, pickle.HIGHEST_PROTOCOL)
        # with open('uiud_num.pkl', 'wb') as f:
        #     pickle.dump(uiud_num, f, pickle.HIGHEST_PROTOCOL)
        # print("保存完成")
        # time.sleep(3)
        with open('uiud.pkl', 'rb') as f:
            uiud = pickle.load(f)
        with open('uiud_num.pkl', 'rb') as f:
            uiud_num = pickle.load(f)
        for u1, u2_num in uiud_num.items():
            sorted_items = sorted(u2_num.items(), key=lambda item: item[1], reverse=True)
            # 仅保留前5个元素
            top_5_items = sorted_items[:10]
            # 将前5个元素转换回字典
            sorted_dict = dict(top_5_items)
            uiud_num[u1] = sorted_dict
        for u1, u2u12 in uiud.items():  ###############################
            for u2, u12s in u2u12.items():
                if u2 not in uiud_num[u1].keys():
                    continue
                for u12 in u12s:
                    urow += [u1]
                    ucol += [u2]
                    urc += [u12]
        irow, icol, irc = [], [], []
        # iuid={}
        # iuid_num = {}
        # for i1 in tqdm(i_u.keys()): ###############################
        #     for i12 in i_u[i1]:
        #         for i2 in u_i[i12]:
        #             if i1==i2:
        #                 continue
        #             if i1 not in iuid.keys():
        #                 iuid[i1] = {i2:[i12]}
        #                 iuid_num[i1] = {i2:1}
        #             else:
        #                 if i2 not in iuid[i1].keys():
        #                     iuid[i1][i2] =[i12]
        #                     iuid_num[i1][i2] = 1
        #                 else:
        #                     iuid[i1][i2]+=[i12]
        #                     iuid_num[i1][i2] += 1
        # with open('iuid.pkl', 'wb') as f:
        #     pickle.dump(iuid, f, pickle.HIGHEST_PROTOCOL)
        # with open('iuid_num.pkl', 'wb') as f:
        #     pickle.dump(iuid_num, f, pickle.HIGHEST_PROTOCOL)
        # print("保存完成")
        # time.sleep(3)
        with open('iuid.pkl', 'rb') as f:
            iuid = pickle.load(f)
        with open('iuid_num.pkl', 'rb') as f:
            iuid_num = pickle.load(f)
        for i1, i2_num in iuid_num.items():
            sorted_items = sorted(i2_num.items(), key=lambda item: item[1], reverse=True)
            # 仅保留前5个元素
            top_5_items = sorted_items[:10]
            # 将前5个元素转换回字典
            sorted_dict = dict(top_5_items)
            iuid_num[i1] = sorted_dict
        for i1, i2i12 in iuid.items():
            for i2, i12s in i2i12.items():
                if i2 not in iuid_num[i1].keys():
                    continue
                for i12 in i12s:
                    irow += [i1]
                    icol += [i2]
                    irc += [i12]

        ##《《二阶

        aurow = []
        aucol = []
        airow = []
        aicol = []

        aurow5 = []
        aucol5 = []
        airow5 = []
        aicol5 = []

        su = []
        si = []
        ju = []
        ji = []
        sus = []
        sis = []
        a2r1 = []
        a2r2 = []
        a2rs = []
        # a2rj = []
        ##统计每个用户项目aspect出现的个数
        #######
        # u_a_w = np.array()
        #######
        # #去除1个去除1个去除1个去除1个去除1个去除1个去除1个去除1个去除1个去除1个
        # aspect_sentiment_tmp={}
        # for ask,asv in aspect_sentiment.items():
        #     asd={}
        #     for asvk,asvv in asv.items():
        #         if len(asvv)!=1:
        #             asd[asvk]=asvv
        #     if len(asd)!=0:
        #         aspect_sentiment_tmp[ask]=asd

        # aspect_sentiment=aspect_sentiment_tmp#########10.3##############################################################################################################
        for x in range(len(rrow)):
            # if self.train_datas[2][x]!=5:continue
            if (rrow[x], rcol[x]) in aspect_sentiment:
                score = self.train_datas[2][x]
                for a, v in aspect_sentiment[(rrow[x], rcol[x])].items():
                    a2r1 += [a]  ## 评论增强
                    a2r2 += [x]  ## 评论增强
                    aurow += [a]  ## 增强用户
                    aucol += [rrow[x]]  ## 增强用户
                    sus += [score]
                    airow += [a]  ## 增强项目
                    aicol += [rcol[x]]  ## 增强项目
                    if score == 5:
                        aurow5 += [a]  ## 增强用户
                        aucol5 += [rrow[x]]  ## 增强用户
                        airow5 += [a]  ## 增强项目
                        aicol5 += [rcol[x]]  ## 增强项目
                    sis += [score]
                    a2rs_temp = []
                    su_temp = []
                    juitemp = []
                    for s, j in v.items():
                        a2rs_temp += [sentiment_feat[s]]
                        su_temp += [sentiment_feat[s]]
                        juitemp += [j]
                        # ju += [j]
                        # si_temp += [sentiment_feat[s]]
                        # ji += [j]
                    a2rs_temp = torch.mean(torch.stack(a2rs_temp, 0), 0)
                    su_temp = torch.mean(torch.stack(su_temp, 0), 0)
                    juitemp = sum(juitemp) / len(juitemp)
                    a2rs += [a2rs_temp]
                    su += [su_temp]
                    si += [su_temp]
                    ju += [juitemp]
                    ji += [juitemp]
            else:
                pass  # print("(rrow[x], rcol[x]) not in aspect_sentiment")

        ##aspect高阶》》》》》》》》
        # uaui=[]
        # uauj=[]
        # def geng(row,col):
        #     rcd={}
        #     crd={}
        #     rcd2,rcd2num={},{}
        #     crd2,crd2num = {},{}
        #     for i in range(len(row)):
        #         if row[i] not in rcd:
        #             rcd[row[i]] = [col[i]]
        #         else:
        #             rcd[row[i]] += [col[i]]
        #
        #         if col[i] not in crd:
        #             crd[col[i]] = [row[i]]
        #         else:
        #             crd[col[i]] += [row[i]]
        #     ###---
        #     for k,v in rcd.items():
        #         for v1 in crd[v]:
        #             if k not in rcd2:
        #                 rcd2[k]={v1:v}
        #                 rcd2num[k]={v1:1}
        #             else:
        #                 if v1 not in rcd2[k]:
        #                     rcd2[k] = {v1: v}
        #                     rcd2num[k] = {v1: 1}
        #                 else:
        #                     rcd2[k]
        #     row2,col2=[],[]
        ######>>
        def gen(rrow, rcol):
            import pickle
            u_i = {}
            i_u = {}
            for i in trange(len(rrow)):
                if rrow[i] not in u_i:
                    u_i[rrow[i]] = [rcol[i]]
                else:
                    u_i[rrow[i]] += [rcol[i]]
                if rcol[i] not in i_u:
                    i_u[rcol[i]] = [rrow[i]]
                else:
                    i_u[rcol[i]] += [rrow[i]]
            urow, ucol, urc = [], [], []  ## u-u
            uiud = {}
            uiud_num = {}
            for u1 in tqdm(u_i.keys()):  ###############################
                for u12 in u_i[u1]:
                    for u2 in i_u[u12]:
                        if u1 == u2:
                            continue
                        if u1 not in uiud.keys():
                            uiud[u1] = {u2: [u12]}
                            uiud_num[u1] = {u2: 1}
                        else:
                            if len(uiud[u1]) > 1000:  #######################先只保留100个
                                break
                            if u2 not in uiud[u1].keys():
                                uiud[u1][u2] = [u12]
                                uiud_num[u1][u2] = 1
                            else:
                                if u12 in uiud[u1][u2]:
                                    continue
                                else:
                                    uiud[u1][u2] += [u12]
                                uiud_num[u1][u2] += 1
            for u1, u2_num in uiud_num.items():
                sorted_items = sorted(u2_num.items(), key=lambda item: item[1], reverse=True)
                # 仅保留前5个元素
                top_5_items = sorted_items[:25]
                # 将前5个元素转换回字典
                sorted_dict = dict(top_5_items)
                uiud_num[u1] = sorted_dict
            for u1, u2u12 in uiud.items():  ###############################
                for u2, u12s in u2u12.items():
                    if u2 not in uiud_num[u1].keys():
                        continue
                    for u12 in u12s:
                        urow += [u1]
                        ucol += [u2]
                        urc += [u12]
            return urow, ucol, urc

        def gen2(au_a, au_u, ai_a, ai_i):
            uia = {}
            uia_n = {}
            for i in range(len(au_a)):
                if au_u[i] not in uia.keys():
                    uia[au_u[i]] = {ai_i[i]: [au_a[i]]}
                    uia_n[au_u[i]] = {ai_i[i]: 1}
                else:
                    if ai_i[i] not in uia[au_u[i]].keys():
                        uia[au_u[i]][ai_i[i]] = [au_a[i]]
                        uia_n[au_u[i]] = {ai_i[i]: 1}
                    else:
                        if au_a[i] in uia[au_u[i]][ai_i[i]]:
                            continue
                        else:
                            uia[au_u[i]][ai_i[i]] += [au_a[i]]
                            uia_n[au_u[i]][ai_i[i]] += 1
            for u1, u2_num in uia_n.items():
                sorted_items = sorted(u2_num.items(), key=lambda item: item[1], reverse=True)
                # 仅保留前5个元素
                top_5_items = sorted_items[:25]
                # 将前5个元素转换回字典
                sorted_dict = dict(top_5_items)
                uia_n[u1] = sorted_dict
            urow, ucol, urc = [], [], []
            for u1, u2u12 in uia.items():  ###############################
                for u2, u12s in u2u12.items():
                    if u2 not in uia_n[u1].keys():
                        continue
                    for u12 in u12s:
                        urow += [u1]
                        ucol += [u2]
                        urc += [u12]
            return urow, ucol, urc

        # re = gen(aucol5, aurow5)  #########
        # red = {}
        # red[0] = re[0]
        # red[1] = re[1]
        # red[2] = re[2]
        # import pickle
        # with open('aucol-aurow-10.pkl', 'wb') as f:
        #     pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
        # #
        # re = gen(aicol5, airow5)  #########
        # red = {}
        # red[0] = re[0]
        # red[1] = re[1]
        # red[2] = re[2]
        # import pickle
        # with open('aicol-airow-10.pkl', 'wb') as f:
        #     pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
        # ##
        # re = gen2(aurow5, aucol5,airow5, aicol5)  #########gen2(au_a, au_u,ai_a, ai_i):
        # red = {}
        # red[0] = re[0]
        # red[1] = re[1]
        # red[2] = re[2]
        # import pickle
        # with open('aucol-aicol-10.pkl', 'wb') as f:-+
        #     pickle.dump(red, f, pickle.HIGHEST_PROTOCOL)
        # print("over")
        # # os.exit()
        ##<<
        with open('aucol-aurow-10.pkl', 'rb') as f:
            aucol_aurow = pickle.load(f)
        with open('aicol-airow-10.pkl', 'rb') as f:
            aicol_airow = pickle.load(f)
        with open('aucol-aicol-10.pkl', 'rb') as f:
            aucol_aicol = pickle.load(f)
        graph_data[("user", "aucol_aurow", "user")] = (aucol_aurow[0], aucol_aurow[1])
        graph_data[("item", "aicol_airow", "item")] = (aicol_airow[0], aicol_airow[1])
        graph_data[("user", "aucol_aicol", "item")] = (aucol_aicol[0], aucol_aicol[1])
        graph_data[("item", "aucol_aicol_r", "user")] = (aucol_aicol[1], aucol_aicol[0])
        ##aspect高阶《《《《《《《《
        graph_data[("aspect", "aspect->user", "user")] = (aurow, aucol)
        graph_data[("aspect", "aspect->item", "item")] = (airow, aicol)
        ##
        # graph_data[("user", "user->aspect", "aspect")] = (aucol,aurow)
        # graph_data[("item", "item->aspect", "aspect")] = (aicol,airow)
        graph_data[("aspect", "aspect->review", "review")] = (a2r1, a2r2)

        ##> 二阶
        graph_data[("user", "user<item>user", "user")] = (urow, ucol)
        graph_data[("item", "item<user>item", "item")] = (irow, icol)
        ##< 二阶

        graph = dgl.heterograph(graph_data, num_nodes_dict)
        ##########################   找到没有aspect连接的
        aumask = []
        for iu in range(self._num_user):
            if iu not in graph_data[("aspect", "aspect->user", "user")][1]:
                aumask += [0]
            else:
                aumask += [1]
        graph.nodes['user'].data['amask'] = torch.tensor(aumask)
        aimask = []
        for iu in range(self._num_item):
            if iu not in graph_data[("aspect", "aspect->item", "item")][1]:
                aimask += [0]
            else:
                aimask += [1]
        graph.nodes['item'].data['amask'] = torch.tensor(aimask)

        ##########################找到没有aspect连接的
        graph.edges['review'].data['review_feat'] = review_feat_list[:int(len(review_feat_list) / 2)]

        graph.edges['review_r'].data['review_feat'] = review_feat_list[:int(len(review_feat_list) / 2)]

        graph.edges['review'].data['score'] = torch.tensor([x - 1 for x in self.train_datas[2]]).int()
        graph.edges['review_r'].data['score'] = torch.tensor([x - 1 for x in self.train_datas[2]]).int()

        graph.edges['aspect->user'].data['sentiment_feat'] = torch.stack(su, 0).float()
        graph.edges['aspect->user'].data['score'] = torch.tensor([x - 1 for x in sus]).int()
        graph.edges['aspect->user'].data['jixing'] = torch.tensor(ju).int()
        graph.edges['aspect->item'].data['sentiment_feat'] = torch.stack(si, 0).float()
        graph.edges['aspect->item'].data['score'] = torch.tensor([x - 1 for x in sis]).int()
        graph.edges['aspect->item'].data['jixing'] = torch.tensor(ji).int()

        ##
        # graph.edges['user->aspect'].data['sentiment_feat'] = torch.stack(su, 0).float()
        # graph.edges['user->aspect'].data['score'] = torch.tensor([x - 1 for x in sus]).int()
        # graph.edges['item->aspect'].data['sentiment_feat'] = torch.stack(si, 0).float()
        # graph.edges['item->aspect'].data['score'] = torch.tensor([x - 1 for x in sis]).int()

        # graph.edges['aspect->review'].data['sentiment_feat'] = torch.stack(a2rs, 0).float()
        # graph.edges['aspect->review'].data['jixing'] = torch.tensor(a2rj).int()
        # attention

        def _calc_norm(x, d):
            x = x.numpy().astype('float32')
            x[x == 0.] = np.inf
            x = torch.FloatTensor(1. / np.power(x, d))
            return x.unsqueeze(1)

        # ca_sum = _calc_norm(graph.out_degrees(etype='aspect->user'), 0.5)
        graph.nodes["user"].data["cur"] = _calc_norm(graph.out_degrees(etype='review'), 0.5)
        graph.nodes["item"].data["cir"] = _calc_norm(graph.out_degrees(etype='review_r'), 0.5)

        graph.nodes["aspect"].data["cau"] = _calc_norm(graph.out_degrees(etype='aspect->user'), 0.5)
        graph.nodes["aspect"].data["cai"] = _calc_norm(graph.out_degrees(etype='aspect->item'), 0.5)
        graph.nodes["aspect"].data["car"] = _calc_norm(graph.out_degrees(etype='aspect->review'), 0.5)
        graph.nodes["review"].data["car"] = _calc_norm(graph.in_degrees(etype='aspect->review'), 0.5)
        graph.nodes["user"].data["cau"] = _calc_norm(graph.in_degrees(etype='aspect->user'), 0.5)
        graph.nodes["item"].data["cai"] = _calc_norm(graph.in_degrees(etype='aspect->item'), 0.5)

        ##> 二阶
        graph.nodes["user"].data["cuuq"] = _calc_norm(graph.out_degrees(etype='user<item>user'), 0.5)
        graph.nodes["item"].data["ciiq"] = _calc_norm(graph.out_degrees(etype='item<user>item'), 0.5)
        graph.nodes["user"].data["cuuqr"] = _calc_norm(graph.in_degrees(etype='user<item>user'), 0.5)
        graph.nodes["item"].data["ciiqr"] = _calc_norm(graph.in_degrees(etype='item<user>item'), 0.5)
        # graph.edges['user<item>user'].data['item'] = torch.tensor(urc)
        # graph.edges['item<user>item'].data['user'] = torch.tensor(irc)
        ##>
        graph.edges['aucol_aurow'].data['aspect'] = torch.tensor(aucol_aurow[2])  ####
        graph.edges['aicol_airow'].data['aspect'] = torch.tensor(aicol_airow[2])  ####
        graph.edges['aucol_aicol'].data['aspect'] = torch.tensor(aucol_aicol[2])  ####
        graph.edges['aucol_aicol_r'].data['aspect'] = torch.tensor(aucol_aicol[2])  ####反向
        graph.nodes["user"].data["caucol_aurow"] = _calc_norm(graph.in_degrees(etype='aucol_aurow'), 0.5)
        graph.nodes["user"].data["caucol_aurow_r"] = _calc_norm(graph.out_degrees(etype='aucol_aurow'), 0.5)
        graph.nodes["item"].data["caicol_airow"] = _calc_norm(graph.in_degrees(etype='aicol_airow'), 0.5)
        graph.nodes["item"].data["caicol_airow_r"] = _calc_norm(graph.out_degrees(etype='aicol_airow'), 0.5)
        graph.nodes["user"].data["caucol_aicol"] = _calc_norm(graph.out_degrees(etype='aucol_aicol'), 0.5)
        graph.nodes["item"].data["caucol_aicol"] = _calc_norm(graph.in_degrees(etype='aucol_aicol'), 0.5)
        graph.nodes["item"].data["caucol_aicol_r"] = _calc_norm(graph.out_degrees(etype='aucol_aicol_r'), 0.5)  ##反向
        graph.nodes["user"].data["caucol_aicol_r"] = _calc_norm(graph.in_degrees(etype='aucol_aicol_r'), 0.5)  ##反向
        ##< 二阶

        return graph

    def _train_data(self, batch_size=1024):
        # ;nnnnnnnssssssssssssssssdsssddddsssssssss
        rating_pairs, rating_values = self.train_rating_pairs, self.train_rating_values
        idx = np.arange(0, len(rating_values))
        np.random.shuffle(idx)
        rating_pairs = (rating_pairs[0][idx], rating_pairs[1][idx])
        rating_values = rating_values[idx]

        data_len = len(rating_values)

        users, items = rating_pairs[0], rating_pairs[1]
        u_list, i_list, r_list = [], [], []
        review_list = []
        n_batch = data_len // batch_size + 1 if data_len != batch_size else 1

        for i in range(n_batch):
            begin_idx = i * batch_size
            end_idx = begin_idx + batch_size
            batch_users, batch_items, batch_ratings = users[begin_idx: end_idx], items[
                                                                                 begin_idx: end_idx], rating_values[
                                                                                                      begin_idx: end_idx]
            batch_reviews = self.train_review_pairs[begin_idx: end_idx]

            u_list.append(torch.LongTensor(batch_users).to('cuda:0'))
            i_list.append(torch.LongTensor(batch_items).to('cuda:0'))
            r_list.append(torch.LongTensor(batch_ratings - 1).to('cuda:0'))

            review_list.append(torch.FloatTensor(batch_reviews).to('cuda:0'))

        return u_list, i_list, r_list  # u_list，i_list有28358个不同条set(list(u_list[0].data.cpu().numpy()))

    def _test_data(self, flag='valid'):
        if flag == 'valid':
            rating_pairs, rating_values = self.valid_rating_pairs, self.valid_rating_values
        else:
            rating_pairs, rating_values = self.test_rating_pairs, self.test_rating_values
        u_list, i_list, r_list = [], [], []
        for i in range(len(rating_values)):
            u_list.append(rating_pairs[0][i])
            i_list.append(rating_pairs[1][i])
            r_list.append(rating_values[i])
        u_list = torch.LongTensor(u_list).to('cuda:0')
        i_list = torch.LongTensor(i_list).to('cuda:0')
        r_list = torch.FloatTensor(r_list).to('cuda:0')
        return u_list, i_list, r_list


def config():
    parser = argparse.ArgumentParser(description='model')
    parser.add_argument('--device', default='0', type=int, help='gpu.')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")

    parser.add_argument('--gcn_dropout', type=float, default=0.5)
    parser.add_argument('--train_max_iter', type=int, default=1000)
    parser.add_argument('--train_optimizer', type=str, default="Adam")
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_early_stopping_patience', type=int, default=50)

    args = parser.parse_args()
    args.model_short_name = 'RGC'
    args.dataset_name = dataset_name
    args.dataset_path = dataset_name_path
    args.emb_size = 64
    args.emb_dim = 64

    args.gcn_dropout = 0.7
    args.device = torch.device(args.device)
    args.train_max_iter = 1000
    # batch_size 设置为数据集的交互数量
    args.batch_size = 271466

    return args


gloabl_dropout = 0.7
global_review_size = 64

class GCN_RED(nn.Module):
    def __init__(self):
        super(GCN_RED, self).__init__()
        self.dropout = nn.Dropout(0.7)
        self.review_w = nn.Linear(64, int(global_review_size), bias=False, device='cuda:0')
        # self.emb_w = nn.Linear(12, global_review_size, bias=False, device='cuda:0')

    def forward(self, g, feature):
        g.srcdata['h_r'] = feature
        g.edata['r'] = self.review_w(g.edata['review_feat'])

        # RED中应该使用拼接concate，需要修改并实验验证
        g.update_all(lambda edges: {
            'm': (edges.data['r'] + edges.src['h_r']) * self.dropout(edges.src['ci'])},
                     fn.sum(msg='m', out='h'))

        rst = g.dstdata['h'] * g.dstdata['ci']

        return rst


class ContrastLoss(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, feat_size))
        torch.nn.init.xavier_uniform_(self.w.data)
        #  self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x @ self.w * y).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x @ self.w * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss


class ContrastLoss_(nn.Module, ABC):

    def __init__(self, feat_size):
        super(ContrastLoss_, self).__init__()
        # self.H = nn.Parameter(torch.Tensor(feat_size, feat_size))
        # torch.nn.init.xavier_uniform_(self.H.data)
        #  self.bilinear = nn.Bilinear(feat_size, feat_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # positive
        #  scores = self.bilinear(x, y).squeeze()
        scores = (x * y).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        #  neg2_scores = self.bilinear(x, y_neg).squeeze()
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg2_scores = (x * y_neg).sum(1)
        neg2_labels = neg2_scores.new_zeros(neg2_scores.shape)
        neg2_loss = self.bce_loss(neg2_scores, neg2_labels)

        loss = pos_loss + neg2_loss
        return loss

class GCN(nn.Module):
    def __init__(self, params, dropout_rate):
        super(GCN, self).__init__()
        self.num_users = params.num_users
        self.num_items = params.num_items
        self.dropout = nn.Dropout(0.7)
        # self.dropout_sta = nn.Dropout(0.7)
        self.dropout_hma = nn.Dropout(0.53)
        self.dropout1 = nn.Dropout(0.4)
        self.score = nn.Embedding(5, params.emb_dim * 4)
        self.score_r = nn.Embedding(5, params.emb_dim * 4)
        self.review_w = nn.Linear(params.emb_size, params.emb_dim * 4, bias=False)
        self.review_r_w = nn.Linear(params.emb_size, params.emb_dim * 4, bias=False)

        self.score2 = nn.Embedding(5, params.emb_dim * 1)
        self.score_r2 = nn.Embedding(5, params.emb_dim * 1)
        self.review_w2 = nn.Linear(params.emb_size, params.emb_dim * 1, bias=False)
        self.review_r_w2 = nn.Linear(params.emb_size, params.emb_dim * 1, bias=False)

        # self.sentiment_a_r = nn.Linear(params.emb_size, params.emb_dim, bias=False)  ###
        self.aspect_w = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.aspect_w_r = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.aspect_w_3 = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.sentiment_w = nn.Linear(params.emb_dim, params.emb_dim, bias=False)  ###
        self.sentiment_w_r = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        # self.sentiment_w_3 = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        # @ss>
        aspect_feat = torch.load(aspect_feat_path)

        self.aspect_feat = torch.stack(list(aspect_feat.values())).to(torch.float32).cuda()

        self.w34 = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
        self.weight = nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))

        self.contrast_loss = ContrastLoss(params.emb_dim)
        # self.contrast_loss_hma = ContrastLoss(params.emb_dim)
        # self.contrast_loss_sta = ContrastLoss_(params.emb_dim)

    def forward(self, g, feature):
        g.nodes["user"].data["fe"], g.nodes["item"].data["fe"] = torch.split(feature, [self.num_users, self.num_items],
                                                                             dim=0)
        g.nodes["user"].data["fee"], g.nodes["item"].data["fee"] = torch.split(self.weight,
                                                                               [self.num_users, self.num_items], dim=0)

        g.nodes["aspect"].data['fe'] = self.aspect_w(self.aspect_feat)
        g.nodes["aspect"].data['fe1'] = self.aspect_w_r(self.aspect_feat)

        g.edges['review'].data['r'] = self.review_w(g.edges['review'].data['review_feat'])
        g.edges['review_r'].data['r'] = self.review_r_w(g.edges['review_r'].data['review_feat'])
        g.edges['review'].data['s'] = self.score(g.edges['review'].data['score'])
        g.edges['review_r'].data['s'] = self.score_r(g.edges['review_r'].data['score'])

        g.edges['review'].data['r2'] = self.review_w2(g.edges['review'].data['review_feat'])
        g.edges['review_r'].data['r2'] = self.review_r_w2(g.edges['review_r'].data['review_feat'])
        g.edges['review'].data['s2'] = self.score2(g.edges['review'].data['score'])
        g.edges['review_r'].data['s2'] = self.score_r2(g.edges['review_r'].data['score'])

        g.edges['aspect->user'].data['r'] = self.sentiment_w(g.edges['aspect->user'].data['sentiment_feat'])
        # g.edges['aspect->user'].data['a_score'] = self.score_a(g.edges['aspect->user'].data['score'])
        g.edges['aspect->item'].data['r'] = self.sentiment_w_r(g.edges['aspect->item'].data['sentiment_feat'])

        g.edges['aucol_aurow'].data['r'] = g.nodes["aspect"].data["fe1"][g.edges['aucol_aurow'].data['aspect']]
        g.edges['aicol_airow'].data['r'] = g.nodes["aspect"].data["fe1"][g.edges['aicol_airow'].data['aspect']]
        g.edges['aucol_aicol'].data['r'] = g.nodes["aspect"].data["fe1"][g.edges['aucol_aicol'].data['aspect']]
        g.edges['aucol_aicol_r'].data['r'] = g.nodes["aspect"].data["fe1"][g.edges['aucol_aicol_r'].data['aspect']]

        # 第一个CL图
        funcs_cl_1 = {
            "aspect->user": (lambda edges: {  # STA
                'm': ((edges.src['fe'] + edges.data['r'])) * self.dropout1(
                    edges.src['cau'])}, fn.sum(msg='m', out='h')),
            "aspect->item": (lambda edges: {  # STA
                'm': ((edges.src['fe'] + edges.data['r'])) * self.dropout1(
                    edges.src['cai'])}, fn.sum(msg='m', out='h')),
            "aucol_aurow": (  # 用户-aspect-用户
                lambda edges: {'m': (edges.src['fe'] + edges.data['r']) * self.dropout(edges.src['caucol_aurow'])},
                fn.sum(msg='m', out='h1')),
            "aicol_airow": (  # 项目-aspect-项目
                lambda edges: {'m': (edges.src['fe'] + edges.data['r']) * self.dropout(edges.src['caicol_airow'])},
                fn.sum(msg='m', out='h2')),
        }
        g.multi_update_all(funcs_cl_1, "stack")
        CL1_u1 = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cau']
        CL1_u2 = g.nodes['user'].data['h1'][:, 0, :] * g.nodes['user'].data['caucol_aurow_r']
        CL1_i1 = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['cai']
        CL1_i2 = g.nodes['item'].data['h2'][:, 0, :] * g.nodes['item'].data['caicol_airow_r']

        # 第二个CL图
        funcs_cl_2 = {
            "aspect->user": (lambda edges: {  # STA
                'm': ((edges.src['fe'] + edges.data['r'])) * self.dropout_hma(
                    edges.src['cau'])}, fn.sum(msg='m', out='h')),
            "aspect->item": (lambda edges: {  # STA
                'm': ((edges.src['fe'] + edges.data['r'])) * self.dropout_hma(
                    edges.src['cai'])}, fn.sum(msg='m', out='h')),
            "aucol_aurow": (  # 用户-aspect-用户
                lambda edges: {'m': (edges.src['fe'] + edges.data['r']) * self.dropout(edges.src['caucol_aurow'])},
                fn.sum(msg='m', out='h1')),
            "aicol_airow": (  # 项目-aspect-项目
                lambda edges: {'m': (edges.src['fe'] + edges.data['r']) * self.dropout(edges.src['caicol_airow'])},
                fn.sum(msg='m', out='h2')),
        }
        g.multi_update_all(funcs_cl_2, "stack")
        CL2_u1 = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cau']
        CL2_u2 = g.nodes['user'].data['h1'][:, 0, :] * g.nodes['user'].data['caucol_aurow_r']
        CL2_i1 = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['cai']
        CL2_i2 = g.nodes['item'].data['h2'][:, 0, :] * g.nodes['item'].data['caicol_airow_r']



        contrast_loss = self.contrast_loss(CL1_u1, CL2_u2).mean() + \
                        self.contrast_loss(CL1_u2, CL2_u1).mean() + \
                        self.contrast_loss(CL1_i1, CL2_i2).mean() + \
                        self.contrast_loss(CL1_i2, CL2_i1).mean() + \
                        self.contrast_loss(CL1_u1, CL2_u1).mean() + \
                        self.contrast_loss(CL1_i1, CL2_i1).mean()#+ \
                        # 0 * self.contrast_loss(CL1_u2, CL2_u2).mean() + \
                        # 0 * self.contrast_loss(CL1_i2, CL2_i2).mean()



        # contrast_loss = self.contrast_loss(CL1_u1, CL1_u2).mean() + \
        #                 self.contrast_loss(CL1_i1, CL1_i2).mean() + \
        #                 self.contrast_loss(CL1_u1, CL2_u1).mean() + \
        #                 self.contrast_loss(CL1_i1, CL2_i1).mean() \
            # +\
        # self.contrast_loss_sta(CL1_u2, CL2_u2).mean()+\
        # self.contrast_loss_sta(CL1_i2, CL2_i2).mean()

        g.nodes['user'].data['from_a'] = torch.cat([CL1_u1, CL1_u2, CL2_u1, CL2_u2], -1)
        g.nodes['item'].data['from_a'] = torch.cat([CL1_i1, CL1_i2, CL2_i1, CL2_i2], -1)

        funcs1 = {
            "review": (lambda edges: {'m': (edges.src['from_a'] + edges.data['r']) * torch.sigmoid(
                edges.data['s']) * self.dropout(edges.src['cur'])}, fn.sum(msg='m', out='h')),
            "review_r": (lambda edges: {'m': (edges.src['from_a'] + edges.data['r']) * torch.sigmoid(
                edges.data['s']) * self.dropout(edges.src['cir'])}, fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs1, "stack")
        ua = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cur']
        ia = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['cir']

        ############# id-embedding -> 评论和评分嵌入
        funcs2 = {
            "review": (lambda edges: {'m': (edges.src['fee'] + edges.data['r2']) * torch.sigmoid(
                edges.data['s2']) * self.dropout(edges.src['cur'])}, fn.sum(msg='m', out='h')),
            "review_r": (lambda edges: {'m': (edges.src['fee'] + edges.data['r2']) * torch.sigmoid(
                edges.data['s2']) * self.dropout(edges.src['cir'])}, fn.sum(msg='m', out='h')),
        }
        g.multi_update_all(funcs2, "stack")
        u_rr = g.nodes['user'].data['h'][:, 0, :] * g.nodes['user'].data['cur']
        i_rr = g.nodes['item'].data['h'][:, 0, :] * g.nodes['item'].data['cir']
        ###############
        uu = ua  # torch.cat([ua, ua], -1)  # ur #ua#
        ii = ia  # torch.cat([ia, ia], -1)  # ir #ir#

        return torch.cat([uu, ii], 0), torch.cat([u_rr, i_rr], 0), contrast_loss


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        print("#################", params)
        self.weight = nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
        # self.weight1 = nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))
        self.weight_RED = nn.ParameterDict({
            str(r): nn.Parameter(torch.Tensor(params.num_users + params.num_items, params.emb_dim))  # @ss 2512
            for r in [1, 2, 3, 4, 5]
        })

        self.encoder = GCN(params, gloabl_dropout)
        self.encoder_RED = nn.ModuleDict({
            str(i + 1): GCN_RED() for i in range(5)
        })
        self.fc_user_RED = nn.Linear(global_review_size * 5 * 1,
                                     global_review_size * 1 * 1)
        self.fc_item_RED = nn.Linear(global_review_size * 5 * 1,
                                     global_review_size * 1 * 1)
        self.dropout_RED = nn.Dropout(0.7)
        self.predictor_RED = nn.Sequential(
            nn.Linear(global_review_size * 1 * 1,
                      global_review_size * 1 * 1, bias=False),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(global_review_size * 1 * 1, 5, bias=False),
        )
        # self.aspect_encoder=GCN2(params, gloabl_dropout)

        self.num_user = params.num_users
        self.num_item = params.num_items

        self.fc_user2 = nn.Linear(params.emb_dim * 4, params.emb_dim * 4)
        self.fc_item2 = nn.Linear(params.emb_dim * 4, params.emb_dim * 4)

        self.dropout1 = nn.Dropout(0.3)  # 0.3)#gloabl_dropout)

        self.predictor1 = nn.Sequential(
            nn.Linear(params.emb_dim * 4, params.emb_dim * 4, bias=False),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(params.emb_dim * 4, 5, bias=False),
        )

        self.fc_user_rr = nn.Linear(params.emb_dim * 1, params.emb_dim * 1)
        self.fc_item_rr = nn.Linear(params.emb_dim * 1, params.emb_dim * 1)

        self.dropout_rr = nn.Dropout(0.3)  # 0.3)#gloabl_dropout)

        self.predictor_rr = nn.Sequential(
            nn.Linear(params.emb_dim * 1, params.emb_dim * 1, bias=False),
            nn.ReLU(),
            nn.Linear(params.emb_dim * 1, 5, bias=False),
        )
        self.gen_rew = nn.Sequential(
            nn.Linear(global_review_size * 4,
                      global_review_size * 4, bias=False),
            nn.Tanh(),
            nn.Linear(global_review_size * 4,
                      global_review_size * 1, bias=False),
        )

        self.gen_rew_2 = nn.Sequential(
            nn.Linear(global_review_size * 1,
                      global_review_size * 1, bias=False),
            nn.Tanh(),
            nn.Linear(global_review_size * 1,
                      global_review_size * 1, bias=False),
        )
        self.gen_rew_3 = nn.Sequential(
            nn.Linear(global_review_size * 1,
                      global_review_size * 1, bias=False),
            nn.Tanh(),
            nn.Linear(global_review_size * 1,
                      global_review_size * 1, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def l2_norm_loss(self, x, y):  # 特征蒸馏损失
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=-1)
        l2_norm_loss = torch.nn.functional.mse_loss(x_norm, y_norm)
        return l2_norm_loss

    def get_loss_sim_inf(self, x, y):  # 特征蒸馏损失
        return - torch.mean(torch.cosine_similarity(x, y, dim=-1))

    def forward(self, enc_graph_dict, graph_dict_RED, users, items):
        # 用于RED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        feat_all_RED = []
        feat_all_user_RED, feat_all_item_RED = [], []
        rate_list = [1, 2, 3, 4, 5]
        for r in rate_list:
            feat_RED = self.encoder_RED[str(r)](graph_dict_RED[str(r)], self.weight_RED[str(r)])
            feat_all_RED.append(feat_RED)
            feat_all_user_RED.append(feat_RED[users])
            feat_all_item_RED.append(feat_RED[items])

        feat_RED = torch.cat(feat_all_RED, dim=-1)
        user_feat_RED, item_feat_RED = torch.split(feat_RED, [self.num_user, self.num_item], dim=0)
        user_feat_RED = self.dropout_RED(user_feat_RED)
        u_feat_RED = self.fc_user_RED(user_feat_RED)

        item_feat_RED = self.dropout_RED(item_feat_RED)
        i_feat_RED = self.fc_item_RED(item_feat_RED)
        feat_RED = torch.cat([u_feat_RED, i_feat_RED], dim=0)
        user_embeddings_RED, item_embeddings_RED = feat_RED[users], feat_RED[items]
        pred_ratings_RED = self.predictor_RED(user_embeddings_RED * item_embeddings_RED).squeeze()

        # 用于RED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # aspect->评论评级图,  单独评论评级图rr
        feat, feat_rr, contrast_loss = self.encoder(enc_graph_dict, self.weight)
        # feat
        u_feat, i_feat = torch.split(feat, [self.num_user, self.num_item], dim=0)
        ua = self.fc_user2(self.dropout1(u_feat))
        ia = self.fc_item2(self.dropout1(i_feat))
        feat = torch.cat([ua, ia], dim=0)
        user_embeddings, item_embeddings = feat[users], feat[items]  # @ss 这边的users或items是指所有用户项目
        pred_ratings1 = self.predictor1(user_embeddings * item_embeddings)
        # feat_rr 评论和评级
        u_feat_rr, i_feat_rr = torch.split(feat_rr, [self.num_user, self.num_item], dim=0)
        u_rr = self.fc_user_rr(self.dropout1(u_feat_rr))
        i_rr = self.fc_item_rr(self.dropout1(i_feat_rr))
        feat_rr = torch.cat([u_rr, i_rr], dim=0)
        user_embeddings_rr, item_embeddings_rr = feat_rr[users], feat_rr[items]  # @ss 这边的users或items是指所有用户项目
        pred_ratings_rr = self.predictor_rr(user_embeddings_rr * item_embeddings_rr)

        # 特征蒸馏: feat_RED,feat_rr,feat
        feat_loss = 0.

        # 和评论蒸馏
        if self.training:
            # print(user_embeddings.shape)
            gen_rew = torch.chunk(self.gen_rew(user_embeddings * item_embeddings), 2, 0)[0]
            # print(gen_rew.shape)

            gen_rew_rr = torch.chunk(self.gen_rew_2(user_embeddings_rr * item_embeddings_rr), 2, 0)[0]
            gen_rew_RED = torch.chunk(self.gen_rew_3(user_embeddings_RED * item_embeddings_RED), 2, 0)[0]
            re0 = enc_graph_dict['user', 'review', 'item'].edata["review_feat"]

            feat_loss = self.get_loss_sim_inf(re0,gen_rew_RED)+self.get_loss_sim_inf(re0,gen_rew)+self.get_loss_sim_inf(re0,gen_rew_rr)

        return pred_ratings1, pred_ratings_rr, pred_ratings_RED, feat_loss, contrast_loss


def evaluate(args, net, dataset, flag='valid'):
    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(args.device)

    u_list, i_list, r_list = dataset._test_data(flag=flag)

    enc_graph = dataset.train_enc_graph
    enc_graph_RED = dataset.graph_dict_RED
    # graph_aspect=dataset.graph_aspect

    net.eval()
    with torch.no_grad():
        pred_ratings, pred_ratings_rr, pred_ratings_RED,  _, _ = net(enc_graph, enc_graph_RED, u_list, i_list)

        real_pred_ratings = (torch.softmax(pred_ratings_rr, dim=1) *
                             nd_possible_rating_values.view(1, -1)).sum(dim=1)

        real_pred_ratings2 = (torch.softmax(pred_ratings, dim=1) *
                              nd_possible_rating_values.view(1, -1)).sum(dim=1)

        real_pred_ratings3 = (torch.softmax(pred_ratings_RED, dim=1) *
                              nd_possible_rating_values.view(1, -1)).sum(dim=1)

        real_pred_ratings_mean = (torch.softmax((pred_ratings_RED + pred_ratings_rr + pred_ratings) / 3, dim=1) *
                                  nd_possible_rating_values.view(1, -1)).sum(dim=1)

        # real_pred_ratings = (real_pred_ratings + real_pred_ratings2 + real_pred_ratings3 + real_pred_ratings_mean) / 4
        real_pred_ratings = (real_pred_ratings + real_pred_ratings2 + real_pred_ratings3) / 3

        u_list = u_list.cpu().numpy()
        r_list = r_list.cpu().numpy()
        real_pred_ratings = real_pred_ratings.cpu().numpy()

        mse = ((real_pred_ratings - r_list) ** 2.).mean()

    return mse


def train(params):
    dataset = Data(params.dataset_name,
                   params.dataset_path,
                   params.device,
                   params.emb_size,
                   )
    print("Loading data finished.\n")

    params.num_users = dataset._num_user
    params.num_items = dataset._num_item

    params.rating_vals = dataset.possible_rating_values
    print(
        f'Dataset information:\n \tuser num: {params.num_users}\n\titem num: {params.num_items}\n\ttrain interaction num: {len(dataset.train_rating_values)}\n')

    net = Net(params)
    net = net.to(params.device)

    rating_loss_net = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    learning_rate = params.train_lr

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
    print("Loading network finished.\n")

    best_test_mse = np.inf
    no_better_valid = 0
    best_iter = -1

    for r in [1, 2, 3, 4, 5]:
        dataset.graph_dict_RED[str(r)] = dataset.graph_dict_RED[str(r)].int().to(params.device)

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(params.device)

    nd_possible_rating_values = torch.FloatTensor([1, 2, 3, 4, 5]).to(params.device)

    kd_mse_loss = nn.MSELoss()
    kd_l1_loss = nn.L1Loss()

    print("Training and evaluation.")
    for iter_idx in range(1, params.train_max_iter):
        net.train()
        # n_batch = len(dataset.train_rating_values) // params.batch_size + 1
        u_list, i_list, r_list = dataset._train_data(batch_size=params.batch_size)
        train_mse = 0.

        for idx in range(len(r_list)):
            batch_user = u_list[idx]
            batch_item = i_list[idx]
            batch_rating = r_list[idx]
            pred_ratings, pred_ratings_rr, pred_ratings_RED, feat_loss, contrast_loss = net(
                dataset.train_enc_graph, dataset.graph_dict_RED, batch_user, batch_item)

            real_pred_ratings = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
            if 1:  # 下面是交叉熵
                loss1 = rating_loss_net(pred_ratings, batch_rating).mean()
                loss1 += rating_loss_net(pred_ratings_rr, batch_rating).mean()
                loss1 += rating_loss_net(pred_ratings_RED, batch_rating).mean()
                # loss1 += 0.5 * rating_loss_net((pred_ratings + pred_ratings_rr + pred_ratings_RED)/3, batch_rating).mean()
            else:  # 下面是MSE, MSE不行
                real_pred_ratings1 = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(
                    dim=1)
                real_pred_ratings2 = (
                            torch.softmax(pred_ratings_rr, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
                real_pred_ratings3 = (
                            torch.softmax(pred_ratings_RED, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
                loss1 = mse_loss(real_pred_ratings1, batch_rating.float())
                loss1 += mse_loss(real_pred_ratings2, batch_rating.float())
                loss1 += mse_loss(real_pred_ratings3, batch_rating.float())
            # real_pred_ratings1 = (torch.softmax(pred_ratings, dim=1) * nd_possible_rating_values.view(1, -1)).sum(dim=1)
            # real_pred_ratings2 = (torch.softmax(pred_ratings_rr, dim=1) * nd_possible_rating_values.view(1, -1)).sum(
            #     dim=1)
            # loss1 = mse_loss(real_pred_ratings1, batch_rating.float())
            # loss1 += mse_loss(real_pred_ratings2, batch_rating.float())
            # loss1 += rating_loss_net(pred_ratings_RED, batch_rating).mean()

            # 下面是蒸馏
            kd_loss1 = (kd_mse_loss(pred_ratings, pred_ratings_rr) + \
                        kd_mse_loss(pred_ratings_rr, pred_ratings_RED) + \
                        kd_mse_loss(pred_ratings_RED, pred_ratings)) / 3

            kd_loss2 = (kd_l1_loss(pred_ratings, pred_ratings_rr) + \
                        kd_l1_loss(pred_ratings_rr, pred_ratings_RED) + \
                        kd_l1_loss(pred_ratings_RED, pred_ratings)) / 3
            kd_loss = -(kd_loss2 + kd_loss1) / 2
            # feat_loss = 0
            loss = loss1 + contrast_loss + feat_loss

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_mse += ((real_pred_ratings - batch_rating - 1) ** 2).sum()

        train_mse = train_mse / len(dataset.train_rating_values)

        # valid_mse = evaluate(args=params, net=net, dataset=dataset, flag='valid')

        test_mse = evaluate(args=params, net=net, dataset=dataset, flag='test')

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_iter = iter_idx
            no_better_valid = 0
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience:
                print("Early stopping threshold reached. Stop training.")
                break

        print(
            f'Epoch {iter_idx}, Loss={loss:.4f}, Train_MSE={train_mse:.4f}, Valid_MSE={0:.4f}, Test_MSE={test_mse:.4f}')
    import datetime

    current_time = datetime.datetime.now()
    print(current_time)
    print(f'Best Iter Idx={best_iter}, Best Test MSE={best_test_mse:.4f}')


if __name__ == '__main__':
    config_args = config()
    train(config_args)

