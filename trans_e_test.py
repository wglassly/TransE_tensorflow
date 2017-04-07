import math
import numpy as np
from util import dataset


####two test methods###
##rank##hit@10##

#### HyperParam Setting###
embedding_size = 50
##########################



def eval_ranking(rank_l, index_l):
    assert len(rank_l) == len(index_l)
    return np.mean([list(r).index(i) for r, i in zip(rank_l, index_l)])


def eval_top_k(rank_l, index_l, k=10):
    assert len(rank_l) == len(index_l)
    z = 0.0
    for r, i in zip(rank_l, index_l):
        if i in list(r[:k]):
            z += 1
    return z / len(index_l)


def l1_distance(h, t, r):
    return np.sum(np.fabs(t - h - r))


def test(path, filter=True):
    # read dataset
    ds = dataset(path)
    entity_size = ds.entity_nums + 1  # add 1 avoid out_of_dict
    relation_size = ds.relation_nums[0] + 1
    model_path = path + '/model/'

    e_emb, r_emb = np.load(model_path + '_TransE_ent.npy'), \
        np.load(model_path + '_TransE_rel.npy')

    # filter build
    def get_head_tail(pairs, filter_list=[]):
        for p in pairs:
            filter_list.append(p[0])
            filter_list.append(p[1])
        filter_list = list(set(filter_list))
        return filter_list

    filter_l = get_head_tail(ds.train_pair)
    filter_l = get_head_tail(ds.test_pair, filter_l)
    filter_l = get_head_tail(ds.val_pair, filter_l)

    print("filter build done.")

    # eval
    eval_h, eval_t, index_h, index_t = [], [], [], []

    for test_p in ds.test_pair[:100]:
        h, t, r = e_emb[test_p[0]], e_emb[test_p[1]], r_emb[test_p[2]]
        index_h.append(test_p[0])
        index_t.append(test_p[1])

        if filter:
            head_predict_list = [l1_distance(e_emb[i], t, r) for i in filter_l]
            tail_predict_list = [l1_distance(h, e_emb[i], r) for i in filter_l]
        else:
            head_predict_list = [l1_distance(e_emb[i], t, r) for i in range(entity_size)]
            tail_predict_list = [l1_distance(h, e_emb[i], r) for i in range(entity_size)]

        head_sorted_rank = np.argsort(head_predict_list)
        tail_sorted_rank = np.argsort(tail_predict_list)

        eval_h.append(head_sorted_rank)
        eval_t.append(tail_sorted_rank)

    h_result = eval_ranking(rank_l=eval_h, index_l=index_h), eval_top_k(
            rank_l=eval_h, index_l=index_h)
    t_result = eval_ranking(rank_l=eval_t, index_l=index_t), eval_top_k(
            rank_l=eval_t, index_l=index_t)

    print("result of h predict is {0} (rank,top_10), t predict is {1}.".format(
            h_result, t_result))
    return h_result, t_result


if __name__ == '__main__':
    test(path='./data')
