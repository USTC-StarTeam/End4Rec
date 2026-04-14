import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.communication import get_group_size


def hit_ratio_at_k(preds, labels, k):
    topk = np.argsort(-preds, axis=1)[:, :k]
    hits = 0.0
    for i in range(labels.shape[0]):
        if labels[i] in topk[i]:
            hits += 1.0
    return hits / labels.shape[0]
def ndcg_at_k(preds, labels, k):
    topk = np.argsort(-preds, axis=1)[:, :k]
    ndcg = 0.0
    for i in range(labels.shape[0]):
        if labels[i] in topk[i]:
            rank = np.where(topk[i]==labels[i])[0][0] + 1
            ndcg += 1.0 / np.log2(rank+1)
    return ndcg / labels.shape[0]
def allreduce_mean(tensor):
    from mindspore.communication import get_group_size
    group_size = get_group_size()
    allreduce = ops.AllReduce()
    return allreduce(tensor) / group_size
def evaluate(model, dataset, k=10):
    hr_total = 0.0
    ndcg_total = 0.0
    count = 0
    for data in dataset.create_tuple_iterator():
        item_ids, behavior_ids, positions, labels = data
        logits, *_ = model(item_ids, behavior_ids, positions)
        preds = logits.asnumpy()
        labels_np = labels.asnumpy().squeeze()
        hr = hit_ratio_at_k(preds, labels_np, k)
        ndcg = ndcg_at_k(preds, labels_np, k)
        hr_total += hr
        ndcg_total += ndcg
        count += 1
    hr_avg = allreduce_mean(ms.Tensor(hr_total / count, ms.float32)).asnumpy()
    ndcg_avg = allreduce_mean(ms.Tensor(ndcg_total / count, ms.float32)).asnumpy()
    return hr_avg, ndcg_avg
