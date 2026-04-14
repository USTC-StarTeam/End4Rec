import mindspore as ms
from mindspore import Tensor, dataset as ds
import random
from mindspore.dataset import GeneratorDataset

def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_users = len(lines)
    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, num_users, sample_seq


def get_seq_dic(args):
    args.data_file = args.data_dir + args.data_name + '.txt'
    args.sample_file = args.data_dir + args.data_name + '_sample.txt'
    user_seq, max_item, num_users, sample_seq = \
        get_user_seqs_and_sample(args.data_file, args.sample_file)
    seq_dic = {'user_seq': user_seq, 'num_users': num_users, 'sample_seq': sample_seq}
    return seq_dic, max_item

class EBMRecDataset:
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length

        if data_type == 'train':
            for seq in user_seq:
                input_ids = seq[-(self.max_len + 2):-2]
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])
        elif data_type == 'valid':
            for sequence in user_seq:
                self.user_seq.append(sequence[:-1])
        else:
            self.user_seq = user_seq

        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        seq_set = set(items)
        neg_answer = neg_sample(seq_set, self.args.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        assert len(input_ids) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                Tensor(index, ms.int32),
                Tensor(input_ids, ms.int32),
                Tensor(answer, ms.int32),
                Tensor(neg_answer, ms.int32),
                Tensor(test_samples, ms.int32),
            )
        else:
            cur_tensors = (
                Tensor(index, ms.int32),
                Tensor(input_ids, ms.int32),
                Tensor(answer, ms.int32),
                Tensor(neg_answer, ms.int32),
            )

        return cur_tensors

def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def get_dataloader(args, seq_dic):
    train_dataset = EBMRecDataset(args, seq_dic['user_seq'], data_type='train')
    train_dataloader = GeneratorDataset(train_dataset, column_names=["index", "input_ids", "answer", "neg_answer"], shuffle=True).batch(args.batch_size)

    eval_dataset = EBMRecDataset(args, seq_dic['user_seq'], test_neg_items=seq_dic['sample_seq'], data_type='valid')
    eval_dataloader = GeneratorDataset(eval_dataset, column_names=["index", "input_ids", "answer", "neg_answer", "test_samples"], shuffle=False).batch(args.batch_size)

    test_dataset = EBMRecDataset(args, seq_dic['user_seq'], test_neg_items=seq_dic['sample_seq'], data_type='test')
    test_dataloader = GeneratorDataset(test_dataset, column_names=["index", "input_ids", "answer", "neg_answer", "test_samples"], shuffle=False).batch(args.batch_size)
    
    return train_dataloader, eval_dataloader, test_dataloader



