import json
from transformers import AutoTokenizer
import torch
import random
import numpy as np
import sklearn.metrics as metrics
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from .dataset import PhemeSet, TwitterSet, WeiboSet
# from dataset import PhemeSet, TwitterSet


def save_file(a, path):
    with open(path, 'w') as f:
        f.write(json.dumps(a))


def read_file(path):
    with open(path, 'r') as f:
        a = json.load(f)
    return a


def compute_mean_std(seeds):
    seeds = np.array(seeds)
    np.mean(seeds[:, 0])
    mean_value = [np.mean(seeds[:, 0]), np.mean(seeds[:, 1]), np.mean(seeds[:, 2]), np.mean(seeds[:, 3])]
    std_value = [np.std(seeds[:, 0]), np.std(seeds[:, 1]), np.std(seeds[:, 2]), np.std(seeds[:, 3])]
    final_value = np.mean(seeds, axis=1)
    final_mean = np.mean(final_value)
    final_std = np.std(final_value)
    return mean_value, std_value, final_mean, final_std


class PadCollate_Pheme:
    def __init__(self, text_dim=0, img_dim=1, label_dim=2, dep_dim=3, type=0, tokenizer_type="bert", vocab_file=None):
        """

        Args:
            text_dim:
            img_dim:
            label_dim:
            dep_dim: the dependencies between words
            type: 0 is without dependency, 1 is with dependency.
            tokenizer_type: different encoding methods, bert, rob
        """
        self.text_dim = text_dim
        self.img_dim = img_dim
        self.label_dim = label_dim
        self.dep_dim = dep_dim
        self.type = type
        self.tokenizer_type = tokenizer_type

        if self.tokenizer_type == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained('./pretrain_model/bert')
        elif self.tokenizer_type == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('./pretrain_model/roberta')
        elif self.tokenizer_type == "chinese":
            self.tokenizer = AutoTokenizer.from_pretrained('./pretrain_model/chinese-roberta-wwm')

        else:
            # for textcnn
            self.tokenizer = get_tokenizer("spacy")
            self.vocab = torch.load(vocab_file)
            self.pad_index = self.vocab['<pad>']

    def pad_collate(self, batch):
        """

        Args:
            batch:
        Returns:
            labels: (N), 1 is rumor, 0 is non-rumor.

        """
        texts = list(map(lambda t: t[self.text_dim], batch))
        encoded_texts = []
        if self.tokenizer_type == "bert" or self.tokenizer_type == "roberta" or self.tokenizer_type == "chinese":
            encoded_texts = self.tokenizer(texts, is_split_into_words=False, return_tensors="pt", truncation=True,
                                           max_length=100, padding=True)['input_ids']
        else:
            for line in texts:
                words = self.tokenizer(line.strip())
                # truncate
                if len(words) > 100:
                    encoded_texts.append(torch.LongTensor(self.vocab(words[:100])))
                else:
                    encoded_texts.append(torch.LongTensor(self.vocab(words)))
            # pad
            encoded_texts = pad_sequence(encoded_texts, batch_first=True, padding_value=self.pad_index)

        imgs = list(map(lambda t: t[self.img_dim].clone().detach(), batch))
        imgs = torch.stack(imgs, dim=0)
        labels = list(map(lambda t: t[self.label_dim], batch))
        labels = torch.tensor(labels, dtype=torch.long)
        if self.type == 0:
            return encoded_texts, imgs, labels

    def __call__(self, batch):
        return self.pad_collate(batch)


def construct_edge_text(deps, max_length, chunk=None):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    np_mask = []

    for i, dep in enumerate(deps):
        if len(dep) > 3 and len(chunk[i]) > 1:
            dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
            gnn_mask.append(False)
            np_mask.append(True)
            dep_np = torch.cat(dep_np, dim=0).T.contiguous()
        else:
            dep_np = torch.tensor([])
            gnn_mask.append(True)
            np_mask.append(False)
            dep_se.append(dep_np.long())

    np_mask = torch.tensor(np_mask).unsqueeze(1)
    np_mask_ = [torch.tensor(
        [True] * max_length) if gnn_mask[i] else torch.tensor([True] * max_length).index_fill_(0, chunk_,
                                                                                               False).clone().detach()
                for i, chunk_ in enumerate(chunk)]
    np_mask_ = torch.stack(np_mask_)
    np_mask = torch.cat([np_mask_, np_mask], dim=1)
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask, np_mask


def seed_everything(seed: int = 0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_mask_text(seq_len, max_length):
    """

    Args:
        seq_len1(N): list of number of words in a caption without padding in a minibatch
        max_length: the dimension one of shape of embedding of captions of a batch

    Returns:
        mask(N,max_length): Boolean Tensor
    """
    # the realistic max length of sequence
    max_len = max(seq_len)
    if max_len <= max_length:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool), torch.ones(max_length - len, dtype=bool)]) for len in seq_len])
    else:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool),
                        torch.ones(max_length - len, dtype=bool)]) if len <= max_length else torch.zeros(max_length,
                                                                                                         dtype=bool) for
             len in seq_len])

    return mask


def get_metrics(y):
    """
        Computes how accurately model learns correct matching of object with the caption in terms of accuracy

        Args:
            y(N,2): Tensor(cpu). the incongruity score of negataive class, positive class.

        Returns:
            predict_label (list): predict results
    """
    predict_label = (y[:, 0] < y[:, 1]).clone().detach().long().numpy().tolist()
    return predict_label


def get_four_metrics(labels, predicted_labels):
    confusion = metrics.confusion_matrix(labels, predicted_labels)
    # tn, fp, fn, tp
    total = confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
    acc = (confusion[0][0] + confusion[1][1]) / total
    # about sarcasm
    # if confusion[1][1] == 0:
    #     recall = 0
    #     precision = 0
    # else:
    #     recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
    #     precision = confusion[1][1] / (confusion[1][1] + confusion[0][1])
    # f1 = 2 * recall * precision / (recall + precision)
    # return acc, recall, precision, f1
    return acc


def get_accuracy(labels: torch.Tensor, y: torch.Tensor):
    # y[:, 0] < y[:, 1] true sarcasm -> 1
    # （N）
    y = y.cpu()
    predict_labels = (y[:, 0] < y[:, 1]).long().numpy()
    labels = labels.cpu().numpy()
    return (labels == predict_labels).sum() / labels.shape[0]


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def l1_norm(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.abs_(a - b)) / a.size(0)


def l2_norm(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.sqrt(torch.sum((a - b) * (a - b), dim=1))) / a.size(0)


def get_three_source_loader(args, train_source_drop=True, train_target_drop=True, test_target_drop=False):
    multi_source_domains_set = []
    multi_source_domains_loader = []
    multi_source_domains_iter = []
    domain_nums = len(args.source)

    vocab_path_pheme = "./dataset/vocab/pheme_vocab.pt"
    vocab_path_twitter = "./dataset/vocab/twitter_vocab.pt"

    dataset_dict = {'Pheme': PhemeSet, 'Twitter': TwitterSet, 'Weibo': WeiboSet}
    DatasetClass = dataset_dict[args.data]

    json_train_path = f"./dataset/{args.data.lower()}/train_70.json"
    json_test_path = f"./dataset/{args.data.lower()}/test_30.json"
    img_path = f"./dataset/{args.data.lower()}/images"


    for i in range(domain_nums):
        if args.data in ['Pheme', 'Twitter', 'Weibo']:
            multi_source_domains_set.append(DatasetClass(json_path=json_train_path, img_path=img_path, type=0,
                                                         events=args.source[i], visual_type='resnet', stage='train'))
            multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.workers,
                                                          drop_last=train_source_drop,
                                                          collate_fn=PadCollate_Pheme(type=0,
                                                                                      tokenizer_type=args.tokenizer_type)))

        elif args.data == 'Cross':
            if args.source == ["charliehebdo", "ottawashooting", "ferguson"]:
                multi_source_domains_set.append(PhemeSet(json_path="./dataset/pheme/train_70.json",
                                                img_path="./dataset/pheme/images",
                                                type=0, events=args.source[i], visual_type='resnet', stage='train'))
                multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers, drop_last=True,
                                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                             vocab_file=
                                                                             vocab_path_pheme)))
            elif args.source == ["sandy", "boston", "sochi"]:
                multi_source_domains_set.append(TwitterSet(
                    json_path="./dataset/twitter/train_70.json",
                    img_path="./dataset/twitter/images",
                    type=0, events=args.source[i], visual_type='resnet', stage='train'))
                multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers, drop_last=True,
                                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                             vocab_file=
                                                                             vocab_path_twitter)))

        else:
            print("Wrong Source Dataset")
            exit()

        multi_source_domains_iter.append(ForeverDataIterator(multi_source_domains_loader[i]))



    if args.data in ['Pheme', 'Twitter', 'Weibo']:

        train_target_dataset = DatasetClass(json_path=json_train_path, img_path=img_path, type=0,
                                            events=args.target, visual_type='resnet', stage='train')
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers, drop_last=train_target_drop,
                                         collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type))
        test_target_dataset = DatasetClass(json_path=json_test_path, img_path=img_path, type=0,
                                           events=args.target, visual_type='resnet', stage='test')
        test_loader = DataLoader(test_target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type), drop_last=test_target_drop)

    elif args.data == 'Cross':
        if args.source == ["charliehebdo", "ottawashooting", "ferguson"]:
            train_target_dataset = TwitterSet(
                json_path="./dataset/twitter/final_twitter.json",
                img_path="./dataset/twitter/images",
                type=0, events=args.target, visual_type='resnet', stage='train')

            train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_twitter), drop_last=True)
            test_loader =  DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_twitter), drop_last=False)

        elif args.source == ["sandy", "boston", "sochi"]:
            train_target_dataset = PhemeSet(json_path="./dataset/pheme/phemeset.json",
                                            img_path="./dataset/pheme/images",
                                            type=0, events=args.target, visual_type='resnet', stage='train')

            train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_pheme), drop_last=True)

            test_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.workers,
                                     collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                 vocab_file=vocab_path_pheme), drop_last=False)

    else:
        print("Wrong Target Dataset")
        exit()

    train_target_iter = ForeverDataIterator(train_target_loader)

    print("source domain: {}".format(args.source))
    print("target domain: {}".format(args.target))

    return multi_source_domains_loader, multi_source_domains_iter, train_target_loader, train_target_iter, test_loader


def get_merge_source_target_ds(args):
    pheme_train_json = "./dataset/pheme/train_70.json"
    pheme_test_json = "./dataset/pheme/test_30.json"
    pheme_img_path = "./dataset/pheme/images"
    weibo_train_json = "./dataset/weibo/train_70.json"
    weibo_test_json = "./dataset/weibo/test_30.json"
    weibo_img_path = "./dataset/weibo/images"

    # dataset_class = {
    #     'Pheme': PhemeSet,
    #     'Twitter': TwitterSet
    # }

    if args.dataset == 'Pheme':
        src_json, src_img = pheme_train_json, pheme_img_path
        tgt_train_json, tgt_test_json, tgt_img = pheme_train_json, pheme_test_json, pheme_img_path
        srcDataset = tgtDataset = PhemeSet
    elif args.dataset == 'Weibo':
        src_json, src_img = weibo_train_json, weibo_img_path
        tgt_train_json, tgt_test_json, tgt_img = weibo_train_json, weibo_test_json, weibo_img_path
        srcDataset = tgtDataset = WeiboSet
    elif args.dataset == 'Cross':
        if "charliehebdo" in args.source:
            src_json, src_img = pheme_train_json, pheme_img_path
            tgt_train_json, tgt_test_json, tgt_img = weibo_train_json, weibo_test_json, weibo_img_path
            srcDataset, tgtDataset = PhemeSet, TwitterSet
        else:
            src_json, src_img = weibo_train_json, weibo_img_path
            tgt_train_json, tgt_test_json, tgt_img = pheme_train_json, pheme_test_json, pheme_img_path
            srcDataset, tgtDataset = TwitterSet, PhemeSet

    src_dataset = srcDataset(json_path=src_json, img_path=src_img, type=0, events=args.source,
                             visual_type='resnet', stage='train')
    tgt_train_dataset = tgtDataset(json_path=tgt_train_json, img_path=tgt_img, type=0, events=args.target,
                                   visual_type='resnet', stage='train')
    tgt_selected = tgtDataset(json_path=tgt_train_json, img_path=tgt_img, type=0, events=[],
                                   visual_type='resnet', stage='train')
    tgt_test_dataset = tgtDataset(json_path=tgt_test_json, img_path=tgt_img, type=0, events=args.target,
                                  visual_type='resnet', stage='test')

    return src_dataset, tgt_train_dataset, tgt_test_dataset, tgt_selected


def get_dataloader(args, dataset, shuffle=True, drop_last=False, return_iter=False):
    if args.dataset == 'Pheme' or isinstance(dataset, PhemeSet):
        vocab_path = "./dataset/vocab/pheme_vocab.pt"
    elif args.dataset == 'Twitter' or isinstance(dataset, TwitterSet):
        vocab_path = "./dataset/vocab/twitter_vocab.pt"
    elif args.dataset == 'Weibo' or isinstance(dataset, WeiboSet):   # args.tokenizer_type = 'chinese'
        vocab_path = None

    # batch_size = args.batch_size if dataset.stage == 'train' else 8
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                         drop_last=drop_last, pin_memory=True, collate_fn=PadCollate_Pheme(type=0,
                                                                                           tokenizer_type=args.tokenizer_type,
                                                                                           vocab_file=vocab_path))

    if return_iter:
        return loader, ForeverDataIterator(loader)
    else:
        return loader


def get_three_source_iter(args, train_source_drop=True):
    multi_source_domains_set = []
    multi_source_domains_loader = []
    multi_source_domains_iter = []
    domain_nums = len(args.source)

    vocab_path_pheme = "./dataset/vocab/pheme_vocab.pt"
    vocab_path_twitter = "./dataset/vocab/twitter_vocab.pt"
    for i in range(domain_nums):
        if args.dataset == 'Pheme':
            multi_source_domains_set.append(PhemeSet(json_path="./dataset/pheme/train_70.json",
                                                     img_path="./dataset/pheme/images",
                                                     type=0, events=args.source[i], visual_type='resnet',
                                                     stage='train'))
            multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.num_workers,
                                                          drop_last=train_source_drop,
                                                          collate_fn=PadCollate_Pheme(type=0,
                                                                                      tokenizer_type=args.tokenizer_type,
                                                                                      vocab_file=vocab_path_pheme)))
        elif args.dataset == 'Twitter':
            multi_source_domains_set.append(TwitterSet(json_path="./dataset/twitter/train_70.json",
                                                       img_path="./dataset/twitter/images",
                                                       type=0, events=args.source[i], visual_type='resnet',
                                                       stage='train'))
            multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.num_workers,
                                                          drop_last=train_source_drop,
                                                          collate_fn=PadCollate_Pheme(type=0,
                                                                                      tokenizer_type=args.tokenizer_type,
                                                                                      vocab_file=vocab_path_twitter)))

        elif args.dataset == 'Weibo':
            multi_source_domains_set.append(WeiboSet(json_path="./dataset/weibo/train_70.json",
                                                       img_path="./dataset/weibo/images",
                                                       type=0, events=args.source[i], visual_type='resnet',
                                                       stage='train'))
            multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                          shuffle=True, num_workers=args.num_workers,
                                                          drop_last=train_source_drop,
                                                          collate_fn=PadCollate_Pheme(type=0,
                                                                                      tokenizer_type='chinese')))

        elif args.dataset == 'Cross':
            if args.source == ["charliehebdo", "ottawashooting", "ferguson"]:
                multi_source_domains_set.append(PhemeSet(json_path="./dataset/pheme/train_70.json",
                                                img_path="./dataset/pheme/images",
                                                type=0, events=args.source[i], visual_type='resnet', stage='train'))
                multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers, drop_last=True,
                                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                             vocab_file=
                                                                             vocab_path_pheme)))
            elif args.source == ["sandy", "boston", "sochi"]:
                multi_source_domains_set.append(TwitterSet(
                    json_path="./dataset/twitter/train_70.json",
                    img_path="./dataset/twitter/images",
                    type=0, events=args.source[i], visual_type='resnet', stage='train'))
                multi_source_domains_loader.append(DataLoader(multi_source_domains_set[i], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.workers, drop_last=True,
                                                 collate_fn=PadCollate_Pheme(type=0, tokenizer_type=args.tokenizer_type,
                                                                             vocab_file=
                                                                             vocab_path_twitter)))

        else:
            print("Wrong Source Dataset")
            exit()

        multi_source_domains_iter.append(ForeverDataIterator(multi_source_domains_loader[i]))


    print("source domain: {}".format(args.source))
    print("target domain: {}".format(args.target))

    return multi_source_domains_loader, multi_source_domains_iter


def get_target_ds(args):
    pheme_train_json = "./dataset/pheme/train_70.json"
    pheme_test_json = "./dataset/pheme/test_30.json"
    pheme_img_path = "./dataset/pheme/images"
    twitter_train_json = "./dataset/twitter/train_70.json"
    twitter_test_json = "./dataset/twitter/test_30.json"
    twitter_img_path = "./dataset/twitter/images"
    weibo_train_json = "./dataset/weibo/train_70.json"
    weibo_test_json = "./dataset/weibo/test_30.json"
    weibo_img_path = "./dataset/weibo/images"

    if args.dataset == 'Pheme':
        tgt_train_json, tgt_test_json, tgt_img = pheme_train_json, pheme_test_json, pheme_img_path
        datasetclass = PhemeSet

    elif args.dataset == 'Twitter':
        tgt_train_json, tgt_test_json, tgt_img = twitter_train_json, twitter_test_json, twitter_img_path
        datasetclass = TwitterSet

    elif args.dataset == 'Weibo':
        tgt_train_json, tgt_test_json, tgt_img = weibo_train_json, weibo_test_json, weibo_img_path
        datasetclass = WeiboSet

    elif args.dataset == 'Cross':
        if "charliehebdo" in args.source:
            tgt_train_json, tgt_test_json, tgt_img = twitter_train_json, twitter_test_json, twitter_img_path
            datasetclass = TwitterSet
        else:
            tgt_train_json, tgt_test_json, tgt_img = pheme_train_json, pheme_test_json, pheme_img_path
            datasetclass = PhemeSet

    tgt_train_ds = datasetclass(json_path=tgt_train_json, img_path=tgt_img, type=0, events=args.target,
                                visual_type='resnet', stage='train')
    tgt_test_ds = datasetclass(json_path=tgt_test_json, img_path=tgt_img, type=0, events=args.target,
                             visual_type='resnet', stage='test')
    tgt_select_ds = datasetclass(json_path=tgt_train_json, img_path=tgt_img, type=0, events=[],
                              visual_type='resnet', stage='train')


    return tgt_train_ds, tgt_test_ds, tgt_select_ds

