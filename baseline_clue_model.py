import os
import sys
import random
import distutils
from distutils import util
import argparse
import copy
import pprint
from collections import defaultdict
from tqdm import tqdm, trange
import logging
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score

import numpy as np
import torch
import json

from baseline.adapt.models.models import get_model
from baseline.adapt.solvers.solver import get_solver
from baseline.sample import *

from utils.logger import setup_logger, MetricLogger
from utils.eutils import seed_everything, get_merge_source_target_ds, get_dataloader
from model.classify_model import MLPModel_TextCnn


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="result/active/",
                        help="The directory to save the output results.")
    parser.add_argument("--dataset", type=str, default='Pheme', choices=['Pheme', 'Weibo', 'Cross'],
                        help='The dataset name to train')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--source', type=str, default='charliehebdo-sydneysiege-ottawashooting',
                        help='source domain(s)')
    parser.add_argument('--target', type=str, default='ferguson', help='target domain(s)')
    parser.add_argument("--log_name", type=str, default="log.txt")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--tokenizer_type", type=str, default='roberta')
    parser.add_argument("--textcnn_mode", type=str, default="roberta-non")
    parser.add_argument('--freeze_resnet', type=int, default=8)
    parser.add_argument('--d_rop', type=float, default=0.3)

    # Experiment identifiers
    parser.add_argument('--id', type=str, default='debug', help="Experiment identifier")
    parser.add_argument('--al_strat', type=str, default='CLUE', help="Active learning strategy")
    parser.add_argument('--da_strat', type=str, default='mme', help="DA strat. Currently supports: {ft, DANN, MME}")
    parser.add_argument('--model_init', type=str, default='source', help="Active DA model initialization")

    # Load existing configuration?
    parser.add_argument('--load_from_cfg', type=lambda x: bool(distutils.util.strtobool(x)), default=False,
                        help="Load from config?")
    parser.add_argument('--cfg_file', type=str, help="Experiment configuration file",
                        default="config/digits/adaclue.yml")

    # Experimental details
    parser.add_argument('--runs', type=int, default=2, help="Number of experimental runs")
    parser.add_argument('--total_budget', type=int, default=300, help="Total target budget")
    parser.add_argument('--num_rounds', type=int, default=5, help="Target dataset number of splits")

    # Training hyperparameters
    parser.add_argument('--cnn', type=str, default="LeNet", help="CNN architecture")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--wd', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--num_epochs', type=int, default=40, help="Number of Epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--use_cuda', type=lambda x: bool(distutils.util.strtobool(x)), help="Use GPU?")

    # Domain Adaptation hyperparameters
    parser.add_argument('--uda_lr', type=float, default=2e-4, help="Unsupervised (Round 0) DA Learning rate")
    parser.add_argument('--adapt_lr', type=float, default=2e-4, help="SSDA (Round 1 and onwards) Learning rate")
    parser.add_argument('--src_sup_wt', type=float, default=0.1, help="Source supervised loss weight")
    parser.add_argument('--unsup_wt', type=float, default=1.0, help="SSDA unsupervised loss weight")
    parser.add_argument('--cent_wt', type=float, default=0.1, help="SSDA conditional entropy minimization weight")
    parser.add_argument('--adapt_num_epochs', type=int, default=40, help="Semi-supervised DA number of epochs")
    parser.add_argument('--uda_num_epochs', type=int, default=40, help="Unsupervised DA number of epochs")

    # CLUE hyperparameters
    parser.add_argument('--clue_softmax_t', type=float, default=1.0, help="CLUE softmax temperature")


    args = parser.parse_args()

    if args.dataset == 'Weibo':
        args.tokenizer_type = 'chinese'
        args.textcnn_mode = 'chinese-non'

    return args


def utils_train(model, device, train_loader, optimizer, epoch):
	"""
	Test model on provided data for single epoch
	"""
	model.train()
	total_loss, correct = 0.0, 0
	for batch_idx, (text_data, img_data, label) in enumerate(tqdm(train_loader)):
		text_data, img_data, label = text_data.to(device), img_data.to(device), label.to(device)
		optimizer.zero_grad()
		_, _, output, _ = model(text_data, img_data)
		loss = nn.CrossEntropyLoss()(output, label)
		total_loss += loss.item()
		pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
		corr =  pred.eq(label.view_as(pred)).sum().item()
		correct += corr
		loss.backward()
		optimizer.step()

	train_acc = 100. * correct / len(train_loader.sampler)
	avg_loss = total_loss / len(train_loader.sampler)
	print('\nTrain Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
	return avg_loss


def utils_test(model, device, test_loader, split="test"):
    print('\nEvaluating model on {}...'.format(split))
    model.eval()
    test_loss = 0
    correct = 0
    test_acc = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for text_data, img_data, label in test_loader:
            text_data, img_data, label = text_data.to(device), img_data.to(device), label.to(device)
            _, _, output, _ = model(text_data, img_data)
            loss = nn.CrossEntropyLoss()(output, label)
            test_loss += loss.item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            corr =  pred.eq(label.view_as(pred)).sum().item()
            correct += corr

            all_preds.extend(pred.squeeze().cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())

            del loss, output

    test_loss /= len(test_loader.sampler)
    test_acc = correct / len(test_loader.sampler)

    # 将预测和标签转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算F1值
    f1_fake = f1_score(all_labels, all_preds, pos_label=1)  # 标签1 (fake)
    f1_real = f1_score(all_labels, all_preds, pos_label=0)  # 标签0 (real)

    return test_acc, test_loss, f1_fake, f1_real


def utils_run_unsupervised_da(task, src_train_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, args):
    """
    Unsupervised adaptation of source model to target at round 0
    Returns:
        Model post adaptation
    """
    device = torch.device("cuda")
    print('No pretrained checkpoint found, training...')
    source_file = '{}_source.pth'.format(task)
    source_path = os.path.join('checkpoints', 'source', source_file)
    adapt_model = get_model('AdaptNet', num_cls=args.num_classes, textcnn=args.textcnn_mode,
                            dataset=args.dataset, src_weights_init=source_path)
    opt_net_tgt = optim.Adam(adapt_model.tgt_net.parameters(), lr=args.lr, weight_decay=args.wd)
    uda_solver = get_solver(args.da_strat, adapt_model.tgt_net, src_train_loader, tgt_sup_loader, tgt_unsup_loader,
                            train_idx, opt_net_tgt, 0, device, args)
    for epoch in range(args.uda_num_epochs):
        if args.da_strat == 'dann':
            opt_dis_adapt = optim.Adam(discriminator.parameters(), lr=args.uda_lr, betas=(0.9, 0.999),
                                       weight_decay=0)
            uda_solver.solve(epoch, discriminator, opt_dis_adapt)
        elif args.da_strat in ['mme', 'ft']:
            uda_solver.solve(epoch)
    # adapt_model.save(adapt_net_file)

    model, src_model, discriminator = adapt_model.tgt_net, adapt_model.src_net, adapt_model.discriminator
    return model, src_model, discriminator


def run_active_adaptation(args, task, source_model, src_loader, tgt_train_ds, tgt_test_ds):
    """
    Runs active domain adaptation experiments
    """
    device = torch.device("cuda")
    # # Load target data
    # target_dset = ASDADataset(args.target, valid_ratio=0)
    # target_train_dset, _, _ = target_dset.get_dsets(apply_transforms=False)
    # target_train_loader, _, target_test_loader, train_idx = target_dset.get_loaders()
    tgt_train_loader = get_dataloader(args, tgt_train_ds, shuffle=False, drop_last=False)
    tgt_test_loader = get_dataloader(args, tgt_test_ds, shuffle=False, drop_last=False)
    train_idx = np.arange(len(tgt_train_ds))

    # Bookkeeping
    target_accs = defaultdict(list)
    ada_strat = '{}_{}'.format(args.model_init, args.al_strat)

    # Sample varying % of target data
    args.total_budget = len(tgt_train_ds) * 0.1
    sampling_ratio = [(args.total_budget / args.num_rounds) * n for n in range(args.num_rounds + 1)]

    # Evaluate source model on target test
    transfer_perf, _, _, _ = utils_test(source_model, device, tgt_test_loader)

    # Choose appropriate model initialization
    model, src_model = source_model, source_model

    # Run unsupervised DA at round 0, where applicable
    discriminator = None
    if args.da_strat != 'ft':
        print('Round 0: Unsupervised DA to target via {}'.format(args.da_strat))
        model, src_model, discriminator = utils_run_unsupervised_da(task, src_loader, None,
                                                                    tgt_train_loader, train_idx, args)

        # Evaluate adapted source model on target test
        start_perf, _, _, _ = utils_test(model, device, tgt_test_loader)
        out_str = '{}->{} performance (After {}): {:.2f}'.format(args.source, args.target, args.da_strat, start_perf)
        print(out_str)
        print('\n------------------------------------------------------\n')

    #################################################################
    # Main Active DA loop
    #################################################################

    tqdm_run = trange(args.runs)
    for run in tqdm_run:  # Run over multiple experimental runs
        tqdm_run.set_description('Run {}'.format(str(run)))
        tqdm_run.refresh()
        tqdm_rat = trange(len(sampling_ratio[1:]))
        target_accs[0.0].append(start_perf)

        # Making a copy for current run
        curr_model = copy.deepcopy(model)
        curr_source_model = curr_model

        # Keep track of labeled vs unlabeled data
        idxs_lb = np.zeros(len(train_idx), dtype=bool)

        # Instantiate active sampling strategy
        sampling_strategy = get_strategy(args.al_strat, tgt_train_ds, train_idx, \
                                         curr_model, discriminator, device, args)

        for ix in tqdm_rat:  # Iterate over Active DA rounds
            ratio = sampling_ratio[ix + 1]
            tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
            tqdm_rat.refresh()

            # Select instances via AL strategy
            print('\nSelecting instances...')
            idxs = sampling_strategy.query(int(sampling_ratio[1]))
            idxs_lb[idxs] = True
            sampling_strategy.update(idxs_lb)

            # Update model with new data via DA strategy
            best_model = sampling_strategy.train(tgt_train_ds, da_round=(ix + 1), src_loader=src_loader,
                                                 src_model=curr_source_model)

            # Evaluate on target test and train splits
            test_perf, _, f1_fake, f1_real = utils_test(best_model, device, tgt_test_loader)
            train_perf, _, _, _ = utils_test(best_model, device, tgt_train_loader, split='train')

            out_str = '{}->{} Test performance (Round {}, # Target labels={:d}): {:.2f}'.format(args.source,
                                                                                                args.target, ix,
                                                                                                int(ratio), test_perf)
            out_str += '\n\tTrain performance (Round {}, # Target labels={:d}): {:.2f}'.format(ix, int(ratio),
                                                                                               train_perf)
            print('\n------------------------------------------------------\n')
            print(out_str)

            target_accs[ratio].append((test_perf, f1_fake, f1_real))

        # Log at the end of every run
        wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        # target_accs['args'] = wargs
        print(target_accs)
        # utils.log(target_accs, exp_name)

    return target_accs


def train(args, task):
    logger = logging.getLogger("main.trainer")
    device = torch.device("cuda")

    # Load source data
    src_ds, tgt_train_ds, tgt_test_ds, tgt_select = get_merge_source_target_ds(args)

    src_loader = get_dataloader(args, src_ds, shuffle=True, drop_last=True)
    tgt_train_loader = get_dataloader(args, tgt_train_ds, shuffle=True, drop_last=True)
    tgt_candidate_loader = get_dataloader(args, tgt_train_ds, shuffle=False, drop_last=False)
    tgt_test_loader = get_dataloader(args, tgt_test_ds, shuffle=False, drop_last=False)

    # src_dset = ASDADataset(args.source, batch_size=args.batch_size)
    # src_train_loader, src_val_loader, src_test_loader, _ = src_dset.get_loaders()
    # num_classes = src_dset.get_num_classes()


    # Train / load a source model
    source_model = MLPModel_TextCnn(
            out_size=256, num_label=2, freeze_id=args.freeze_resnet,
            d_prob=args.d_rop, kernel_sizes=[3, 4, 5], num_filters=100,
            mode=args.textcnn_mode, dataset_name=args.dataset)
    source_model = source_model.cuda()
    source_file = '{}_source.pth'.format(task)
    source_path = os.path.join('checkpoints', 'source', source_file)
    os.makedirs(os.path.join('checkpoints', 'source'), exist_ok=True)

    if os.path.exists(source_path):  # Load existing source model
        print('Loading source checkpoint: {}'.format(source_path))
        source_model.load_state_dict(torch.load(source_path, map_location=device), strict=False)
        best_source_model = source_model
    else:
        print('Training {} model...'.format(task))
        best_val_acc, best_source_model = 0.0, None
        source_optimizer = optim.Adam(source_model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.num_epochs):
            utils_train(source_model, device, src_loader, source_optimizer, epoch)
            val_acc, _, _, _ = utils_test(source_model, device, tgt_test_loader, split="val")
            out_str = '[Epoch: {}] Val Accuracy: {:.3f} '.format(epoch, val_acc)
            print(out_str)

            if (val_acc > best_val_acc):
                best_val_acc = val_acc
                best_source_model = copy.deepcopy(source_model)
                torch.save(best_source_model.state_dict(), os.path.join('checkpoints', 'source', source_file))

    # Evaluate on source test set
    test_acc, _, _, _ = utils_test(best_source_model, device, tgt_test_loader, split="test")
    out_str = '{} Test Accuracy: {:.3f} '.format(args.source, test_acc)
    print(out_str)

    # Run active adaptation experiments
    target_accs = run_active_adaptation(args, task, best_source_model, src_loader, tgt_train_ds, tgt_test_ds)
    # pp.pprint(target_accs)
    print(target_accs)
    return task, target_accs


def main():
    args = get_parser()

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("main", output_dir, 0, filename=args.log_name)
    seed_everything(42)

    torch.multiprocessing.set_sharing_strategy('file_system')


    # combine all source train files into one path file
    # generate file such as "Art_Clipart_Product_train.txt"

    if args.dataset == "Pheme":
        domains = ["charliehebdo", "sydneysiege", "ottawashooting", "ferguson"]
        setting = ["sof2c", "cof2s", "csf2o", "cso2f"]
    elif args.dataset == 'Cross':
        domains = ["malaysia",  "sandy", "sydneysiege", "ottawashooting"]
        source_list = [["charliehebdo","ottawashooting", "ferguson"], ["charliehebdo","ottawashooting", "ferguson"], ["sandy", "boston", "sochi"], ["sandy", "boston", "sochi"]]
        setting = ["cof2m", "cof2a", "abi2s", "abi2o"]
    elif args.dataset == 'Weibo':
        domains = ["society", "entertainment", "education", "health"]
        setting = ["edh2s", "sdh2e", "seh2d", "sed2h"]
    else:
        domains = ["sandy", "boston", "malaysia", "sochi"]
        setting = ["bmi2a", "ami2b", "abi2m", "abm2i"]

    all_task_result = dict()
    for i, target_domain in enumerate(domains):

        index = [0, 1, 2, 3]
        index.remove(i)
        args.target = target_domain
        if args.dataset == 'Cross':
            args.source = source_list[i]
        else:
            args.source = [domains[j] for j in index]
        print("source={}, target={}".format(args.source, args.target))

        task, target_accs = train(args, task=setting[i])
        all_task_result[task] = target_accs
        # all_task_result.append({'task': task, 'final_acc': final_acc, 'best_acc': best_acc})
        # print(all_task_result)
        # logger.info('task: {} final_acc: {:.2f} best_acc: {:.2f} '.format(task, final_acc, best_acc))


    # record all results for all tasks
    with open(os.path.join(output_dir, 'all_task_result.json'), 'w') as f:
        json.dump(all_task_result, f, indent=4)
        # for i, rec in enumerate(all_task_result):
        #     if i == 0:
        #         f.write(','.join(list(rec.keys())) + '\n')
        #     line = [str(rec[key]) for key in rec.keys()]
        #     f.write(','.join(line) + '\n')


if __name__ == '__main__':
    main()