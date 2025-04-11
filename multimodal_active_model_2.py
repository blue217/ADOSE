import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import logging
import torch.optim as optim
import time
import torch.nn as nn
import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
from utils.logger import setup_logger, MetricLogger
from utils.eutils import seed_everything, get_merge_source_target_ds, get_dataloader, get_three_source_iter, get_target_ds
from model.classify_model import JointModel_TextCnn, DomainDiscriminator, grad_reverse
from model.active import EDL_Loss, active_select
from model.jmmd import JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from model.contrastive_loss import compute_nego_loss, compute_domain_adversarial_loss, Intro_alignment_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau


def result_to_file(file_dir, file_name, result_list):
    # record results for test epochs
    best_acc = 0.0
    best_epoch = 0

    with open(os.path.join(file_dir, file_name), 'w') as handle:
        for i, rec in enumerate(result_list):
            keys_list = list(rec.keys())
            if rec[keys_list[1]] > best_acc:
                best_acc = rec[keys_list[1]]
                best_epoch = rec[keys_list[0]]
            if i == 0:
                handle.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            handle.write(','.join(line) + '\n')
        handle.write(','.join(['best epoch', 'best acc']) + '\n')
        line = [str(best_epoch), str(best_acc)]
        handle.write(','.join(line) + '\n')


def get_parser():
    parser = argparse.ArgumentParser(description='Active Multimodal DA')
    parser.add_argument("--output_dir", type=str, default="result/active/",
                        help="The directory to save the output results.")
    parser.add_argument("--dataset", type=str, default='Pheme', choices=['Pheme', 'Twitter', 'Cross', 'Weibo'],
                        help='The dataset name to train')
    parser.add_argument("--log_name", type=str, default="log.txt",
                        help="The log file to record output.")
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing training.')
    parser.add_argument('--source', type=str, default='charliehebdo-sydneysiege-ottawashooting',
                        help='source domain(s)')
    parser.add_argument('--target', type=str, default='ferguson', help='target domain(s)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='The number of workers for Dataloader.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 32)')
    parser.add_argument("--tokenizer_type", type=str, default='roberta',
                        help="the tokenizer of text consisting of bert, roberta, spacy")
    parser.add_argument('--freeze_resnet', type=int, default=8,
                        help='free parameters of resnet')
    parser.add_argument('--d_rop', type=float, default=0.3,
                        help='d_rop rate of neural network model')
    parser.add_argument("--textcnn_mode", type=str, default="roberta-non",
                        choices=["rand", "roberta-yes", "roberta-non", "bert-yes", "bert-non"],
                        help="The embedding mode of textcnn")
    parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--active_round', type=int, nargs='+', default=[6, 8, 10, 12, 14],
                        help='trainer active round')
    parser.add_argument('--z_dim', type=int, default=256, help='network dim')
    parser.add_argument('--train_epochs', type=int, default=20, help='number of total epochs to train')
    parser.add_argument('--adapt_epochs', type=int, default=20, help='number of total epochs to adapt')
    parser.add_argument('--num_labels', type=int, default=2, help='The number of labels of classifier')
    parser.add_argument('--train_beta', type=float, default=1.0, help='')
    parser.add_argument('--train_lambda', type=float, default=0.05, help='')
    parser.add_argument('--clip_grad_norm', type=float, default=15, help='')
    parser.add_argument('--print_freq', type=int, default=10, help='')
    parser.add_argument('--test_freq', type=int, default=1, help='')
    parser.add_argument("--save_model", action="store_true", help="whether to save best model.")
    parser.add_argument('--active_ratio', type=float, default=0.02,
                        help='The ratio of labeled samples selected from target domain in each round(5 rounds).')
    parser.add_argument("--active_type", type=str, default='random',
                        choices=['detective', 'ldms', 'random', 'entropy', 'margin', 'ldms2', 'ldms3'],
                        help="strategy for selecting samples to annotate")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'adapt'])
    parser.add_argument('--tsigma', default="2#4#8#16", type=str,
                        help='the sigma of Gaussian Kernel for textual feature')
    parser.add_argument('--vsigma', default="2#4#8#16", type=str,
                        help='the sigma of Gaussian Kernel for visual feature')
    parser.add_argument('--linear', default='false', type=str,
                        help='whether use the linear version of MMD')
    parser.add_argument('--intratheta', default='true', type=str, help='whether use the mlp for intra loss')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='the temperature for contrastive loss')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='the threshold for contrastive loss')
    parser.add_argument('--cls_dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--ctsize', default=64, type=int,
                        help='the size for contrastive learning')
    parser.add_argument('--lambda1', default=0.1, type=float,
                        help='the trade-off hyper-parameter for inter-domain transfer loss. 0 means non-transfer')
    parser.add_argument('--lambda2', default=0.5, type=float,
                        help='the trade-off hyper-parameter for intra-domain transfer loss')
    parser.add_argument('--patience', default=5, type=int, metavar='M',
                        help='patience')
    parser.add_argument('--max_iter', default=20, type=int,
                        help='the maximum number of iteration in each epoch ')
    parser.add_argument('--loss_type', default=1, type=int,
                        help='the maximum number of iteration in each epoch ')

    args = parser.parse_args()

    args.tsigma = [float(i) for i in str(args.tsigma).strip().split('#')]
    args.vsigma = [float(i) for i in str(args.vsigma).strip().split('#')]

    if args.dataset == 'Weibo':
        args.tokenizer_type = 'chinese'
        args.textcnn_mode = 'chinese-non'

    return args


def test(model, test_loader):
    start_test = True
    model.eval()
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            text, img, label = test_data[0], test_data[1], test_data[2]
            text, img = text.cuda(), img.cuda()
            logits, _ = model(text, img)
            logits = logits[0]

            # alpha = torch.exp(logits)
            # total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
            # outputs = alpha / total_alpha
            outputs = F.softmax(logits, dim=1)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = label.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)
    _, predict = torch.max(all_output, dim=1)
    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    return acc


def train(args, task):

    logger = logging.getLogger("main.trainer")

    # load dataset and model
    multi_source_loader, multi_source_iter = get_three_source_iter(args)
    tgt_train_ds, tgt_test_ds, tgt_select_ds = get_target_ds(args)
    _, tgt_train_iter = get_dataloader(args, tgt_train_ds, shuffle=True, drop_last=True, return_iter=True)
    tgt_candidate_loader = get_dataloader(args, tgt_train_ds, shuffle=False, drop_last=False)  # ! ! !
    tgt_test_loader = get_dataloader(args, tgt_test_ds, shuffle=False, drop_last=False)

    model = JointModel_TextCnn(out_size=256, num_label=2, freeze_id=args.freeze_resnet,
                               d_prob=args.d_rop, kernel_sizes=[3, 4, 5], num_filters=100,
                               mode=args.textcnn_mode, dataset_name=args.dataset)
    intra_loss = Intro_alignment_loss(theta=args.intratheta, temperature=args.temperature, threshold=args.threshold,
                                      input_dim=args.cls_dim, output_dim=args.cls_dim)
    domain_discriminator_t = DomainDiscriminator(in_size=256, hidden_size=128)
    domain_discriminator_v = DomainDiscriminator(in_size=256, hidden_size=128)

    model.cuda()
    intra_loss.cuda()
    domain_discriminator_t.cuda()
    domain_discriminator_v.cuda()

    parameters = model.get_param()
    for para in parameters:
        para["lr"] = args.lr
    parameters += [{"params": intra_loss.parameters(), 'lr': args.lr}]
    parameters += [{"params": domain_discriminator_t.parameters(), 'lr': args.lr}]
    parameters += [{"params": domain_discriminator_v.parameters(), 'lr': args.lr}]


    optimizer = optim.Adam(params=parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.wd, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience, verbose=True)


    # total number of target samples
    totality = len(tgt_train_ds)
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_training_time = time.time()
    end = time.time()

    final_model = None
    lr_history = []
    best_acc = 0.0
    final_acc = 0.
    all_epoch_result = []
    ckt_path = os.path.join(args.output_dir, args.dataset, task)
    os.makedirs(ckt_path, exist_ok=True)
    active_round = 1

    for epoch in range(1, args.train_epochs + 1):
        model.train()
        intra_loss.train()
        domain_discriminator_t.train()
        domain_discriminator_v.train()

        for batch_idx in range(args.max_iter):
            data_time = time.time() - end

            texts_s1, imgs_s1, labels_s1 = next(multi_source_iter[0])
            texts_s2, imgs_s2, labels_s2 = next(multi_source_iter[1])
            texts_s3, imgs_s3, labels_s3 = next(multi_source_iter[2])
            texts_t, imgs_t, _ = next(tgt_train_iter)

            texts_s1, imgs_s1, labels_s1 = texts_s1.cuda(), imgs_s1.cuda(), labels_s1.cuda()
            texts_s2, imgs_s2, labels_s2 = texts_s2.cuda(), imgs_s2.cuda(), labels_s2.cuda()
            texts_s3, imgs_s3, labels_s3 = texts_s3.cuda(), imgs_s3.cuda(), labels_s3.cuda()
            texts_t, imgs_t = texts_t.cuda(), imgs_t.cuda()

            optimizer.zero_grad()

            logits_s1, feats_s1 = model(train_texts=texts_s1, train_imgs=imgs_s1)
            logits_s2, feats_s2 = model(train_texts=texts_s2, train_imgs=imgs_s2)
            logits_s3, feats_s3 = model(train_texts=texts_s3, train_imgs=imgs_s3)
            _, feats_t = model(train_texts=texts_t, train_imgs=imgs_t)

            y_final = torch.cat([logits_s1[0], logits_s2[0], logits_s3[0]], dim=0)
            y_text = torch.cat([logits_s1[1], logits_s2[1], logits_s3[1]], dim=0)
            y_img = torch.cat([logits_s1[2], logits_s2[2], logits_s3[2]], dim=0)
            y_fuse = torch.cat([logits_s1[3], logits_s2[3], logits_s3[3]], dim=0)

            labels = torch.cat([labels_s1, labels_s2, labels_s3], dim=0)

            feats_text = torch.cat([feats_s1[0], feats_s2[0], feats_s3[0]], dim=0)
            feats_img = torch.cat([feats_s1[1], feats_s2[1], feats_s3[1]], dim=0)
            feats_instance = torch.cat([feats_s1[2], feats_s2[2], feats_s3[2]], dim=0)

            if len(tgt_select_ds) > 0:
                texts_tl, imgs_tl, label_tl = next(tgt_select_iter)
                texts_tl, imgs_tl, label_tl = texts_tl.cuda(), imgs_tl.cuda(), label_tl.cuda()
                logits_tl, feats_tl = model(train_texts=texts_tl, train_imgs=imgs_tl)

                y_final = torch.cat([y_final, logits_tl[0]], dim=0)
                y_text = torch.cat([y_text, logits_tl[1]], dim=0)
                y_img = torch.cat([y_img, logits_tl[2]], dim=0)
                y_fuse = torch.cat([y_fuse, logits_tl[3]], dim=0)

                labels = torch.cat([labels, label_tl], dim=0)

                feats_text = torch.cat([feats_text, feats_tl[0]], dim=0)
                feats_img = torch.cat([feats_img, feats_tl[1]], dim=0)
                feats_instance = torch.cat([feats_instance, feats_tl[2]], dim=0)


            final_ce_loss = F.cross_entropy(y_final, labels)
            # separate_ce_loss = F.cross_entropy(y_text, labels) + F.cross_entropy(y_img, labels) + F.cross_entropy(y_fuse, labels)
            nego_loss = compute_nego_loss(y_text, y_img, y_fuse, labels)

            # modal alignment loss: all source {texts, imgs, labels, instances}
            contrast_loss, _ = intra_loss(feats_text, feats_img, labels, feats_instance)
            adv_text_loss = compute_domain_adversarial_loss(domain_discriminator_t, feats_s1[0], feats_s2[0],
                                                            feats_s3[0], feats_t[0])
            adv_img_loss = compute_domain_adversarial_loss(domain_discriminator_v, feats_s1[1], feats_s2[1],
                                                           feats_s3[1], feats_t[1])
            adv_loss = adv_text_loss + adv_img_loss
            meters.update(Loss_adversarial=adv_loss.item()) # modal loss : all source {texts, imgs, labels, instances}
            meters.update(Loss_nego=nego_loss.item())
            meters.update(Loss_classifier=final_ce_loss.item())

            total_loss = final_ce_loss + 0.2 * nego_loss + 0.2 * adv_loss + 0.5 * contrast_loss
            # if args.loss_type == 1:
            #     total_loss = final_ce_loss + 0.5 * contrast_loss
            # elif args.loss_type == 2:
            #     total_loss = final_ce_loss + 0.5 * contrast_loss + 0.2 * nego_loss
            # elif args.loss_type == 3:
            #     total_loss = final_ce_loss + 0.5 * contrast_loss + 0.2 * adv_loss
            # elif args.loss_type == 4:
            #     total_loss = final_ce_loss + 0.5 * nego_loss + 0.2 * adv_loss + 0.2 * contrast_loss
            total_loss.backward()
            optimizer.step()
            meters.update(Total_Loss=total_loss.item())

            # if nan occurs, stop training
            if torch.isnan(total_loss):
                logger.info("total_loss is nan, stop training")
                # return task, final_train_acc, best_train_acc
                return task, best_acc, best_acc


            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (args.max_iter * args.train_epochs - batch_idx * epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if batch_idx % args.print_freq == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "task: {task}",
                            "epoch: {epoch}",
                            f"[iter: {batch_idx}/{args.max_iter}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        task=task,
                        eta=eta_string,
                        epoch=epoch,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )

        if epoch % args.test_freq == 0:
            testacc = test(model, tgt_test_loader)
            logger.info('Task: {} Train Epoch: {} testacc: {:.2f}'.format(task, epoch, testacc))
            all_epoch_result.append({'train_epoch': epoch, 'acc': testacc})
            if testacc > best_acc:
                best_acc = testacc
            final_acc = testacc
            lr_scheduler.step(float(testacc))


        # active selection rounds
        if epoch in args.active_round:
            active_samples = active_select(tgt_candidate_loader=tgt_candidate_loader,
                                           tgt_dataset=tgt_train_ds,
                                           active_ratio=args.active_ratio,
                                           totality=totality,
                                           model=model,
                                           t_step=active_round,
                                           active_type=args.active_type)

            tgt_select_ds.add_item(active_samples)
            if active_round == 1:
                _, tgt_select_iter = get_dataloader(args, tgt_select_ds, shuffle=True, drop_last=False, return_iter=True)
            active_round += 1
            logger.info('Task: {} Active Epoch: {}, samples num: {}, tgt_select_ds len:{}'.format(
                    task, epoch, len(active_samples), len(tgt_select_ds)))


    result_to_file(ckt_path, 'all_epoch_result.csv', all_epoch_result)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / ep)".format(
            total_time_str, total_training_time / args.adapt_epochs
        )
    )
    if lr_history:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = f"learning_rate_schedule_{current_time}.png"
        epoch_ticks = np.arange(len(lr_history)) / args.max_iter
        plt.figure()
        plt.plot(epoch_ticks, lr_history, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.savefig(os.path.join(ckt_path, filename))
        plt.close()

    return task, final_acc, best_acc



def main():
    args = get_parser()

    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("main", output_dir, 0, filename=args.log_name)
    seed_everything(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')


    if args.dataset == "Pheme":
        domains = ["charliehebdo", "sydneysiege", "ottawashooting", "ferguson"]
        setting = ["sof2c", "cof2s", "csf2o", "cso2f"]
    elif args.dataset == 'Cross':
        domains = ["malaysia",  "sandy", "sydneysiege", "ottawashooting"]
        source_list = [["charliehebdo","ottawashooting", "ferguson"], ["charliehebdo","ottawashooting", "ferguson"], ["sandy", "boston", "sochi"], ["sandy", "boston", "sochi"]]
        setting = ["cof2m", "cof2a", "abi2s", "abi2o"]
    elif args.dataset == 'Twitter':
        domains = ["sandy", "boston", "malaysia", "sochi"]
        setting = ["bmi2a", "ami2b", "abi2m", "abm2i"]
    elif args.dataset == 'Weibo':
        domains = ["society", "entertainment", "education", "health"]
        setting = ["edh2s", "sdh2e", "seh2d", "sed2h"]

    all_task_result = []
    for i, target_domain in enumerate(domains):

        index = [0, 1, 2, 3]
        index.remove(i)
        args.target = target_domain
        if args.dataset == 'Cross':
            args.source = source_list[i]
        else:
            args.source = [domains[j] for j in index]
        print("source={}, target={}".format(args.source, args.target))

        task, final_acc, best_acc = train(args, task=setting[i])
        all_task_result.append({'task': task, 'final_acc': final_acc, 'best_acc': best_acc})
        print(all_task_result)
        logger.info('task: {} final_acc: {:.2f} best_acc: {:.2f} '.format(task, final_acc, best_acc))


    # record all results for all tasks
    with open(os.path.join(output_dir, 'all_task_result.csv'), 'w') as f:
        for i, rec in enumerate(all_task_result):
            if i == 0:
                f.write(','.join(list(rec.keys())) + '\n')
            line = [str(rec[key]) for key in rec.keys()]
            f.write(','.join(line) + '\n')



if __name__ == '__main__':
    main()


















