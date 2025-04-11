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
from utils.logger import setup_logger, MetricLogger
from utils.eutils import seed_everything, get_merge_source_target_ds, get_dataloader
from model.classify_model import HyperModel_TextCnn, MLPModel_TextCnn, AttenModel_TextCnn
from model.active import EDL_Loss, active_select


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
    parser.add_argument("--input_dir", type=str, default="result/active/")
    parser.add_argument("--dataset", type=str, default='Pheme', choices=['Pheme', 'Twitter', 'Cross'],
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
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay (default: 5e-4)')
    parser.add_argument('--active_round', type=int, nargs='+', default=[2, 4, 6, 8, 10],
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
    parser.add_argument('--active_ratio', type=float, default=0.1,
                        help='The ratio of labeled samples selected from target domain in each round(5 rounds).')
    parser.add_argument("--active_type", type=str,
                        help="strategy for selecting samples to annotate")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'adapt'])

    args = parser.parse_args()

    return args


def test(model, test_loader):
    start_test = True
    model.eval()
    with torch.no_grad():
        for batch_idx, test_data in enumerate(test_loader):
            text, img, label = test_data[0], test_data[1], test_data[2]
            text, img = text.cuda(), img.cuda()
            logits = model(text, img)

            alpha = torch.exp(logits)
            total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
            outputs = alpha / total_alpha

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
    src_ds, tgt_train_ds, tgt_test_ds, tgt_select = get_merge_source_target_ds(args)
    src_loader = get_dataloader(args, src_ds, shuffle=True, drop_last=True)
    tgt_train_loader = get_dataloader(args, tgt_train_ds, shuffle=True, drop_last=True)
    tgt_candidate_loader = get_dataloader(args, tgt_train_ds, shuffle=False, drop_last=False)
    tgt_test_loader = get_dataloader(args, tgt_test_ds, shuffle=False, drop_last=False)
    # tgt_test_loader = get_dataloader(args, tgt_test_ds, shuffle=False, drop_last=False)
    # tgt_select_loader = get_dataloader(args, tgt_select, shuffle=True, drop_last=False)

    model = AttenModel_TextCnn(
            out_size=256, num_label=2, freeze_id=args.freeze_resnet,
            d_prob=args.d_rop, kernel_sizes=[3, 4, 5], num_filters=100,
            mode=args.textcnn_mode, dataset_name=args.dataset)
    model.cuda()

    iter_per_epoch = max(len(src_loader), len(tgt_train_loader))
    optimizer = optim.SGD(model.get_param(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_per_epoch)

    # evidence deep learning loss function
    edl_criterion = EDL_Loss()

    # total number of target samples
    totality = len(tgt_train_ds)
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_training_time = time.time()
    end = time.time()

    final_model = None
    lr_history = []
    ckt_path = os.path.join(args.output_dir, args.dataset, task)
    os.makedirs(ckt_path, exist_ok=True)

    if args.phase == 'adapt':
        args.train_epochs = 0
        model_path = os.path.join(args.input_dir, args.dataset, task)
        checkpoint = torch.load(os.path.join(model_path, "final_model.pth"), map_location='cuda')
        model.load_state_dict(checkpoint)
        optimizer = optim.SGD(model.get_param(args.lr), lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=True)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, iter_per_epoch)


    best_acc = 0.0
    all_train_epoch_result = []
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        for batch_idx in range(iter_per_epoch):
            data_time = time.time() - end

            if batch_idx % len(src_loader) == 0:
                src_iter = iter(src_loader)
            if batch_idx % len(tgt_train_loader) == 0:
                tgt_train_iter = iter(tgt_train_loader)

            src_text, src_img, src_lbl = next(src_iter)
            src_text, src_img, src_lbl = src_text.cuda(), src_img.cuda(), src_lbl.cuda()
            tgt_text, tgt_img, tgt_lbl = next(tgt_train_iter)
            tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()

            optimizer.zero_grad()
            total_loss = 0

            # evidence deep learning loss on labeled source data
            src_out = model(src_text, src_img)
            Loss_nll_s, Loss_KL_s = edl_criterion(src_out, src_lbl)
            Loss_KL_s = Loss_KL_s / args.num_labels

            total_loss += Loss_nll_s
            meters.update(Loss_nll_s=Loss_nll_s.item())

            total_loss += Loss_KL_s
            meters.update(Loss_KL_s=Loss_KL_s.item())

            # if nan occurs, stop training
            if torch.isnan(total_loss):
                logger.info("total_loss is nan, stop training")
                # return task, final_train_acc, best_train_acc
                return task, best_acc, best_acc

            if args.train_beta > 0:
                # uncertainty reduction loss on unlabeled target data
                tgt_out = model(tgt_text, tgt_img)
                alpha_t = torch.exp(tgt_out)
                total_alpha_t = torch.sum(alpha_t, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p_t = alpha_t / total_alpha_t
                eps = 1e-7
                point_entropy_t = - torch.sum(expected_p_t * torch.log(expected_p_t + eps), dim=1)
                data_uncertainty_t = torch.sum(
                    (alpha_t / total_alpha_t) * (torch.digamma(total_alpha_t + 1) - torch.digamma(alpha_t + 1)), dim=1)
                loss_Udis = torch.sum(point_entropy_t - data_uncertainty_t) / tgt_out.shape[0]
                loss_Udata = torch.sum(data_uncertainty_t) / tgt_out.shape[0]

                total_loss += args.train_beta * loss_Udis
                meters.update(loss_Udis=(loss_Udis).item())
                total_loss += args.train_lambda * loss_Udata
                meters.update(loss_Udata=(loss_Udata).item())


            total_loss.backward()

            # clip grad norm if necessary
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, norm_type=2)

            optimizer.step()
            # update lr
            if lr_scheduler is not None:
                lr_scheduler.step()
                lr_history.append(lr_scheduler.get_lr()[-1])
            else:
                lr_history.append(optimizer.param_groups[0]['lr'])

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            eta_seconds = meters.time.global_avg * (iter_per_epoch * args.train_epochs - batch_idx * epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if batch_idx % args.print_freq == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "task: {task}",
                            "epoch: {epoch}",
                            f"[iter: {batch_idx}/{iter_per_epoch}]",
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
            all_train_epoch_result.append({'train_epoch': epoch, 'acc': testacc})

            if testacc > best_acc:
                best_acc = testacc
                # best_model = model.state_dict()
                # torch.save(best_model, os.path.join(ckt_path, "best_model.pth"))

        if epoch == args.train_epochs:
            final_model = model.state_dict()
            torch.save(final_model, os.path.join(ckt_path, "final_model.pth"))
            result_to_file(ckt_path, 'all_train_epoch_result.csv', all_train_epoch_result)
            return task, testacc, best_acc



    # active select samples
    active_samples = active_select(tgt_candidate_loader=tgt_candidate_loader,
                                   tgt_dataset=tgt_train_ds,
                                   active_ratio=args.active_ratio,
                                   totality=totality,
                                   model=model,
                                   t_step=1,
                                   active_type=args.active_type)

    tgt_select.add_item(active_samples)
    tgt_select_loader = get_dataloader(args, tgt_select, shuffle=True, drop_last=False)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(tgt_select_loader))
    logger.info('Task: {} Active select samples num: {}, tgt_select num:{}'.format(task, len(active_samples), len(src_ds)))
    logger.info('Active update: tgt_select_loader len: {}'.format(len(src_loader)))

    # evidence deep learning loss on selected target data
    final_acc = 0.
    best_acc = 0.
    all_adapt_epoch_result = []
    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        for batch in tgt_select_loader:
            optimizer.zero_grad()
            total_loss = 0

            tgt_select_text, tgt_select_img, tgt_select_lbl = batch[0], batch[1], batch[2]
            tgt_select_text = tgt_select_text.cuda()
            tgt_select_img = tgt_select_img.cuda()
            tgt_select_lbl = tgt_select_lbl.cuda()

            tgt_select_out = model(tgt_select_text, tgt_select_img)
            selected_Loss_nll_t, selected_Loss_KL_t = edl_criterion(tgt_select_out, tgt_select_lbl)
            selected_Loss_KL_t = selected_Loss_KL_t / args.num_labels
            total_loss += selected_Loss_nll_t
            meters.update(selected_Loss_nll_t=selected_Loss_nll_t.item())
            total_loss += selected_Loss_KL_t
            meters.update(selected_Loss_KL_t=selected_Loss_KL_t.item())

            total_loss.backward()

            # clip grad norm if necessary
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, norm_type=2)

            optimizer.step()
            lr_scheduler.step()


        if epoch % args.test_freq == 0:
            testacc = test(model, tgt_test_loader)
            logger.info('Task: {} Adapt Epoch: {} testacc: {:.2f}'.format(task, epoch, testacc))
            all_adapt_epoch_result.append({'adapt_epoch': epoch, 'acc': testacc})

            if epoch == args.adapt_epochs:
                final_acc = testacc
            if testacc > best_acc:
                best_acc = testacc


    # record results for test epochs
    result_file_name = 'all_adapt_epoch_result.csv'
    result_to_file(ckt_path, result_file_name, all_adapt_epoch_result)


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
        epoch_ticks = np.arange(len(lr_history)) / iter_per_epoch
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

    # combine all source train files into one path file
    # generate file such as "Art_Clipart_Product_train.txt"

    if args.dataset == "Pheme":
        domains = ["charliehebdo", "sydneysiege", "ottawashooting", "ferguson"]
        setting = ["sof2c", "cof2s", "csf2o", "cso2f"]
    elif args.dataset == 'Cross':
        domains = ["malaysia",  "sandy", "sydneysiege", "ottawashooting"]
        source_list = [["charliehebdo","ottawashooting", "ferguson"], ["charliehebdo","ottawashooting", "ferguson"], ["sandy", "boston", "sochi"], ["sandy", "boston", "sochi"]]
        setting = ["cof2m", "cof2a", "abi2s", "abi2o"]
    else:
        domains = ["sandy", "boston", "malaysia", "sochi"]
        setting = ["bmi2a", "ami2b", "abi2m", "abm2i"]

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


















