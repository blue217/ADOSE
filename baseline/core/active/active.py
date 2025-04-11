import random
import math
import numpy as np
import torch


def RAND_active(tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    length = len(tgt_unlabeled_ds.samples)
    index = random.sample(range(length), round(totality * active_ratio))

    active_samples = tgt_unlabeled_ds.samples[index]

    tgt_selected_ds.add_item(active_samples)
    tgt_unlabeled_ds.remove_item(index)

    return active_samples


def EADA_active(tgt_unlabeled_loader_full, tgt_unlabeled_ds, active_ratio, totality, model, args):
    model.eval()
    first_stat = list()
    with torch.no_grad():
        for i, data in enumerate(tgt_unlabeled_loader_full):
            tgt_text, tgt_img, tgt_lbl = data[0], data[1], data[2]
            tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()

            tgt_out = model(tgt_text, tgt_img)

            # MvSM of each sample
            # minimal energy - second minimal energy
            min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
            mvsm_uncertainty = min2[:, 0] - min2[:, 1]

            # free energy of each sample
            output_div_t = -1.0 * tgt_out / args.energy_beta
            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
            free_energy = -1.0 * args.energy_beta * output_logsumexp

            # for i in range(len(free_energy)):
            #     first_stat.append([tgt_path[i], tgt_lbl[i].item(), tgt_index[i].item(),
            #                        mvsm_uncertainty[i].item(), free_energy[i].item()])

            for j in range(len(free_energy)):
                sample_id = i * args.batch_size + j
                first_stat.append([sample_id, tgt_lbl[j].item(), mvsm_uncertainty[j].item(),
                                   free_energy[j].item()])

    first_sample_ratio = args.first_sample_ratio
    first_sample_num = math.ceil(totality * first_sample_ratio)
    second_sample_ratio = active_ratio / args.first_sample_ratio
    second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

    # the first sample using \mathca{F}, higher value, higher consideration
    first_stat = sorted(first_stat, key=lambda x: x[3], reverse=True)   # free_energy
    second_stat = first_stat[:first_sample_num]

    # the second sample using \mathca{U}, higher value, higher consideration
    second_stat = sorted(second_stat, key=lambda x: x[2], reverse=True)
    second_stat = second_stat[:second_sample_num]

    candidate_ds_index = [item[0] for item in second_stat]

    candidate_ds_index.sort()
    # print(candidate_ds_index)
    active_samples = tgt_unlabeled_ds.remove_item(candidate_ds_index)

    return active_samples
