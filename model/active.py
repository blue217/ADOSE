import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_distances
from scipy import stats
from tqdm import tqdm

class EDL_Loss(nn.Module):
    """
    evidence deep learning loss
    """
    def __init__(self):
        super(EDL_Loss, self).__init__()

    def forward(self, logits, labels=None):
        alpha = torch.exp(logits)
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)
        if labels is None:
            labels = torch.max(alpha, dim=1)[1]

        one_hot_y = torch.eye(logits.shape[1]).cuda()
        one_hot_y = one_hot_y[labels]
        one_hot_y.requires_grad = False

        loss_nll = torch.sum(one_hot_y * (total_alpha.log() - alpha.log())) / logits.shape[0]
        uniform_bata = torch.ones((1, logits.shape[1])).cuda()
        uniform_bata.requires_grad = False
        total_uniform_beta = torch.sum(uniform_bata, dim=1)
        new_alpha = one_hot_y + (1.0 - one_hot_y) * alpha
        new_total_alpha = torch.sum(new_alpha, dim=1)
        loss_KL = torch.sum(
            torch.lgamma(new_total_alpha) - torch.lgamma(total_uniform_beta) - torch.sum(torch.lgamma(new_alpha), dim=1) \
            + torch.sum((new_alpha - 1) * (torch.digamma(new_alpha) - torch.digamma(new_total_alpha.unsqueeze(1))), dim=1)
        ) / logits.shape[0]

        return loss_nll, loss_KL


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        # with _disable_tracking_bn_stats(model):
        # calc adversarial direction
        for _ in range(self.ip):
            d.requires_grad_()
            pred_hat = model(x + self.xi * d)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
            adv_distance.backward()
            d = _l2_normalize(d.grad)
            # model.zero_grad()

        # calc LDS
        r_adv = d * self.eps
        pred_hat = model(x + r_adv)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        lds = F.kl_div(logp_hat, pred, reduction='batchmean')
        lds_each = F.kl_div(logp_hat, pred, reduction='none')
        lds_each = torch.sum(lds_each ,1 )

        return lds ,lds_each


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def compute_density(features, uncertainties=None, max_increase=0.1):
    normalized_uncertainty = sigmoid(uncertainties - uncertainties.mean())
    densities = []
    for i in range(features.shape[0]):
        weight = 1 + normalized_uncertainty[i] * max_increase
        distance = torch.norm(features[i] - features, dim=1) * weight
        density = 1.0 / (torch.sum(distance, dim=0).item() + 1e-7)
        densities.append(density)

    return densities


def vus(model, features, select_num):
    vat_loss = VATLoss()
    lds, lds_each = vat_loss(model, features)
    lds_each = lds_each.view(-1)
    _, querry_indices = torch.topk(lds_each, select_num)
    querry_indices = querry_indices.cpu()
    return querry_indices


def bus(model, features, select_num):

    with torch.no_grad():
        # Normalize features if needed (e.g., for cosine similarity)
        # features_normalized = F.normalize(features, p=2, dim=1)
        logits = model(features)  # Shape: (N, num_classes)
        probs = F.softmax(logits, dim=1)  # Shape: (N, num_classes)

    # entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)  # Shape: (N,)
    # _, querry_indices = torch.topk(entropy, select_num)
    # querry_indices = querry_indices.cpu()

    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)  # Shape: (N,)
    num_classes = probs.shape[1]  # Number of classes
    samples_per_class = [select_num // 2 for _ in range(2)]  # Number of samples to select per class

    for c in range(num_classes):
        # Get the indices of samples with the highest entropy for class c
        class_indices = torch.where(torch.argmax(probs, dim=1) == c)[0]
        class_entropy = entropy[class_indices]
        # Select top samples with highest entropy in class c
        if class_entropy.shape[0] < samples_per_class[c]:
            samples_per_class[c] = class_entropy.shape[0]
            samples_per_class[1-c] = select_num - samples_per_class[c]


    querry_indices = []
    for c in range(num_classes):
        # Get the indices of samples with the highest entropy for class c
        class_indices = torch.where(torch.argmax(probs, dim=1) == c)[0]
        class_entropy = entropy[class_indices]

        # Select top samples with highest entropy in class c
        _, top_indices = torch.topk(class_entropy, samples_per_class[c], largest=True)
        querry_indices.extend(class_indices[top_indices].cpu())

    return querry_indices


def get_feature(model, X, n_part=1):
    idx_layer = [i for i, layer in enumerate(model.model) if hasattr(layer, 'weight')][-1]
    m = X.shape[0]
    n = m // n_part

    features = []
    def hook(module, input, output):
        features.append(output.detach())

    target_layer = model.model[idx_layer - 1]
    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        for j in range(int(np.ceil(m / n))):
            start_idx = j * n
            end_idx = min((j + 1) * n, m)
            batch_X = X[start_idx:end_idx]
            _ = model(batch_X)

    handle.remove()
    feature = torch.cat(features, dim=0)

    return feature


def get_ldm_gpu(model, feature, stop_cond=10, n_part=1, device='cuda'):  # 移除 X_S 参数
    m = feature.shape[0]

    trainable_layers = [module for module in model.modules() if hasattr(module, 'weight') and module.weight is not None]
    last_layer = trainable_layers[-1]  # 最后一个有权重的层
    weights_ori = [last_layer.weight.data.T, last_layer.bias.data]  # [W, b]，已在 CUDA 上
    print(weights_ori[0].shape, weights_ori[1].shape)

    feature_P = get_feature(model, feature, n_part=n_part)
    y0_P = torch.argmax(feature_P @ weights_ori[0] + weights_ori[1].unsqueeze(0), dim=1).to(torch.int32)

    # initialize sigma and LDM
    sigmas = 10 ** torch.arange(-5, 0.1, 0.1, device=device)
    ldm = torch.ones(m, dtype=torch.float32, device=device)
    ldm_before = ldm.clone()

    pbar = tqdm(total=len(sigmas))
    for sigma in sigmas:
        count = 0
        while count < stop_cond:
            count += 1
            rhos_P = torch.ones(m, dtype=torch.float32, device=device)
            # 扰动权重
            weights_ = [
                torch.normal(mean=weights_ori[0], std=sigma),
                torch.normal(mean=weights_ori[1], std=sigma)
            ]
            y_P = torch.argmax(feature_P @ weights_[0] + weights_[1].unsqueeze(0), dim=1).to(torch.int32)
            rho_P = torch.mean((y0_P != y_P).float())
            rhos_P[y0_P != y_P] = rho_P
            ldm = torch.minimum(ldm, rhos_P)
            if torch.sum(ldm < ldm_before) > 0:
                count = 0
            ldm_before = ldm.clone()
        pbar.update(1)
    pbar.close()

    return ldm.cpu().numpy()


def compute_PoE_logits(weights_ori, feature_P_t, feature_P_v, feature_P_f):
    logits_text = feature_P_t @ weights_ori['text'][0] + weights_ori['text'][1].unsqueeze(0)
    logits_img = feature_P_v @ weights_ori['img'][0] + weights_ori['img'][1].unsqueeze(0)
    logits_fuse = feature_P_f @ weights_ori['fuse'][0] + weights_ori['fuse'][1].unsqueeze(0)

    output_num = (torch.log_softmax(logits_text, dim=-1) +
                  torch.log_softmax(logits_img, dim=-1) +
                  torch.log_softmax(logits_fuse, dim=-1))
    output_den = torch.logsumexp(output_num, dim=-1)
    y_logits = output_num - output_den.unsqueeze(1)

    return y_logits


def get_ldm_gpu2(model, text_feature, img_feature, fuse_feature, stop_cond=10, n_part=1, device='cuda'):  # 移除 X_S 参数
    m = text_feature.shape[0]
    text_model = model.text_classifier
    img_model = model.image_classifier
    fuse_model = model.cat_classifier

    weights_ori = dict()
    model_type = ['text', 'img', 'fuse']
    for i, cls_model in enumerate([text_model, img_model, fuse_model]):
        trainable_layers = [module for module in cls_model.modules() if hasattr(module, 'weight') and module.weight is not None]
        last_layer = trainable_layers[-1]  # 最后一个有权重的层
        weights_ori[model_type[i]] = [last_layer.weight.data.T, last_layer.bias.data]  # [W, b]，已在 CUDA 上

    print(weights_ori['text'][0].shape, weights_ori['text'][1].shape)

    feature_P_t = get_feature(text_model, text_feature, n_part=n_part)
    feature_P_v = get_feature(img_model, img_feature, n_part=n_part)
    feature_P_f = get_feature(fuse_model, fuse_feature, n_part=n_part)
    y0_P = torch.argmax(compute_PoE_logits(weights_ori, feature_P_t, feature_P_v, feature_P_f), dim=1).to(torch.int32)

    # initialize sigma and LDM
    sigmas = 10 ** torch.arange(-5, 0.1, 0.1, device=device)
    ldm = torch.ones(m, dtype=torch.float32, device=device)
    ldm_before = ldm.clone()

    pbar = tqdm(total=len(sigmas))
    weights_ = dict()
    for sigma in sigmas:
        count = 0
        while count < stop_cond:
            count += 1
            rhos_P = torch.ones(m, dtype=torch.float32, device=device)
            # 扰动权重
            for type in model_type:
                weights_[type] = [
                    torch.normal(mean=weights_ori[type][0], std=sigma),
                    torch.normal(mean=weights_ori[type][1], std=sigma)
                ]
            y_P = torch.argmax(compute_PoE_logits(weights_, feature_P_t, feature_P_v, feature_P_f), dim=1).to(torch.int32)
            rho_P = torch.mean((y0_P != y_P).float())
            rhos_P[y0_P != y_P] = rho_P
            ldm = torch.minimum(ldm, rhos_P)
            if torch.sum(ldm < ldm_before) > 0:
                count = 0
            ldm_before = ldm.clone()
        pbar.update(1)
    pbar.close()

    return ldm.cpu().numpy()


def query_by_LDMS(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    # ldm_text = get_ldm_gpu(model.text_classifier, text_f)
    # ldm_img = get_ldm_gpu(model.image_classifier, img_f)
    # ldm_fuse = get_ldm_gpu(model.cat_classifier, fuse_f)
    # ldm = (ldm_text + ldm_img + ldm_fuse) / 3

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)    # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    gamma = np.exp(-np.maximum(ldm - ldm_q, 0) / ldm_q)
    gamma[nQuery:] *= (nQuery / np.sum(gamma[nQuery:]))
    gamma[gamma > 1] = 1

    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature[idx_ordered].cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature[idx_ordered].cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()
    # text_feature = text_f[idx_ordered].cpu().numpy()
    # img_feature = img_f[idx_ordered].cpu().numpy()
    # fuse_feature = fuse_f[idx_ordered].cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3
    D_mat[D_mat < 1e-5] = 0
    # print(D_mat)

    idxs, D2 = [0], D_mat[0]
    pbar = tqdm(total=nQuery, initial=1)
    while len(idxs) < nQuery:     # D2: 当前选择的所有样本到其他样本的最近距离
        if len(idxs) > 1: D2 = np.min([D2, D_mat[idx]], axis=0)   # D_mat[idx]: 当前新选择的样本到所有样本的距离
        px = gamma * D2
        Ddist = (px ** 2) / np.sum(px ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), Ddist))
        idx = customDist.rvs(size=1)[0]
        while idx in idxs: idx = customDist.rvs(size=1)[0]
        idxs.append(idx)
        pbar.update(1)
    pbar.close()

    return idx_ordered[idxs]


def query_by_LDMS2(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)  # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    gamma = np.exp(-np.maximum(ldm - ldm_q, 0) / ldm_q)
    gamma[nQuery:] *= (nQuery / np.sum(gamma[nQuery:]))
    gamma[gamma > 1] = 1

    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature[idx_ordered].cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature[idx_ordered].cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3
    # D_mat[D_mat < 1e-5] = 0
    diversity_score = D_mat.mean(axis=1)
    score = gamma * np.log(1 + diversity_score)
    idxs = np.argsort(score)[-nQuery:][::-1]

    return idx_ordered[idxs]


def query_by_LDMS3(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)    # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    gamma = np.exp(-np.maximum(ldm - ldm_q, 0) / ldm_q)
    gamma[nQuery:] *= (nQuery / np.sum(gamma[nQuery:]))
    gamma[gamma > 1] = 1

    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature[idx_ordered].cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature[idx_ordered].cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3
    # D_mat[D_mat < 1e-5] = 0
    diversity_score = D_mat.mean(axis=1)
    score = gamma * diversity_score
    idxs = np.argsort(score)[-nQuery:][::-1]

    return idx_ordered[idxs]


def query_by_LDMS4(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)
    return idx_ordered[:nQuery]


def query_by_LDMS5(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)    # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    gamma = np.exp(-ldm)


    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature[idx_ordered].cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature[idx_ordered].cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3
    # D_mat[D_mat < 1e-5] = 0
    diversity_score = D_mat.mean(axis=1)
    score = gamma * diversity_score
    idxs = np.argsort(score)[-nQuery:][::-1]

    return idx_ordered[idxs]


def query_by_LDMS6(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)    # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    gamma = np.exp(-ldm)


    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature[idx_ordered].cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature[idx_ordered].cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3
    # D_mat[D_mat < 1e-5] = 0
    diversity_score = D_mat.mean(axis=1)
    score = gamma * np.log(1 + diversity_score)
    idxs = np.argsort(score)[-nQuery:][::-1]

    return idx_ordered[idxs]


def query_by_LDMS7(model, all_features, nQuery, ratio=2):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu2(model, text_f, img_f, fuse_f)

    idx_ordered = np.argsort(ldm)    # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]
    ldm_q = ldm[nQuery]
    if ldm_q == 0: ldm_q = np.min(ldm[ldm > 0])
    # gamma = np.exp(-ldm)

    idx_ordered = idx_ordered[:ratio * nQuery]
    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature[idx_ordered].cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature[idx_ordered].cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3
    # D_mat[D_mat < 1e-5] = 0
    diversity_score = D_mat.mean(axis=1)
    # score = gamma * np.log(1 + diversity_score)
    idxs = np.argsort(diversity_score)[-nQuery:][::-1]

    return idx_ordered[idxs]


def query_by_ablation2(model, all_features, nQuery):
    text_f = all_features['text']
    img_f = all_features['img']
    fuse_f = all_features['fuse']

    text_feature = get_feature(model.text_classifier, text_f)
    text_feature = text_feature.cpu().numpy()
    img_feature = get_feature(model.image_classifier, img_f)
    img_feature = img_feature.cpu().numpy()
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature.cpu().numpy()

    D_mat = (cosine_distances(text_feature, text_feature) + cosine_distances(img_feature, img_feature) +
             cosine_distances(fuse_feature, fuse_feature)) / 3

    diversity_score = D_mat.mean(axis=1)

    idxs = np.argsort(diversity_score)[-nQuery:][::-1]

    return idxs

def query_by_ablation_model(model, all_features, nQuery, ratio=2):

    fuse_f = all_features['fuse']

    ldm = get_ldm_gpu(model.cat_classifier, fuse_f)

    idx_ordered = np.argsort(ldm)  # return idx_ordered[:nQuery]
    ldm = ldm[idx_ordered]

    idx_ordered = idx_ordered[:ratio * nQuery]
    fuse_feature = get_feature(model.cat_classifier, fuse_f)
    fuse_feature = fuse_feature[idx_ordered].cpu().numpy()

    D_mat = cosine_distances(fuse_feature, fuse_feature)
    # D_mat[D_mat < 1e-5] = 0
    diversity_score = D_mat.mean(axis=1)
    # score = gamma * np.log(1 + diversity_score)
    idxs = np.argsort(diversity_score)[-nQuery:][::-1]

    return idx_ordered[idxs]


def active_select(tgt_candidate_loader, tgt_dataset, active_ratio, totality, model, t_step, active_type=None):
    lambda_1 = 7
    lambda_2 = 0.5

    sample_num = math.ceil(totality * active_ratio)
    remove_num = t_step * sample_num

    model.eval()
    all_text_feats_list = []
    all_img_feats_list = []
    all_fuse_feats_list = []
    all_out_logits_list = []

    with torch.no_grad():
        for i, data in enumerate(tgt_candidate_loader):
            tgt_text, tgt_img, tgt_lbl = data[0], data[1], data[2]
            tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()

            logits, feats = model(tgt_text, tgt_img)  # [batch_size, 256]

            all_text_feats_list.append(feats[0])
            all_img_feats_list.append(feats[1])
            all_fuse_feats_list.append(feats[3])
            all_out_logits_list.append(logits[0])

    all_text_feats = torch.cat(all_text_feats_list, dim=0)  # [total_samples, 256]
    all_img_feats = torch.cat(all_img_feats_list, dim=0)  # [total_samples, 256]
    all_fuse_feats = torch.cat(all_fuse_feats_list, dim=0)  # [total_samples, 256]
    all_out_logits = torch.cat(all_out_logits_list, dim=0)  # [total_samples, 2]

    print(f"all_text_feats shape: {all_text_feats.shape}")
    print(f"all_img_feats shape: {all_img_feats.shape}")
    print(f"all_fuse_feats shape: {all_fuse_feats.shape}")

    all_features = {'text': all_text_feats, 'img': all_img_feats, 'fuse': all_fuse_feats, 'logits': all_out_logits}

    first_stat = list()

    if active_type == 'detective':
        with torch.no_grad():
            for i, data in enumerate(tgt_candidate_loader):
                tgt_text, tgt_img, tgt_lbl = data[0], data[1], data[2]
                tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()
                _, _, tgt_out, _, tgt_features = model(tgt_text, tgt_img, return_feature=True)
                # tgt_out, tgt_features = model(tgt_text, tgt_img, return_feature=True)

                alpha = torch.exp(tgt_out)
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p = alpha / total_alpha
                eps = 1e-7

                point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
                data_uncertainty = torch.sum(
                    (alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
                distributional_uncertainty = point_entropy - data_uncertainty

                final_uncertainty = lambda_1 * distributional_uncertainty + lambda_2 * data_uncertainty
                for j in range(len(distributional_uncertainty)):
                    sample_id = i * 32 + j
                    first_stat.append([sample_id, final_uncertainty[j].item(), tgt_features[j].cpu().numpy()])

            print(f"Detective_activa: len(first_stat) = {len(first_stat)}, last sample id = {first_stat[-1][0]}")

        first_stat = sorted(first_stat, key=lambda x: x[1], reverse=True)  # reverse=True: descending order; final_uncertainty
        first_stat = first_stat[:sample_num + remove_num]

        features_array = np.stack([item[2] for item in first_stat])
        features = torch.tensor(features_array)
        uncertainties = [stat[1] for stat in first_stat]
        uncertainties = torch.tensor(uncertainties)

        densities = compute_density(features, uncertainties)

        first_stat = [item[:2] for item in first_stat]
        for i in range(len(first_stat)):
            first_stat[i].append(densities[i])

        first_stat = sorted(first_stat, key=lambda x: x[2], reverse=False)  # reverse=False: ascending order
        first_stat = first_stat[:sample_num]

        select_indices = [item[0] for item in first_stat]

    elif active_type == 'vus':
        x = None
        for text, img, _ in tgt_candidate_loader:
            img = img.cuda()
            text = text.cuda()

            with torch.no_grad():
                _, _, _, _, feature = model(text, img, return_feature=True)
                # _, feature = model(text, img, return_feature=True)
                if x is not None:
                    x = torch.cat((x, feature), dim=0)
                else:
                    x = feature

        pred_model = model.hyper_forward
        vus_ac = vus(pred_model, x, sample_num)

        select_indices = list(vus_ac)

    elif active_type == 'entropy':

        tgt_out = all_features['logits']
        alpha = torch.exp(tgt_out)
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
        expected_p = alpha / total_alpha
        eps = 1e-7

        point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
        first_stat = [[i, val] for i, val in enumerate(point_entropy.tolist())]

        first_stat = sorted(first_stat, key=lambda x: x[1], reverse=True)  # reverse=True: descending order; final_uncertainty
        first_stat = first_stat[:sample_num]

        select_indices = [item[0] for item in first_stat]

    elif active_type == 'margin':

        tgt_out = all_features['logits']  # [total_samples, 2]
        tgt_probs = tgt_out.softmax(dim=1)  # [total_samples, 2]

        # 提取每个样本的前两个最高概率的索引（这里 num_label=2，直接取全部）
        two_most_probable_classes_idx = tgt_probs.argsort(dim=1, descending=True)[:, :2]  # [total_samples, 2]
        gen_y_2_class_probs = torch.gather(tgt_probs, dim=1, index=two_most_probable_classes_idx)  # [total_samples, 2]

        # 计算前两个概率的差值（margin）
        gen_y_2_class_prob_diff = abs(gen_y_2_class_probs[:, 0] - gen_y_2_class_probs[:, 1])  # [total_samples, ]

        # 将 margin 值转为 numpy 数组并排序
        margin_values = gen_y_2_class_prob_diff.cpu().numpy()
        select_indices = margin_values.argsort()[:sample_num]  # 从小到大排序，取前 sample_num 个
        select_indices = select_indices.tolist()

    elif active_type == 'ldms':

        with torch.no_grad():
            idx_query = query_by_LDMS(model, all_features, sample_num)
            select_indices = idx_query.tolist()

    elif active_type == 'random':
        select_indices = np.random.choice(len(tgt_dataset), sample_num, replace=False)
        select_indices = select_indices.tolist()

    elif active_type == 'ldms2':

        with torch.no_grad():
            idx_query = query_by_LDMS2(model, all_features, sample_num)
            select_indices = idx_query.tolist()

    elif active_type == 'ldms3':

        with torch.no_grad():
            idx_query = query_by_LDMS3(model, all_features, sample_num)
            select_indices = idx_query.tolist()

    elif active_type == 'ldms4':

        with torch.no_grad():
            idx_query = query_by_LDMS4(model, all_features, sample_num)
            select_indices = idx_query.tolist()

    elif active_type == 'ldms5':

        with torch.no_grad():
            idx_query = query_by_LDMS5(model, all_features, sample_num)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms6':

        with torch.no_grad():
            idx_query = query_by_LDMS6(model, all_features, sample_num)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms7':

        with torch.no_grad():
            idx_query = query_by_LDMS7(model, all_features, sample_num, ratio=2)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms8':

        with torch.no_grad():
            idx_query = query_by_LDMS7(model, all_features, sample_num, ratio=3)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms9':

        with torch.no_grad():
            idx_query = query_by_LDMS7(model, all_features, sample_num, ratio=4)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms10':

        with torch.no_grad():
            idx_query = query_by_LDMS7(model, all_features, sample_num, ratio=5)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms11':

        with torch.no_grad():
            idx_query = query_by_LDMS7(model, all_features, sample_num, ratio=t_step + 1)
            select_indices = idx_query.tolist()
    elif active_type == 'ldms12':

        with torch.no_grad():
            idx_query = query_by_LDMS7(model, all_features, sample_num, ratio=10)
            select_indices = idx_query.tolist()
    elif active_type == 'ablation1':

        with torch.no_grad():
            idx_query = query_by_LDMS4(model, all_features, sample_num)
            select_indices = idx_query.tolist()
    elif active_type == 'ablation2':

        with torch.no_grad():
            idx_query = query_by_ablation2(model, all_features, sample_num)
            select_indices = idx_query.tolist()


    select_indices.sort()
    selected_samples = tgt_dataset.remove_item(select_indices)

    return selected_samples


def active_select2(tgt_candidate_loader, tgt_dataset, active_ratio, totality, model, t_step, active_type=None):
    lambda_1 = 7
    lambda_2 = 0.5

    sample_num = math.ceil(totality * active_ratio)
    remove_num = t_step * sample_num

    model.eval()
    first_stat = list()

    if active_type == 'detective':
        with torch.no_grad():
            for i, data in enumerate(tgt_candidate_loader):
                tgt_text, tgt_img, tgt_lbl = data[0], data[1], data[2]
                tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()
                _, _, tgt_out, _, tgt_features = model(tgt_text, tgt_img, return_feature=True)
                # tgt_out, tgt_features = model(tgt_text, tgt_img, return_feature=True)

                alpha = torch.exp(tgt_out)
                total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
                expected_p = alpha / total_alpha
                eps = 1e-7

                point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
                data_uncertainty = torch.sum(
                    (alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
                distributional_uncertainty = point_entropy - data_uncertainty

                final_uncertainty = lambda_1 * distributional_uncertainty + lambda_2 * data_uncertainty
                for j in range(len(distributional_uncertainty)):
                    sample_id = i * 32 + j
                    first_stat.append([sample_id, final_uncertainty[j].item(), tgt_features[j].cpu().numpy()])

            print(f"Detective_activa: len(first_stat) = {len(first_stat)}, last sample id = {first_stat[-1][0]}")

        first_stat = sorted(first_stat, key=lambda x: x[1], reverse=True)  # reverse=True: descending order; final_uncertainty
        first_stat = first_stat[:sample_num + remove_num]

        features_array = np.stack([item[2] for item in first_stat])
        features = torch.tensor(features_array)
        uncertainties = [stat[1] for stat in first_stat]
        uncertainties = torch.tensor(uncertainties)

        densities = compute_density(features, uncertainties)

        first_stat = [item[:2] for item in first_stat]
        for i in range(len(first_stat)):
            first_stat[i].append(densities[i])

        first_stat = sorted(first_stat, key=lambda x: x[2], reverse=False)  # reverse=False: ascending order
        first_stat = first_stat[:sample_num]

        select_indices = [item[0] for item in first_stat]

    elif active_type == 'EADA':
        energy_beta = 1.0
        first_sample_ratio = 0.5

        with torch.no_grad():
            for i, data in enumerate(tgt_candidate_loader):
                tgt_text, tgt_img, tgt_lbl = data[0], data[1], data[2]
                tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()

                _, _, tgt_out, _ = model(tgt_text, tgt_img)

                # MvSM of each sample
                # minimal energy - second minimal energy
                min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
                mvsm_uncertainty = min2[:, 0] - min2[:, 1]

                # free energy of each sample
                output_div_t = -1.0 * tgt_out / energy_beta
                output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
                free_energy = -1.0 * energy_beta * output_logsumexp

                # for i in range(len(free_energy)):
                #     first_stat.append([tgt_path[i], tgt_lbl[i].item(), tgt_index[i].item(),
                #                        mvsm_uncertainty[i].item(), free_energy[i].item()])

                for j in range(len(free_energy)):
                    sample_id = i * 32 + j
                    first_stat.append([sample_id, tgt_lbl[j].item(), mvsm_uncertainty[j].item(),
                                       free_energy[j].item()])

        first_sample_ratio = first_sample_ratio
        first_sample_num = math.ceil(totality * first_sample_ratio)
        second_sample_ratio = active_ratio / first_sample_ratio
        second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

        # the first sample using \mathca{F}, higher value, higher consideration
        first_stat = sorted(first_stat, key=lambda x: x[3], reverse=True)  # free_energy
        second_stat = first_stat[:first_sample_num]

        # the second sample using \mathca{U}, higher value, higher consideration
        second_stat = sorted(second_stat, key=lambda x: x[2], reverse=True)
        second_stat = second_stat[:second_sample_num]

        select_indices = [item[0] for item in second_stat]

    elif active_type == 'entropy':

        tgt_out = all_features['logits']
        alpha = torch.exp(tgt_out)
        print(alpha.shape)
        total_alpha = torch.sum(alpha, dim=1, keepdim=True)  # total_alpha.shape: [B, 1]
        expected_p = alpha / total_alpha
        eps = 1e-7

        point_entropy = - torch.sum(expected_p * torch.log(expected_p + eps), dim=1)
        first_stat = [[i, val] for i, val in enumerate(point_entropy.tolist())]

        first_stat = sorted(first_stat, key=lambda x: x[1], reverse=True)  # reverse=True: descending order; final_uncertainty
        first_stat = first_stat[:sample_num]

        select_indices = [item[0] for item in first_stat]

    elif active_type == 'random':
        select_indices = np.random.choice(len(tgt_dataset), sample_num, replace=False)
        select_indices = select_indices.tolist()

    elif active_type == 'abla_model':

        all_fuse_feats_list = []
        with torch.no_grad():
            for i, data in enumerate(tgt_candidate_loader):
                tgt_text, tgt_img, tgt_lbl = data[0], data[1], data[2]
                tgt_text, tgt_img, tgt_lbl = tgt_text.cuda(), tgt_img.cuda(), tgt_lbl.cuda()

                logits, feats = model(tgt_text, tgt_img)  # [batch_size, 256]
                all_fuse_feats_list.append(feats[3])

            all_fuse_feats = torch.cat(all_fuse_feats_list, dim=0)  # [total_samples, 256]
            all_features = {'fuse': all_fuse_feats}

            idx_query = query_by_ablation_model(model, all_features, sample_num, ratio=2)
            select_indices = idx_query.tolist()

    select_indices.sort()
    selected_samples = tgt_dataset.remove_item(select_indices)

    return selected_samples
