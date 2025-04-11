import torch
import torch.nn as nn
import torchvision
from baseline import TextCNN
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=256, num_heads=4, hidden_dim=128):
        super(CrossAttentionFusion, self).__init__()
        self.feature_dim = feature_dim

        # 文本和图像的自注意力
        self.text_self_att = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.img_self_att = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

        # 交叉注意力
        self.text_to_img_att = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.img_to_text_att = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

        # 特征降维与融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, text_features, img_features):
        # 调整维度以适配MultiheadAttention: (seq_len, batch_size, feature_dim)
        text_features = text_features.unsqueeze(0)  # (1, N, feature_dim)
        img_features = img_features.unsqueeze(0)  # (1, N, feature_dim)

        # 自注意力增强特征
        text_att, _ = self.text_self_att(text_features, text_features, text_features)
        img_att, _ = self.img_self_att(img_features, img_features, img_features)

        # 交叉注意力
        text_enhanced, _ = self.text_to_img_att(text_att, img_att, img_att)  # 文本关注图像
        img_enhanced, _ = self.img_to_text_att(img_att, text_att, text_att)  # 图像关注文本

        # 融合特征
        fused_features = torch.cat([text_enhanced.squeeze(0), img_enhanced.squeeze(0)], dim=1)  # (N, feature_dim*2)
        output = self.fusion_layer(fused_features)  # (N, feature_dim)

        return output


class Encoder_TextCnn(nn.Module):
    def __init__(self, out_size=300, freeze_id=-1, d_prob=0.3,
                 kernel_sizes=[3, 4, 5], num_filters=100, mode='rand', dataset_name="Pheme"):
        super(Encoder_TextCnn, self).__init__()
        self.out_size = out_size
        self.droprate = d_prob
        self.mode = mode
        self.dataset_name = dataset_name
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters

        self.visual_encoder = torchvision.models.resnet50(pretrained=True)
        # self.visual_encoder = torchvision.models.resnet101(pretrained=True)
        # remove classification layer
        # kernel_sizes, num_filters, num_classes, d_prob, mode = 'rand-', dataset_name = "Pheme"
        self.textcnn = TextCNN(kernel_sizes=self.kernel_sizes,
                               num_filters=self.num_filters,
                               num_classes=self.out_size, d_prob=self.droprate, mode=self.mode,
                               dataset_name=self.dataset_name
                               )
        self.instance_discriminator = torchvision.models.resnet50(pretrained=True)

        # self.instance_discriminator = self.visual_encoder.fc
        self.visual_encoder = torch.nn.Sequential(*(list(self.visual_encoder.children())[:-1]))
        self.text_linear = nn.Sequential(nn.LayerNorm(self.out_size))
        self.img_linear = nn.Sequential(nn.Linear(in_features=2048, out_features=1024), nn.ReLU(),
                                        # nn.Dropout(p=0.2),
                                        nn.Linear(in_features=1024, out_features=self.out_size),
                                        nn.ReLU(),
                                        nn.Dropout(self.droprate),
                                        nn.LayerNorm(self.out_size)
                                        )
        # id = 8 freeze all parameter
        # id=7 freeze the last
        # id=-1 do not freeze
        self.freeze_layer(self.visual_encoder, freeze_id)
        for para in self.instance_discriminator.parameters():
            para.requires_grad = False

    def forward(self, texts, imgs):
        # texts = self.bert_model(**texts)[0]
        instance_cls = self.instance_discriminator(imgs)
        imgs = self.visual_encoder(imgs)
        texts = self.textcnn(texts)
        # remove cls token and sep token
        # texts = texts[:, 1:-1, :]
        # only use [cls] token for classification
        imgs = imgs.squeeze()
        # instance_cls = self.instance_discriminator(imgs)
        texts = self.text_linear(texts)
        imgs = self.img_linear(imgs)
        return texts, imgs, instance_cls

    def get_parameters(self):
        textcnn_params = list(map(id, self.textcnn.parameters()))
        conv_params = list(map(id, self.visual_encoder.parameters()))
        params = [
            {"params": self.textcnn.parameters(), "lr": None},
            {"params": self.visual_encoder.parameters(), "lr": None},
            {"params": filter(lambda p: id(p) not in textcnn_params + conv_params, self.parameters()), "lr": None},
        ]
        return params

    def freeze_layer(self, model, freeze_layer_ids):
        count = 0
        para_optim = []
        for k in model.children():
            # 6 should be changed properly
            if count > freeze_layer_ids:  # 6:
                for param in k.parameters():
                    para_optim.append(param)
            else:
                for param in k.parameters():
                    param.requires_grad = False
            count += 1
        # print count
        return para_optim


class HyperNetwork(nn.Module):

    def __init__(self, out_size, in_size, z_dim=256):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.Tensor(self.z_dim, self.out_size * self.in_size))
        self.b1 = Parameter(torch.Tensor(self.out_size * self.in_size))
        self.w2 = Parameter(torch.Tensor(self.z_dim, self.out_size))
        self.b2 = Parameter(torch.Tensor(self.out_size))

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.constant_(self.b1, 0)
        nn.init.constant_(self.b2, 0)

    def forward(self, z):
        # z: (N, z_dim)
        h_final = torch.matmul(z, self.w1) + self.b1  # (1, out_size * in_size)
        w = h_final.view(self.in_size, self.out_size)   # shape = (in_size, out_size)
        b = torch.matmul(z, self.w2) + self.b2          # shape = (out_size)
        return [w, b]


class MLPModel_TextCnn(nn.Module):
    def __init__(self, out_size=256, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(MLPModel_TextCnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)

        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)
        # self.hyper = HyperNetwork(out_size=self.num_label, in_size=self.out_size, z_dim=self.out_size)

    def forward(self, train_texts, train_imgs, return_feature=False, reverse_grad=False):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        # if train_texts.dim() == 1:
        #     train_texts, train_imgs = train_texts.unsqueeze(0), train_imgs.unsqueeze(0)
        train_texts, train_imgs = train_texts.view(-1, self.out_size), train_imgs.view(-1, self.out_size)

        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))  # (N, out_size)
        # z = torch.mean(f, dim=0)  # (out_size)
        # w, b = self.hyper(z.view(-1, self.out_size))
        # train_y = torch.matmul(f, w) + b  # dynamic classifier
        if reverse_grad:
            f_reversed = grad_reverse(f)
            train_y = self.head(f_reversed)
        else:
            train_y = self.head(f)

        # return train_texts, train_imgs, train_y, instance_cls
        if return_feature:
            return train_texts, train_imgs, train_y, instance_cls, f
        else:
            return train_texts, train_imgs, train_y, instance_cls

    def hyper_forward(self, feature):
        pred_y = self.head(feature)
        return pred_y

    def get_param(self):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))

        params = [
            {"params": self.encoder.parameters(), 'lr': None},
            # {'params': self.cls_layer.parameters(), 'lr': 1.0 * initial_lr},
            # {'params': self.hyper.parameters(), 'lr': 1.0 * initial_lr}]
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": None}]

        return params


class HyperModel_TextCnn(nn.Module):
    def __init__(self, out_size=256, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(HyperModel_TextCnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)

        self.cls_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=int(self.out_size)))
        # self.head = nn.Linear(in_features=self.out_size, out_features=self.num_label)
        self.hyper = HyperNetwork(out_size=self.num_label, in_size=self.out_size, z_dim=self.out_size)

    def forward(self, train_texts, train_imgs, return_feature=False):
        """

        Args:
            train_texts: (N,dim)
            train_imgs:
            test_texts:
            test_imgs:
            train_labels:

        Returns:

        """
        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        train_texts, train_imgs = train_texts.view(-1, self.out_size), train_imgs.view(-1, self.out_size)

        f = self.cls_layer(torch.cat([train_texts, train_imgs], dim=1))  # (N, out_size)
        z = torch.mean(f, dim=0)  # (out_size)
        w, b = self.hyper(z.view(-1, self.out_size))
        train_y = torch.matmul(f, w) + b  # dynamic classifier
        # train_y = self.head(f)

        # return train_texts, train_imgs, train_y, instance_cls
        if return_feature:
            return train_y, f
        else:
            return train_y

    def hyper_forward(self, feature):
        z = torch.mean(feature, dim=0)  # (out_size)
        w, b = self.hyper(z.view(-1, self.out_size))
        pred_y = torch.matmul(feature, w) + b
        return pred_y

    def get_param(self, initial_lr=0.1):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))

        params = [
            {"params": self.encoder.parameters(), 'lr': 0.1 * initial_lr},
            # {'params': self.cls_layer.parameters(), 'lr': 1.0 * initial_lr},
            # {'params': self.hyper.parameters(), 'lr': 1.0 * initial_lr}]
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": 1.0 * initial_lr}]

        return params


class AttenModel_TextCnn(nn.Module):
    def __init__(self, out_size=256, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(AttenModel_TextCnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name

        # 编码器
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)

        # 替换简单拼接为交叉注意力融合
        self.fusion = CrossAttentionFusion(feature_dim=self.out_size)

        # 超网络
        self.hyper = HyperNetwork(out_size=self.num_label, in_size=self.out_size, z_dim=self.out_size)

    def forward(self, train_texts, train_imgs, return_feature=False):

        train_texts, train_imgs, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        train_texts = train_texts.view(-1, self.out_size)  # (N, out_size)
        train_imgs = train_imgs.view(-1, self.out_size)  # (N, out_size)


        f = self.fusion(train_texts, train_imgs)  # (N, out_size)


        z = torch.mean(f, dim=0)  # (out_size)
        w, b = self.hyper(z.view(-1, self.out_size))
        train_y = torch.matmul(f, w) + b  # (N, num_label)

        if return_feature:
            return train_y, f
        else:
            return train_y

    def hyper_forward(self, feature):
        z = torch.mean(feature, dim=0)  # (out_size)
        w, b = self.hyper(z.view(-1, self.out_size))
        pred_y = torch.matmul(feature, w) + b
        return pred_y

    def get_param(self, initial_lr=0.1):
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        encoder_params = list(map(id, self.encoder.parameters()))

        params = [
            {"params": self.encoder.parameters(), 'lr': 0.1 * initial_lr},
            {"params": filter(lambda p: id(p) not in encoder_params, self.parameters()), "lr": 1.0 * initial_lr}
        ]

        return params


class ImageClassifier(nn.Module):
    def __init__(self, in_size=256, hidden_size=256, num_label=2):
        super(ImageClassifier, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_label = num_label
        self.model = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_label)
        )

    def forward(self, img_feats):
        return self.model(img_feats)  # [batch_size, num_label]


class TextClassifier(nn.Module):
    def __init__(self, in_size=256, hidden_size=256, num_label=2):
        super(TextClassifier, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_label = num_label
        self.model = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_label)
        )

    def forward(self, text_feats):
        return self.model(text_feats)  # [batch_size, num_label]


class CatClassifier(nn.Module):
    def __init__(self, in_size=256, hidden_size=256, num_label=2):
        super(CatClassifier, self).__init__()
        self.in_size = in_size  # 输入为 text_feats 和 img_feats 拼接后的维度
        self.hidden_size = hidden_size
        self.num_label = num_label
        self.model = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.num_label)
        )

    def forward(self, cat_feats):
        return self.model(cat_feats)  # [batch_size, num_label]


# 梯度反转层实现
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)


# 独立的域判别器类
class DomainDiscriminator(nn.Module):
    def __init__(self, in_size, hidden_size, num_domains=4):
        super(DomainDiscriminator, self).__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_domains)  # 输出域类别数，默认2（源域和目标域）
        )

    def forward(self, x, domain_labels):
        domain_logits = self.model(x)  # 输出域判别器的 logits
        loss = self.criterion(domain_logits, domain_labels)  # 计算对抗损失
        return loss


class JointModel_TextCnn(nn.Module):
    def __init__(self, out_size=256, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(JointModel_TextCnn, self).__init__()
        self.out_size = out_size
        self.freeze_id = freeze_id
        self.num_label = num_label
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name
        # encoder for text and image
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)

        self.fusion_layer = nn.Sequential(nn.Linear(in_features=self.out_size * 2, out_features=self.out_size))
        # three separate classifier
        self.image_classifier = ImageClassifier(in_size=self.out_size, hidden_size=self.out_size, num_label=self.num_label)
        self.text_classifier = TextClassifier(in_size=self.out_size, hidden_size=self.out_size, num_label=self.num_label)
        self.cat_classifier = CatClassifier(in_size=self.out_size, hidden_size=self.out_size, num_label=self.num_label)

        # self.gate = nn.Linear(self.out_size * 2, 3)

    def forward(self, train_texts, train_imgs):
        # feature extraction
        text_feats, img_feats, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        text_feats = text_feats.view(-1, self.out_size)
        img_feats = img_feats.view(-1, self.out_size)
        fuse_feats = self.fusion_layer(torch.cat([text_feats, img_feats], dim=1))  # (N, out_size)

        # calculate logits
        img_logits = self.image_classifier(img_feats)  # [batch_size, num_label]
        text_logits = self.text_classifier(text_feats)  # [batch_size, num_label]
        cat_logits = self.cat_classifier(fuse_feats)

        # # 门控网络生成权重
        # gate_logits = self.gate(torch.cat([text_feats, img_feats], dim=1))  # [batch_size, 3]
        # gate_weights = F.softmax(gate_logits, dim=-1)  # [batch_size, 3]

        # PoE 组合
        output_num = (torch.log_softmax(img_logits, dim=-1) +
                      torch.log_softmax(text_logits, dim=-1) +
                      torch.log_softmax(cat_logits, dim=-1))
        output_den = torch.logsumexp(output_num, dim=-1)
        y_logits = output_num - output_den.unsqueeze(1)  # [batch_size, num_label]

        # # 加权组合专家输出
        # expert_logits = torch.stack([img_logits, text_logits, cat_logits], dim=1)  # [batch_size, 3, num_label]
        # y_logits = torch.einsum('be,bec->bc', gate_weights, expert_logits)  # [batch_size, num_label]

        logits = (y_logits, text_logits, img_logits, cat_logits)
        feats = (text_feats, img_feats, instance_cls, fuse_feats)

        return logits, feats



    def get_param(self):

        # params = [
        #     {"params": self.encoder.parameters(), "lr": None},
        #     {"params": self.fusion_layer.parameters(), "lr": None},
        #     {"params": self.image_classifier.parameters(), "lr": None},
        #     {"params": self.text_classifier.parameters(), "lr": None},
        #     {"params": self.cat_classifier.parameters(), "lr": None}
        # ]
        # return params

        return [{"params": self.parameters(), "lr": None}]


class JointModel_TextCnn2(nn.Module):
    def __init__(self, out_size=256, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(JointModel_TextCnn2, self).__init__()
        self.out_size = out_size
        self.num_label = num_label
        self.freeze_id = freeze_id
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name

        # 文本和图片的编码器
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)

        # 用于对齐的多层感知机 (MLP)
        self.mlp_img = nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=self.out_size),
            nn.ReLU(),
            nn.Linear(in_features=self.out_size, out_features=self.out_size)
        )
        self.mlp_text = nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=self.out_size),
            nn.ReLU(),
            nn.Linear(in_features=self.out_size, out_features=self.out_size)
        )

        # 三个独立的分类器
        self.image_classifier = ImageClassifier(in_size=self.out_size, hidden_size=self.out_size, num_label=self.num_label)
        self.text_classifier = TextClassifier(in_size=self.out_size, hidden_size=self.out_size, num_label=self.num_label)
        self.cat_classifier = CatClassifier(in_size=self.out_size * 2, hidden_size=self.out_size, num_label=self.num_label)


    def forward(self, train_texts, train_imgs, labels=None):
        # 特征提取
        text_feats, img_feats, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        text_feats = text_feats.view(-1, self.out_size)  # [batch_size, out_size]
        img_feats = img_feats.view(-1, self.out_size)    # [batch_size, out_size]

        aligned_text_feats = F.normalize(self.mlp_text(text_feats))
        aligned_img_feats = F.normalize(self.mlp_img(img_feats))

        # aligned_text_feats = self.mlp_text(text_feats)
        # aligned_img_feats = self.mlp_img(img_feats)

        # 计算对比损失并对齐特征
        if labels is not None:
            instance_cls = F.normalize(instance_cls)
            contrastive_data = (aligned_text_feats, aligned_img_feats, instance_cls, labels)
        else:
            contrastive_data = None

        # 特征拼接
        cat_feats = torch.cat([aligned_text_feats, aligned_img_feats], dim=1)

        # 计算各个分类器的 logits
        img_logits = self.image_classifier(img_feats)    # [batch_size, num_label]
        text_logits = self.text_classifier(text_feats)   # [batch_size, num_label]
        cat_logits = self.cat_classifier(cat_feats)     # [batch_size, num_label]

        # PoE 组合
        output_num = (torch.log_softmax(img_logits, dim=-1) +
                      torch.log_softmax(text_logits, dim=-1) +
                      torch.log_softmax(cat_logits, dim=-1))
        output_den = torch.logsumexp(output_num, dim=-1)
        y_logits = output_num - output_den.unsqueeze(1)  # [batch_size, num_label]

        logits = (y_logits, text_logits, img_logits, cat_logits)
        feats = (text_feats, img_feats, instance_cls, cat_feats)

        if contrastive_data is not None:
            return logits, feats, contrastive_data
        return logits, feats

    def get_param(self):

        return [{"params": self.parameters(), "lr": None}]


class JointModel_TextCnn_abla(nn.Module):
    def __init__(self, out_size=256, num_label=2, freeze_id=7, d_prob=0.3, kernel_sizes=[3, 4, 5], num_filters=100,
                 mode='rand', dataset_name="Pheme"):
        super(JointModel_TextCnn_abla, self).__init__()
        self.out_size = out_size
        self.num_label = num_label
        self.freeze_id = freeze_id
        self.d_prob = d_prob
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.mode = mode
        self.dataset_name = dataset_name

        # 文本和图片的编码器
        self.encoder = Encoder_TextCnn(out_size=self.out_size, freeze_id=self.freeze_id,
                                       d_prob=self.d_prob, kernel_sizes=self.kernel_sizes, num_filters=self.num_filters,
                                       mode=self.mode, dataset_name=self.dataset_name)

        # 用于对齐的多层感知机 (MLP)
        self.mlp_img = nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=self.out_size),
            nn.ReLU(),
            nn.Linear(in_features=self.out_size, out_features=self.out_size)
        )
        self.mlp_text = nn.Sequential(
            nn.Linear(in_features=self.out_size, out_features=self.out_size),
            nn.ReLU(),
            nn.Linear(in_features=self.out_size, out_features=self.out_size)
        )

        self.cat_classifier = CatClassifier(in_size=self.out_size * 2, hidden_size=self.out_size, num_label=self.num_label)


    def forward(self, train_texts, train_imgs, labels=None):
        # 特征提取
        text_feats, img_feats, instance_cls = self.encoder(texts=train_texts, imgs=train_imgs)
        text_feats = text_feats.view(-1, self.out_size)  # [batch_size, out_size]
        img_feats = img_feats.view(-1, self.out_size)    # [batch_size, out_size]

        aligned_text_feats = F.normalize(self.mlp_text(text_feats))
        aligned_img_feats = F.normalize(self.mlp_img(img_feats))

        # 计算对比损失并对齐特征
        if labels is not None:
            instance_cls = F.normalize(instance_cls)
            contrastive_data = (aligned_text_feats, aligned_img_feats, instance_cls, labels)
        else:
            contrastive_data = None

        # 特征拼接
        cat_feats = torch.cat([aligned_text_feats, aligned_img_feats], dim=1)


        y_logits = self.cat_classifier(cat_feats)     # [batch_size, num_label]


        logits = y_logits
        feats = (text_feats, img_feats, instance_cls, cat_feats)

        if contrastive_data is not None:
            return logits, feats, contrastive_data
        return logits, feats

    def get_param(self):

        return [{"params": self.parameters(), "lr": None}]