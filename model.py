import os
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from math import sqrt
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F
from block.KeywordDiscriminator import KeywordDiscriminator


#GaussianKLLoss 类用于计算两个高斯分布之间的 KL 散度。KL 散度常用于衡量两个概率分布的差异。在这个模型中，它用于计算由噪声网络生成的分布与先验分布之间的差异。
class GaussianKLLoss(nn.Module): #Kullback-Leibler (KL) 散度
    def __init__(self):
        super(GaussianKLLoss, self).__init__()
    # mu1:第一个分布的均值
    # logvar1：第一个分布的对数方差。
    # mu2：第二个分布的均值（通常是先验分布的均值）。
    # logvar2：第二个分布的对数方差（通常是先验分布的方差）。
    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)

#自定义的 BERT 模型，加入噪声增强和 KL 散度正则化
class NewBert(nn.Module):
    def __init__(self, args):
        super(NewBert, self).__init__()
        self.model_name = args["model"]
        self.bert_model = BertForSequenceClassification.from_pretrained(
            args["model"], num_labels=args["n_class"]) #n_class 分类类别数量
        # parser.add_argument("--n_class", type=int, default=2, help="分类类别数量")
        # parser.add_argument('--model', type=str, default='/data/wyh/graduate/data/bert-base-uncased', help="预训练模型") 模型加载地址
        # self.noise_net :是一个小型的全连接网络，用于生成噪声，基于输入的隐藏层特征生成均值（mu）和对数方差（logvar）
        self.noise_net = nn.Sequential(nn.Linear(args["hidden_size"],
                                                 args["hidden_size"]),
                                       nn.ReLU(),
                                       nn.Linear(args["hidden_size"],
                                                 args["hidden_size"] * 2))
        config = self.bert_model.config
        self.config = config
        self.dropout = config.hidden_dropout_prob  # 0.1
        self.args = args

        # 添加关键词鉴别网络
        self.keyword_discriminator = KeywordDiscriminator(
            hidden_size=args["hidden_size"], num_heads=4, dropout=args["hidden_dropout_prob"]
        )

        # 定义Gate 网络; 假设输入是2 X 768架构
        # self.pool = nn.AdaptiveAvgPool2d((1, 768))
        # Gate网络：self.Gate 网络通过一个全连接层和ReLU激活来生成三类的权重。Gate网络用于动态选择如何结合噪声损失和标准的负对数似然损失（NLL）。
        if self.args["gate"]:
            self.Gate = nn.Sequential(nn.Linear( 2 * args["hidden_size"],
                                                    args["hidden_size"]),
                                        nn.ReLU(),
                                        nn.Linear(args["hidden_size"],
                                                    3))
        # y1= F.softmax(x, dim = 0) #对每一列进行softmax

    # forward方法是模型的核心，负责处理输入数据并计算损失。根据输入的参数（如是否使用噪声增强aug），执行以下步骤：
    # 噪声增强：
    # 如果启用了噪声增强（aug = True），会对BERT的嵌入进行噪声扰动。这个过程会通过self.noise_net网络生成噪声，并将其加入到输入的BERT嵌入中。噪声会根据当前输入数据的隐藏状态生成，并且通过KL散度计算与先验分布之间的差异，从而得到正则化项。
    # Gate网络：
    # 如果启用了Gate网络（gate = True），会通过模型的Gate网络来动态地调整噪声损失和标准负对数似然损失（NLL）的权重。
    # 标准BERT前向传播：
    # 如果没有启用噪声增强，模型仅进行标准的BERT前向传播，返回负对数似然损失。
    def forward(self, input_ids, attention_mask, token_type_ids, input_chunk, labels):
        input_ids, attention_mask, token_type_ids = input_ids.squeeze(), attention_mask.squeeze(), token_type_ids.squeeze()
        if self.args["aug"]:  # 噪声扰动
            embeddings = self.bert_model.get_input_embeddings()  # 获取 BERT 模型的嵌入层
            encoder = self.bert_model.bert  # 获取 BERT 模型的编码器部分
            with torch.no_grad():  # 禁用梯度计算
                encoder_inputs = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}
                outputs = encoder(**encoder_inputs)
                hiddens = outputs[0]  # 获取 BERT 的隐藏状态

            inputs_embeds = embeddings(input_ids)  # 获取输入嵌入

            # # 使用关键词鉴别网络获取关键词的重要性分数
            # keyword_scores = self.keyword_discriminator(hiddens.transpose(0, 1), attention_mask)
            # keyword_scores = keyword_scores.unsqueeze(-1)  # (batch_size, seq_len, 1)

            if self.args['uniform']:
                # 如果启用了 uniform（均匀噪声），生成均匀分布的噪声矩阵
                uniform_noise = torch.empty(inputs_embeds.shape).uniform_(0.9995, 1.0005).to(self.args['device'])
                noise = uniform_noise
            else:
                # 如果没有启用均匀噪声，根据 BERT 编码器的隐藏状态生成噪声
                mask = attention_mask.view(-1)
                indices = (mask == 1)
                mu_logvar = self.noise_net(hiddens)  # 使用噪声网络来生成均值和对数方差
                mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)  # 分割 mu 和 log_var
                zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)  # 生成噪声
                noise = zs

                # KL 散度计算
                prior_mu = torch.ones_like(mu)  # 先验均值为 1
                prior_var = torch.ones_like(mu) * sqrt(self.dropout / (1 - self.dropout))  # 先验方差
                prior_logvar = torch.log(prior_var)  # 先验对数方差

                kl_criterion = GaussianKLLoss()  # 用于计算 KL 散度
                h = hiddens.size(-1)
                _mu = mu.view(-1, h)[indices]
                _log_var = log_var.view(-1, h)[indices]
                _prior_mu = prior_mu.view(-1, h)[indices]
                _prior_logvar = prior_logvar.view(-1, h)[indices]

                kl = kl_criterion(_mu, _log_var, _prior_mu, _prior_logvar)

            # 根据关键词分数调整扰动强度
            # perturbation_strength = 1 - keyword_scores  # 对关键词施加较小的扰动
            # noise = noise * perturbation_strength  # 动态调整噪声强度

            # 随机丢弃（Random Discarding）操作
            rands = list(set([random.randint(1, inputs_embeds.shape[0] - 1) for i in range(self.args["zero_peturb"])]))
            for index in rands:
                embed_ = inputs_embeds[index, :, :]
                length = random.randint(1, 3)
                for iter in range(length):
                    index_ = random.randint(1, inputs_embeds.shape[1] - 1)
                    vec = torch.rand(1, inputs_embeds.shape[-1]).to(self.args["device"])
                    embed_[index_] = vec

            # 重新构建输入并进行前向传播（带噪声的输入）
            inputs = {"inputs_embeds": inputs_embeds * noise,  # 将噪声与嵌入相乘
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "labels": labels}

            noise_outputs = self.bert_model(**inputs, output_hidden_states=True)
            noise_loss = noise_outputs[0]  # 获取带噪声的模型输出的损失

            # 重新构建输入并进行标准前向传播（不带噪声的输入）
            new_inputs = {"inputs_embeds": inputs_embeds,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "labels": labels}

            outputs = self.bert_model(**new_inputs, output_hidden_states=True)
            nll = outputs[0]

            # 如果启用了 Gate 网络，动态调整噪声损失和 NLL 损失的权重
            # 论文中adapter的体现
            if self.args["gate"]:
                last_noise = noise_outputs.hidden_states[-1]
                last = outputs.hidden_states[-1]
                cls_noise = last_noise[:, :1, :].squeeze()
                cls = last[:, :1, :].squeeze()
                cls_total = torch.cat((cls_noise, cls), dim=1)
                cls_total = torch.mean(cls_total, dim=0).unsqueeze(dim=0)
                res = self.Gate(cls_total)
                Gates = F.softmax(res, dim=-1).squeeze()
                loss = noise_loss * Gates[0] + nll * Gates[1]
            else:
                loss = nll + 0.001 * noise_loss

            if self.args['uniform']:
                return (loss, 0 * loss, outputs.logits)
            else:
                return (loss, kl, outputs.logits)
        else:
            # 没有启用噪声增强
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids}
            outputs = self.bert_model(**inputs, labels=labels)
            return outputs.loss, 0, outputs.logits