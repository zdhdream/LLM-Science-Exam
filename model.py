import torch.nn as nn
import numpy as np
import torch
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel, \
    get_cosine_schedule_with_warmup, AutoConfig


class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param  # 指定对哪些参数进行扰动
        self.adv_lr = adv_lr  # 扰动的学习率，控制扰动幅度
        self.adv_eps = adv_eps  # 控制扰动范围
        self.backup = {} # 用于记录原始的word_embedding的模型参数,帮助我们快速进行还原

    def perturb(self, input_ids, attention_mask, token_type_ids, y, criterion):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            # (1) 梯度可导（2）梯度存在（3）在adv_param中
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                # 获取参数的指数平均梯度(这是优化器在训练过程中维护的一个参数的梯度的指数移动平均值)
                grad = self.optimizer.state[param]['exp_avg']
                # 计算梯度的范数，用于衡量梯度大小
                norm_grad = torch.norm(grad)
                # 计算参数的值（张量）的范数，用于衡量参数的大小
                norm_data = torch.norm(param.detach())
                # 检查梯度范式是否不为零且不是NaN
                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    # 计算扰动后的下届
                    param_min = param.data - limit_eps
                    # 计算扰动的上界
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            # (1)参数可导 (2)梯度不为空 (3) 参数在对抗性扰动列表中
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    # 保存要进行对抗扰动的参数
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


class CustomModel(nn.Module):
    def __init__(self, model_dir, *, dropout=0.2, pretrained=True):
        super().__init__()

        # Transformer
        self.config = AutoConfig.from_pretrained(model_dir, add_pooling_layer=False)
        if pretrained:
            self.transformer = AutoModelForMultipleChoice.from_pretrained(model_dir, config=self.config)
        else:
            self.transformer = AutoModelForMultipleChoice.from_config(self.config)

        # self.fc_dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(config.hidden_size, 1)

        # self._init_weights(self.fc, self.config)

    def _init_weights(self, module, config):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.transformer(input_ids, attention_mask, token_type_ids=token_type_ids)
        x = out['logits']  # batch_size x max_length (512) x 768

        # x = self.fc_dropout(x)
        # x = self.fc(x)

        return x


class LlmseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, is_train=False, aug_prob=0.8):
        self.df = df
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.tokenizer = tokenizer
        self.option_to_index = {option: idx for idx, option in enumerate('ABCDE')}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example = self.df.iloc[idx]
        tokenized_example = dict()

        if self.is_train and torch.rand(1) < self.aug_prob:
            prm = torch.randperm(5).numpy()
            # permed_dict_i2p = option_to_index = {i: p for i, p in enumerate(prm)}
            # permed_dict_p2i = option_to_index = {p: i for i, p in enumerate(prm)}
            # permed_dict_p2a = option_to_index = {p: a for p, a in zip(prm, 'ABCDE')}

            permed_a2e = np.array(['A', 'B', 'C', 'D', 'E'])[prm]
            permed_dict_a2p = {a: p for p, a in enumerate(permed_a2e)}
            first_sentence = [example['prompt']] * 5
            second_sentences = [example[option] for option in permed_a2e]
            tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation=False)
            tokenized_example['label'] = permed_dict_a2p[example['answer']]


        else:
            first_sentence = [example['prompt']] * 5
            second_sentences = [example[option] for option in 'ABCDE']
            tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation=False)
            tokenized_example['label'] = self.option_to_index[example['answer']]

        return tokenized_example
