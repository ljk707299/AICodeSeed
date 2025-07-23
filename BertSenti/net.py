import torch
from transformers import BertModel

# 设置计算设备(优先使用GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前计算设备: {DEVICE}")

# 加载预训练的中文BERT模型
pretrained = BertModel.from_pretrained(
    "model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
).to(DEVICE)

class Model(torch.nn.Module):
    """
    情感分析模型，基于BERT的微调模型
    """
    def __init__(self):
        super().__init__()
        # 在BERT输出基础上添加全连接层进行二分类
        self.fc = torch.nn.Linear(768, 2)  # 768是BERT隐藏层维度，2是分类数

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        前向传播
        :param input_ids: 输入token IDs
        :param attention_mask: 注意力掩码
        :param token_type_ids: token类型IDs
        :return: 分类logits
        """
        # 冻结BERT参数，只训练全连接层
        with torch.no_grad():
            out = pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        # 使用[CLS]位置的输出进行分类
        out = self.fc(out.last_hidden_state[:, 0])
        return out