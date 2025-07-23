"""
Tokenizer测试脚本，验证BERT分词器功能
"""
from transformers import BertTokenizer

# 初始化分词器
def initialize_tokenizer():
    """
    初始化BERT中文分词器
    :return: 初始化好的分词器实例
    """
    return BertTokenizer.from_pretrained(
        "model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
    )

# 测试分词器功能
def test_tokenizer(token):
    """
    测试分词器编码和解码功能
    :param token: 分词器实例
    """
    # 测试文本
    sents = [
        "白日依山尽，",  
        "价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所"
    ]

    # 批量编码
    print("编码结果:")
    encoded = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=15,
        padding="max_length",
        return_tensors=None,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_length=True
    )

    # 打印编码结果
    for k, v in encoded.items():
        print(k, ":", v)

    # 解码测试
    print("\n解码结果:")
    for i in range(len(sents)):
        print(f"原文{i+1}: {sents[i]}")
        print(f"解码{i+1}: {token.decode(encoded['input_ids'][i])}")

if __name__ == '__main__':
    tokenizer = initialize_tokenizer()
    test_tokenizer(tokenizer)