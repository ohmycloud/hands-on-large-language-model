from transformers import AutoModel, AutoTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# 加载语言模型
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

#对句子进行分词
tokens = tokenizer("Hello World", return_tensors='pt')

# 处理词元
output = model(**tokens)[0]

print(output.shape)
print(output)

for token in tokens['input_ids'][0]:
    print(tokenizer.decode(token))
