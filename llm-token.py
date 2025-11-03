from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型及其分词器
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    dtype="auto",
    trust_remote_code=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct"
)

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# 对输入提示词进行分词
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
print(input_ids)

# 生成文本
generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=20
)
# 打印输出
print(tokenizer.decode(generation_output[0]))
