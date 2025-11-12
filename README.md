# 图解大模型

```bash
# 比较分词器
uv run python compare-tokenizer.py --tokenizer-name=bert-base-cased
uv run python compare-tokenizer.py --tokenizer-name=bert-base-uncased --sentence="Rakulang Rocks"

# 大模型参数
uv run python llm_config.py
```
