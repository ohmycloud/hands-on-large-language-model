from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    base_url='https://api.apiyi.com/v1'
)

# 定义 Prompt
# 提示：你的 template 中包含了特殊 tokens (<s><|user|>...)，通常用于本地模型或原始补全模型。
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title.<|end|>
<|assistant|>"""

title_prompt = PromptTemplate(template=template, input_variables=["summary"])

# 定义输出解析器
# 直接将 LLM 的输出对象 (AIMessage) 转为纯字符串，相当于以前的 output_key="title"
output_parser = StrOutputParser()

# 构建链 (使用管道符 | )
# 流程：输入 -> 提示词 -> 模型 -> 字符串解析
chain = title_prompt | llm | output_parser

# 调用链 (使用 invoke)
summary_text = "a girl that find her boyfriend"
title = chain.invoke({"summary": summary_text})

print(title)
print("------------end title-----------")

# 角色描述
# 使用故事梗概和标题创建一个链式架构来生成角色描述
template = """<s><|user|>
Describe the main character of a story about {summary} with the title {title}.
Use only two sentences.<|end|>
<|assistant|>"""

character_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title"]
)
chain = character_prompt | llm | output_parser
character_text = "a brave knight who specifically fights data bugs"
character = chain.invoke({"summary": character_text, "title": title})
print(character)
print("------------end character-----------")

# 使用故事梗概、标题和角色描述创建一个链式架构来生成故事
template = """<s><|user|>
Create a story about {summary} with the title {title}. The main character is:
{character}. Only return the story and it cannot be longer than one paragraph.
<|end|>
<|assistant|>"""
story_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title", "character"]
)

chain = story_prompt | llm | output_parser
summary_text = "a girl that find her favorite music"
story = chain.invoke({"summary": character_text, "title": title, "character": character})
print(story)
print("------------end story-----------")
