import json
import re
from typing import List

from openai import OpenAI
import gradio as gr
import PyPDF2
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
import nltk
import jieba
from langdetect import detect
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

#读取pdf
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

#pdf分段成512k大小的上下文
def split_into_chunks(text, chunk_size=512000):
    words = word_tokenize(text)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        if len(words) <= chunk_size:
            chunk = ' '.join(words[:])
        chunks.append(chunk)
        return chunks
    return chunks

#评估每段与输入的关系
def evaluate_relevance(query, chunk):
    prompt = f"""
    评估以下文本段落与用户查询之间的相关性：
    
    文本段落：{chunk}
    
    用户查询：{query}
    
    请根据以下标准进行评估：
    1. 完全相关：文本直接回答或涉及查询的主题
    2. 部分相关：文本包含与查询相关的信息，但不是直接答案
    3. 不相关：文本与查询无关,仅输出None
    
    仅返回评估结果（完全相关/部分相关/不相关）和相关句子（如果有）。不要解释你的推理过程。
    """
    response = response_from_qwen(prompt)
    
    # 解析响应
    if "完全相关" in response or "部分相关" in response:
        # 提取相关句子（假设相关句子在评估结果之后）
        relevant_part = response.split("\n", 1)[-1].strip()
        return relevant_part if relevant_part else chunk
    else:
        return None


#多线程调用评估函数
def parallel_relevance_evaluation(query,chunks):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_relevance, chunks, [query]*len(chunks)))
    return [result for result in results if result != "None"]

#调用大模型处理query提取关键词
def process_query(query):
    prompt = f"""
        请将用户查询中的指令和信息部分分开。
        然后，从信息部分推断出多语言关键词。
        以JSON格式输出结果。
        用户查询: {query}
        输出格式:
        {{
            "information": ["..."],
            "instruction": ["...", "...", "..."],
            "keywords_en": ["...", "...", "..."],
            "keywords_zh": ["...", "...", "..."]
        }}
        """
    try:
        response = response_from_qwen(prompt)
    except json.JSONDecodeError:
        print("API返回的不是有效的JSON")
        return None
    return response

#提取与关键词相关的最大8k的上下文
def auto_tokenize(text: str) -> List[str]:
    """根据自动检测的语言对文本进行分词"""
    try:
        lang = detect(text)
    except:
        # 如果检测失败，默认使用英文分词
        lang = 'en'

    if lang == 'zh-cn' or lang == 'zh-tw':
        return list(jieba.cut(text))
    else:
        return word_tokenize(text.lower())


def count_chars(text: str) -> int:
    """计算文本的字符数，中文字符计为2，其他计为1"""
    return sum(2 if '\u4e00' <= char <= '\u9fff' else 1 for char in text)


def retrieve_with_bm25(key_words, relevant_sentences, max_chars = 8000):
    # 对句子进行分词
    tokenized_sentences = [auto_tokenize(sentence) for sentence in relevant_sentences]

    # 创建BM25模型
    bm25 = BM25Okapi(tokenized_sentences)

    # 计算每个关键词的得分
    all_scores = [bm25.get_scores(auto_tokenize(word)) for word in key_words]

    # 计算平均得分
    avg_scores = [sum(scores) / len(scores) for scores in zip(*all_scores)]

    # 根据得分对句子进行排序
    ranked_sentences = [sent for _, sent in sorted(zip(avg_scores, relevant_sentences), reverse=True)]
    
    # 提取前N个句子，直到达到指定字符数或所有句子都被使用
    context = ""
    current_chars = 0
    for sentence in ranked_sentences:
        sentence_chars = count_chars(sentence)
        if current_chars + sentence_chars <= max_chars:
            context += sentence + " "
            current_chars += sentence_chars + 1  # +1 for the space
        else:
            break
    return context.strip()

def generate_final_answer(query, context, instructions,information):
    prompt = f"""
    根据以下上下文回答下面的问题：{information}。
    遵循这些指令: {instructions}
    如果上下文没有提供足够的信息，请说明。

    上下文: {context}

    问题: {query}

    回答:
    """
    response = response_from_qwen(prompt)
    return response


#调用openai接口实现qwen2.5的回复
def response_from_qwen(prompt) -> str:
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        api_key='EMPTY',
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model='qwen2.5:7b',
        temperature = 0.5
    )

    return chat_completion.choices[0].message.content

# Gradio 接口函数
def process_pdf_query(pdf_file, query):
    if pdf_file is None:
        return "请先上传 PDF 文件。", "{}", ""
    
    if not query.strip():
        return "请输入有效的问题。", "{}", ""

    try:
        # 处理上传的文件
        if isinstance(pdf_file, str): 
            with open(pdf_file, 'rb') as file:
                pdf_bytes = file.read()
        else:  
            pdf_bytes = pdf_file.read()
        
        # 使用BytesIO创建一个内存中的文件对象
        pdf_file_obj = BytesIO(pdf_bytes)
        
        # 提取文本
        text = extract_text_from_pdf(pdf_file_obj)
        
        # 分割文本
        chunks = split_into_chunks(text)
        print(f'fafagasgatgreaghaheagher:{chunks}')
        print(len(chunks))
        # 处理查询
        processed_query = process_query(query)
        print(f"Processed query result: {processed_query}")  # 调试信息
        if processed_query:
            try:
                # 移除可能存在的 "```json" 和 "```"
                processed_query = processed_query.strip()
                if processed_query.startswith("```json"):
                    processed_query = processed_query[7:]
                if processed_query.endswith("```"):
                    processed_query = processed_query[:-3]
                
                query_data = json.loads(processed_query.strip())
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"尝试解析的字符串: {processed_query}")
                return f"查询处理错误: 无效的 JSON - {e}", "{}", ""
            information = query_data.get("information", [])
            instruction = query_data.get("instruction", [])
            keywords_en = query_data.get("keywords_en", [])
            keywords_zh = query_data.get("keywords_zh", [])

            # 合并关键词
            keywords = keywords_en + keywords_zh

            # 评估相关性
            relevant_sentences = parallel_relevance_evaluation(query,chunks)
            print(f'大模型评估后的相关句子是：{relevant_sentences}')
            if not relevant_sentences:
                # 处理没有找到相关句子的情况
                context = "未找到与查询直接相关的内容。"
                answer = generate_final_answer(query, context, instruction, information)
                query_data_json = json.dumps(query_data, ensure_ascii=False, indent=2) if query_data else "{}"
                return answer, query_data_json, [None]

            # 使用BM25检索相关上下文
            context = retrieve_with_bm25(keywords, relevant_sentences)
            print(f'context:{context}')

            # 生成最终答案
            answer = generate_final_answer(query, context, instruction, information)

            query_data_json = json.dumps(query_data, ensure_ascii=False, indent=2) if query_data else "{}"

            return answer, query_data_json, context
        else:
            return "无法处理查询。请重试。", "{}", ""
    except Exception as e:
        return f"处理过程中出错: {str(e)}", "{}", ""
        
# 创建 Gradio 接口
iface = gr.Interface(
    fn=process_pdf_query,
    inputs=[
        gr.File(label="上传PDF文件"),
        gr.Textbox(label="输入你的问题")
    ],
    outputs=[
        gr.Textbox(label="回答"),
        gr.JSON(label="查询处理详情"),
        gr.Textbox(label="相关上下文")
    ],
    title="PDF RAG 问答系统",
    description="上传一个PDF文件并输入你的问题，系统将基于PDF内容回答你的问题。"
)

# 运行 Gradio 应用
if __name__ == "__main__":
    iface.launch(server_port=6006)


