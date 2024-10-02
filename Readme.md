# 本项目是基于用户上传的PDF做检索的问答系统

## 项目实现步骤：
- 大模型选取与部署：本项目使用的是最近刚出的qwen2.5-7B,我是部署在ollama上通过openai接口调用的
- RAG功能实现：  
1. 将文档分成无数个512k大小的chunk
2. 使用大模型评估用户输入的prompt与每个chunk是否有关系，有就提取出来，没有就输出None。
3. 通过多线程调用大模型评估
4. 预处理一下输入prompt,提取prompt中的问题，指令和关键词，然后通过BM250算法在评估好的chunk中提取相关信息
5. 传给大模型进行最终回答

## 最终结果：
![test](/image/01.png)

## 运行：
`./ollama_start.sh`  
`python qwen2.5_pdf.py`
