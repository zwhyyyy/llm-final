from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import BitsAndBytesConfig
import torch
import json
import time
import os
import shutil

# 从 pubmed_articles.json 加载数据
print("正在加载医学文献数据...")
with open('pubmed_articles.json', 'r', encoding='utf-8') as f:
    medical_papers = json.load(f)

print(f"成功加载 {len(medical_papers)} 篇医学文献")


# 方法2：如果JSON数据不大，也可以直接转换为Document对象
def json_to_documents(json_data):
    """将JSON数据直接转换为Document对象"""
    documents = []
    for paper in json_data:
        # 跳过没有摘要的文章
        if not paper.get('abstract') or paper.get('abstract').strip() == '':
            continue

        # 组合标题和摘要作为内容
        content = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}"

        # 创建元数据
        metadata = {
            "pmid": paper.get("pmid", ""),
            "title": paper.get("title", ""),
            "authors": ", ".join(paper.get("authors", [])),
            "journal": paper.get("journal", {}).get("title", ""),
            "pub_date": f"{paper.get('pub_date', {}).get('year', '')}-{paper.get('pub_date', {}).get('month', '')}",
            "source": "medical_literature"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


# 使用方法2（推荐，更简单）
documents = json_to_documents(medical_papers)
print(f"成功处理 {len(documents)} 篇有效文献（已过滤无摘要的文章）")

# 3. 分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_documents(documents)
print(f"成功分割 {len(texts)} 个文本块")

# 4. 创建向量库（使用BGE模型，更稳定）
print("正在加载嵌入模型...")
max_retries = 3
retry_count = 0
embeddings = None

while retry_count < max_retries:
    try:
        # 优先尝试使用 BGE 模型（更小更稳定）
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("嵌入模型加载成功")
        break
    except Exception as e:
        retry_count += 1
        if retry_count < max_retries:
            print(f"模型下载失败，正在重试 ({retry_count}/{max_retries})...")
            time.sleep(5)
        else:
            print(f"使用备用模型...")
            try:
                # 备用方案：使用更小的模型
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'}
                )
                print("备用嵌入模型加载成功")
            except Exception as e2:
                print(f"所有嵌入模型加载失败: {e2}")
                raise

# 如果数据库已存在，先删除（避免维度不匹配）
db_path = "./chroma_medical_papers_db"
if os.path.exists(db_path):
    print("检测到旧的向量数据库，正在删除以避免维度不匹配...")
    shutil.rmtree(db_path)
    print("旧数据库已删除")

# 创建Chroma向量存储（替代FAISS，Windows上更稳定）
print("正在创建向量数据库...")
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=db_path  # 持久化目录
)
print("向量数据库创建完成")

# 5. 初始化Qwen2.5-1.5B-Instruct模型（使用量化以减少内存占用）
print("正在加载Qwen2.5-1.5B-Instruct模型（这可能需要几分钟，首次运行需要下载模型）...")

# 配置4位量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    device=0 if torch.cuda.is_available() else -1,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": quantization_config,
    },
    pipeline_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.1,
        "do_sample": True,
    }
)
print("模型加载完成")

# 6. 创建检索器（增加检索数量以提高召回率）
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # 增加到5个以提高相关文档的召回
)

# 7. 创建医疗专用的提示模板（优化以避免模板内容被输出）
medical_prompt_template = """
你是一个专业的医学研究助手。你需要基于提供的医学文献内容回答用户的问题。
注意:
1. 仅根据提供的文献内容回答
2. 如果文献中没有相关信息，明确告知用户
3. 直接给出答案，不要重复问题或文献内容
4. 回答要简洁专业

参考文献:
{context}

问题：{question}
"""

PROMPT = PromptTemplate(
    template=medical_prompt_template,
    input_variables=["context", "question"]
)


# 格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 使用自定义提示的QA链（LangChain 1.0 新API）
def retrieve_and_format(question_dict):
    """检索文档并格式化"""
    question = question_dict["question"]
    docs = retriever.invoke(question)
    return {
        "context": format_docs(docs),
        "question": question
    }

qa_chain = (
    RunnablePassthrough()
    | retrieve_and_format
    | PROMPT
    | llm
    | StrOutputParser()
)


# 8. 优化的查询函数
def ask_medical_question(question):
    """提问函数 - 优化输出格式和文献展示"""
    print("\n" + "="*80)
    print(f"❓ 问题: {question}")
    print("="*80)

    # 获取相关文档（带相关性评分）
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(question, k=5)
    
    # 过滤低相关性文档（相关性分数 > 0.3）
    filtered_docs = [(doc, score) for doc, score in docs_with_scores if score > 0.3]
    
    if not filtered_docs:
        print("\n⚠️  未找到相关文献，无法回答该问题。")
        return
    
    # 使用过滤后的文档
    docs = [doc for doc, score in filtered_docs]
    
    # 使用链进行问答
    result = qa_chain.invoke({"question": question})
    
    # 清理输出（移除可能残留的模板标记和多余空白）
    result = result.strip()
    
    print(f"\n📝 回答:\n{result}")

    # 显示来源文献（带相关性评分和摘要预览）
    print("\n" + "-"*80)
    print("📚 参考文献（按相关性排序）:")
    print("-"*80)
    
    for i, (doc, score) in enumerate(filtered_docs, 1):
        title = doc.metadata.get('title', 'Unknown')
        pmid = doc.metadata.get('pmid', 'Unknown')
        journal = doc.metadata.get('journal', 'Unknown')
        pub_date = doc.metadata.get('pub_date', 'Unknown')
        
        print(f"\n[{i}] 相关性: {score:.2%}")
        print(f"    标题: {title}")
        print(f"    期刊: {journal}")
        print(f"    发表: {pub_date}")
        print(f"    PMID: {pmid}")
        
        # 显示文档片段（前150字符）
        content_preview = doc.page_content[:150].replace('\n', ' ')
        print(f"    摘要: {content_preview}...")
    
    print("\n" + "="*80)


# 9. 测试一些医学问题
test_questions = [
    "富血小板血浆（PRP）在治疗肌腱损伤中的效果如何？",
    "哪些骨科生物制剂可用于治疗肌肉损伤？",
    "什么是心包异位甲状旁腺腺瘤？它在原发性甲状旁腺功能亢进症的诊断和治疗中为何是一个挑战？"
]

for question in test_questions:
    ask_medical_question(question)

# 10. 交互式问答
print("\n=== 医疗文献RAG系统已启动 ===")
print("输入 '退出' 来结束对话")

while True:
    user_question = input("\n请输入您的医学问题: ")
    if user_question.lower() in ['退出', 'exit', 'quit']:
        break
    ask_medical_question(user_question)