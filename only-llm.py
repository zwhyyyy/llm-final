from langchain_community.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig
import torch

# 检查GPU可用性
if torch.cuda.is_available():
    print(f"检测到 {torch.cuda.device_count()} 个GPU")
    print(f"使用 GPU 1: {torch.cuda.get_device_name(1)}")
    device_id = 1
else:
    print("未检测到GPU，使用CPU")
    device_id = -1

# 初始化Qwen2.5-1.5B-Instruct模型（使用量化以减少内存占用）
print("\n正在加载Qwen2.5-1.5B-Instruct模型（这可能需要几分钟，首次运行需要下载模型）...")

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
    device=device_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": quantization_config,
    },
    pipeline_kwargs={
        "max_new_tokens": 1024,  # 增加生成长度以获得更完整的答案
        "temperature": 0.1,
        "do_sample": True,
    }
)
print("模型加载完成\n")

# 定义问题
test_questions = [
    "富血小板血浆（PRP）在治疗肌腱损伤中的效果如何？",
    "哪些骨科生物制剂可用于治疗肌肉损伤？",
    "什么是心包异位甲状旁腺腺瘤？它在原发性甲状旁腺功能亢进症的诊断和治疗中为何是一个挑战？"
]

# 顺序回答三个问题
for i, question in enumerate(test_questions, 1):
    print("=" * 80)
    print(f"问题 {i}:")
    print(question)
    print("=" * 80)
    print("\n正在生成回答...\n")

    # 使用Qwen模型直接回答问题
    answer = llm.invoke(question)

    print("=" * 80)
    print(f"回答 {i}:")
    print(answer)
    print("=" * 80)
    print("\n" + "=" * 80 + "\n")  # 添加问题之间的分隔线