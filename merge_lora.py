import argparse
import functools
import os

from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizerFast,\
    WhisperProcessor
from peft import PeftModel, PeftConfig
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("lora_model", type=str, default="output/whisper-tiny/checkpoint-best/", help="微调保存的模型路径")
add_arg('output_dir', type=str, default='models/',    help="合并模型的保存目录")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)
# 获取Whisper的基本模型
base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path, device_map={"": "cpu"},
                                                             local_files_only=args.local_files_only)

# 载入基础模型的 tokenizer
tokenizer = WhisperTokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                 local_files_only=args.local_files_only)

# 添加微调时使用的客家话语言 tokens（与 utils/reader.py 中的设置一致）
hakka_languages = {
    'hakka_sixian': '<|hakka_sixian|>',
    'hakka_hailu': '<|hakka_hailu|>',
    'hakka_dapu': '<|hakka_dapu|>',
    'hakka_raoping': '<|hakka_raoping|>',
    'hakka_zhaoan': '<|hakka_zhaoan|>',
    'hakka_nansixian': '<|hakka_nansixian|>'
}

print(f"🔧 重新创建微调时的 tokenizer 设置")
new_tokens = []
vocab = tokenizer.get_vocab()

for lang_code, token in hakka_languages.items():
    if token not in vocab:
        new_tokens.append(token)
        print(f"   ➕ 添加语言 token：{token}")

if new_tokens:
    # 添加特殊 token
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"✅ 成功添加 {len(new_tokens)} 个语言 token")

# 调整模型以适应扩展的词汇表
if len(tokenizer.get_vocab()) > base_model.config.vocab_size:
    print(f"📈 词汇表已扩展：{base_model.config.vocab_size} -> {len(tokenizer.get_vocab())}")
    print("🔧 调整模型 embedding 层大小...")
    
    # 调整模型的 embedding 层
    base_model.resize_token_embeddings(len(tokenizer.get_vocab()))
    
    # 更新模型配置
    base_model.config.vocab_size = len(tokenizer.get_vocab())
    
    print(f"✅ 模型 embedding 层已调整为 {base_model.config.vocab_size} tokens")

# 与Lora模型合并
model = PeftModel.from_pretrained(base_model, args.lora_model, local_files_only=args.local_files_only)
feature_extractor = WhisperFeatureExtractor.from_pretrained(peft_config.base_model_name_or_path,
                                                            local_files_only=args.local_files_only)
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path,
                                             local_files_only=args.local_files_only)

# 合并参数
model = model.merge_and_unload()
model.train(False)

# 保存的文件夹路径
if peft_config.base_model_name_or_path.endswith("/"):
    peft_config.base_model_name_or_path = peft_config.base_model_name_or_path[:-1]
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)

# 保存模型到指定目录中
model.save_pretrained(save_directory, max_shard_size='4GB')
feature_extractor.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
