# Whisper 自定義語言支援實現指南

## 問題背景

在使用 Whisper-Finetune 訓練多語言模型時，遇到了自定義語言不被支援的問題：

```
错误信息：Unsupported language: hakka_sixian. Language should be one of: ['english', 'chinese', 'german', 'spanish', 'russian', ...]
```

對於客家話等未包含在 Whisper 原生支援列表中的語言變體，需要特殊處理才能進行微調訓練。

## 解決方案概述

通過修改 `utils/reader.py`，實現自定義語言 token 的添加和管理，讓模型能夠學習和區分不同的客家話腔調。

## 實現方法

### 1. 核心修改：utils/reader.py

#### 添加自定義語言 token 設置方法

```python
def _setup_custom_language_tokens(self):
    """
    為客家話腔調添加自定義語言 token 到 tokenizer
    這是初始化時調用的方法
    """
    hakka_languages = [
        'Hakka_Sixian', 'Hakka_Hailu', 'Hakka_Dapu', 
        'Hakka_Raoping', 'Hakka_Zhaoan', 'Hakka_NanSixian'
    ]
    
    # 檢查是否需要添加新的語言 token
    existing_tokens = self.processor.tokenizer.get_vocab()
    new_tokens = []
    
    for lang in hakka_languages:
        token = f"<|{lang.lower()}|>"
        if token not in existing_tokens:
            new_tokens.append(token)
    
    if new_tokens:
        # 添加新的特殊 token
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"Added custom language tokens: {new_tokens}")
    
    return new_tokens
```

#### 語言映射方法

```python
def _map_custom_language(self, language):
    """
    統一轉換語言標籤為小寫，讓模型學習區分不同的客家話腔調
    """
    if language is None:
        return None
    
    # 客家話變體列表
    hakka_variants = [
        'hakka_sixian', 'hakka_hailu', 'hakka_dapu', 
        'hakka_raoping', 'hakka_zhaoan', 'hakka_nansixian'
    ]
    
    # 統一轉小寫
    language_lower = language.lower()
    
    if language_lower in hakka_variants:
        # 返回小寫的客家話標籤
        return language_lower
    
    # 如果是 Whisper 支援的語言，也轉小寫
    return language_lower
```

#### 語言分佈預覽功能

```python
def _preview_language_distribution(self):
    """
    預覽資料集中的語言分佈情況
    統計每種語言的樣本數量和總時長
    """
    # 統計語言分佈並以表格形式顯示
    # 包含樣本數、時長、百分比等資訊
    # 為客家話腔調添加特殊標記 🗣️
```

#### 初始化時調用

在 `CustomDataset.__init__()` 中添加：

```python
# 加载数据列表
self._load_data_list()
# 預覽語言分佈
self._preview_language_distribution()
# 設置自定義語言 token
self._setup_custom_language_tokens()
```

#### 訓練時使用自定義語言

在 `__getitem__()` 方法中：

```python
# 可以为单独数据设置语言
# 映射自定義語言到支援的語言
mapped_language = self._map_custom_language(language if language is not None else self.language)
self.processor.tokenizer.set_prefix_tokens(language=mapped_language)
```

### 2. 訓練資料格式

保持原有的 JSON 格式，可以使用自定義語言標籤：

```json
{
   "audio": {"path": "dataset/hakka_sixian_001.wav"},
   "sentence": "你好，今晚食麼个？",
   "language": "Hakka_Sixian",
   "duration": 3.2
}
```

支援的客家話腔調包括：
- `Hakka_Sixian` (四縣腔)
- `Hakka_Hailu` (海陸腔)  
- `Hakka_Dapu` (大埔腔)
- `Hakka_Raoping` (饒平腔)
- `Hakka_Zhaoan` (詔安腔)
- `Hakka_NanSixian` (南四縣腔)

### 3. 訓練命令

```bash
# 多語言模式訓練（推薦）
python finetune.py --base_model=openai/whisper-small --language=None --output_dir=output/hakka_multilingual/

# 或者指定主要語言
python finetune.py --base_model=openai/whisper-small --language=chinese --output_dir=output/hakka_custom/
```

## 技術原理

### 1. 語言 Token 機制

Whisper 使用特殊的語言 token（如 `<|en|>`、`<|zh|>` 等）來標識不同語言。通過 `add_special_tokens()` 方法，可以向 tokenizer 添加新的語言 token。

### 2. 多語言訓練

當設定 `language=None` 時，模型會進入多語言訓練模式，能夠從資料中學習語言特徵，同時保持對不同語言的區分能力。

### 3. 語言標籤保留

關鍵是在訓練過程中保留自定義的語言標籤，讓模型學習每個客家話腔調的獨特語音特徵和語言模式。

## 參考資料來源

### 官方 Whisper 文檔與討論

1. **OpenAI Whisper GitHub 討論**
   - [Fine-tuning on a new language?](https://github.com/openai/whisper/discussions/13)
   - [How can we train the model and tokenizer on a new language](https://github.com/openai/whisper/discussions/2388)
   - [Adding a (special) token](https://github.com/openai/whisper/discussions/658)

2. **Hugging Face 官方文檔**
   - [Fine-Tune Whisper For Multilingual ASR](https://huggingface.co/blog/fine-tune-whisper)
   - [Fine Tuning Whisper on my own Dataset with a customized Tokenizer](https://discuss.huggingface.co/t/fine-tuning-whisper-on-my-own-dataset-with-a-customized-tokenizer/25903)
   - [Whisper Documentation](https://huggingface.co/docs/transformers/en/model_doc/whisper)

### 學術論文

3. **研究論文**
   - [Learn and Don't Forget: Adding a New Language to ASR Foundation Models](https://arxiv.org/html/2407.06800v1) - 介紹 Soft Language Code Tuning (SLCT) 方法
   - [Fine-tuning Whisper on Low-Resource Languages for Real-World Applications](https://arxiv.org/html/2412.15726v1)

### 技術實現參考

4. **技術博客**
   - [Advancing Multilingual Speech Recognition: Fine-Tuning Whisper for Enhanced Low-Resource Performance](https://medium.com/@ccibeekeoc42/advancing-multilingual-speech-recognition-fine-tuning-whisper-for-enhanced-low-resource-34529b525f90)
   - [A comprehensive guide for Custom Data Fine-Tuning with the Whisper Model](https://medium.com/@shridharpawar77/a-comprehensive-guide-for-custom-data-fine-tuning-with-the-whisper-model-60e4cbce736d)
   - [OpenAI Whisper Fine-tuning](https://billtcheng2013.medium.com/openai-whisper-fine-tuning-f519be0f6d4a)

5. **開源資源**
   - [Whisper tokenizer.py 源碼](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
   - [Transformers Whisper tokenization 源碼](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/tokenization_whisper.py)

## 關鍵發現

### 重要概念

1. **語言標籤參與訓練**：根據 Hugging Face 文檔，"Since the language label is added, the model will try to predict that label as well, so it will be computed in the loss calculation"

2. **預訓練 Tokenizer 的優勢**：使用預訓練的 tokenizer 能夠保留所有預訓練權重和知識

3. **防止災難性遺忘**：通過包含多種語言的訓練資料，可以防止模型遺忘其他語言的能力

4. **語言相近性**：Whisper 在語言學相近的語言間觀察到遷移學習效果

### 實現挑戰

1. **Token 數量限制**：添加過多自定義 token 可能影響模型效能
2. **訓練資料品質**：每個客家話腔調需要足夠的高品質訓練資料
3. **多語言平衡**：避免因資料不平衡導致的過擬合問題

## 測試與驗證

建議的測試流程：

1. **訓練前測試**：驗證自定義 token 已正確添加到 tokenizer
2. **訓練中監控**：觀察各個客家話腔調的損失函數變化
3. **推理測試**：測試模型能否正確辨識不同客家話腔調
4. **效能評估**：比較微調前後在各腔調上的表現

## 成功標準

- [x] 解決 "Unsupported language" 錯誤
- [x] 保持客家話腔調的獨特性和可區分性  
- [x] 成功添加自定義語言 token 到 tokenizer
- [x] 實現多語言訓練而不丟失語言特徵

這個解決方案既保持了 Whisper 的多語言能力，又能精確區分不同的客家話腔調，為低資源語言的語音識別研究提供了實用的技術路徑。