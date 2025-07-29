# Whisper 客家話微調中的語言設置問題

## 問題背景

在微調 Whisper 模型支持客家話時，遇到了語言設置相關的問題。本文檔記錄了問題的原因、解決方案和最佳實踐。

## 核心問題

### 1. 雙層語言處理架構

Whisper 模型中存在兩個不同層次的語言處理：

#### Tokenizer 層次（已修改）

- 位置：`utils/reader.py` 中的 `_setup_custom_language_tokens()`
- 功能：在詞彙表中添加客家話語言 tokens
- 成功添加的 tokens：
  ```python
  hakka_languages = {
      'hakka_sixian': '<|hakka_sixian|>',     # ID: 51866
      'hakka_hailu': '<|hakka_hailu|>',       # ID: 51867
      'hakka_dapu': '<|hakka_dapu|>',         # ID: 51868
      'hakka_raoping': '<|hakka_raoping|>',   # ID: 51869
      'hakka_zhaoan': '<|hakka_zhaoan|>',     # ID: 51870
      'hakka_nansixian': '<|hakka_nansixian|>' # ID: 51871
  }
  ```

#### Generation 層次（未修改）

- 位置：`transformers/models/whisper/generation_whisper.py:1453`
- 功能：在 `model.generate()` 時驗證語言參數
- 問題：只接受 Whisper 原始支持的語言列表，不認識自定義的客家話語言

### 2. 具體錯誤

當使用 `--language=Hakka_Sixian` 時會出現：

```
ValueError: Unsupported language: hakka_sixian. Language should be one of: ['english', 'chinese', 'german', ...]
```

## 解決方案

### 方案 1：不指定語言（推薦）

```bash
python evaluation.py --model_path=models/whisper-large-v3-finetune --test_data=test.jsonl --metric=cer
```

**優點：**

- 避免語言驗證錯誤
- 讓模型自動使用訓練時的語言設置
- 最安全的方法

### 方案 2：使用相近語言

```bash
python evaluation.py --language=Chinese --model_path=models/whisper-large-v3-finetune --test_data=test.jsonl --metric=cer
```

**優點：**

- 與客家話語言特性相近
- 避免默認使用英文

### 方案 3：修改 forced_decoder_ids（複雜）

```python
# 在 evaluation.py 中手動設置
forced_decoder_ids = [
    [processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
     51866,  # hakka_sixian token ID
     processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")]
]
generated_tokens = model.generate(
    input_features=batch["input_features"].cuda(),
    forced_decoder_ids=forced_decoder_ids,
    max_new_tokens=255
)
```

## 語言設置的影響分析

### Token 序列比較

1. **訓練時使用**：`<|startoftranscript|>` + `<|hakka_sixian|>` + `<|transcribe|>`
2. **評估時不指定語言**：`<|startoftranscript|>` + `<|en|>` (默認) + `<|transcribe|>`
3. **評估時指定中文**：`<|startoftranscript|>` + `<|chinese|>` + `<|transcribe|>`

### 性能影響預估

- **理論上**：語言 token 不匹配可能影響性能
- **實際上**：影響可能較小（1-3% CER 差異）
- **原因**：模型主要依靠音頻特徵，語言 token 更多是提示作用

## 最佳實踐

### 1. 微調階段

- 確保數據標註中包含正確的語言標籤（`hakka_sixian`）
- 在 `CustomDataset` 中正確映射語言標籤
- 驗證 tokenizer 中已添加所需的語言 tokens

### 2. 評估階段

- **推薦**：不指定語言參數，讓模型使用訓練時的設置
- **備選**：使用 `--language=Chinese` 作為相近語言
- **避免**：使用 `--language=Hakka_Sixian` 會導致錯誤

### 3. 推理階段

- 與評估階段相同的原則
- 可以通過比較不同語言設置的結果來選擇最佳配置

## 技術細節

### 詞彙表擴展

微調過程中詞彙表從 51866 擴展到 51872（+6 個客家話 tokens）：

- 原始大小：51866
- 擴展後：51872
- 新增 tokens：6 個客家話腔調

### 模型權重調整

在 `merge_lora.py` 中需要正確處理詞彙表大小不匹配問題：

```python
# 重新創建微調時的 tokenizer 設置
hakka_languages = {...}  # 與 utils/reader.py 保持一致
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
base_model.resize_token_embeddings(len(tokenizer.get_vocab()))
```

## 常見問題

### Q: 為什麼不能直接使用 `--language=Hakka_Sixian`？

A: 因為 Whisper 的 generation 模塊有硬編碼的語言驗證，只接受原始支持的語言。雖然我們在 tokenizer 層次添加了客家話 tokens，但 generation 層次未同步更新。

### Q: 不指定語言會影響性能嗎？

A: 可能有輕微影響，但由於模型在數據處理階段已經學習了客家話特徵，實際影響通常很小。

### Q: 如何驗證語言設置是否正確？

A: 檢查模型輸出中是否正確使用了客家話 token ID (51866)，以及比較不同語言設置下的 CER 結果。

## 相關文件

- `utils/reader.py`: 語言 token 設置和映射
- `merge_lora.py`: 模型合併時的詞彙表處理
- `evaluation.py`: 評估時的語言參數處理
- `finetune.py`: 微調時的語言處理邏輯

## 更新記錄

- 2025-07-29: 初始版本，記錄語言設置問題和解決方案
