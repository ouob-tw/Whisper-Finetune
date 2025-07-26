# Whisper Fine-tuning 顯存優化指南

## 顯存不足問題

當使用 `whisper-large-v3` 等大模型時，容易遇到 CUDA OUT OF MEMORY 錯誤。本文檔詳細解釋如何調整參數來優化顯存使用。

## 核心參數解釋

### 1. Batch Size 參數

#### `--per_device_train_batch_size`（訓練批次大小）

**定義**：每個 GPU 上同時處理的樣本數量

```bash
# 預設值：8
--per_device_train_batch_size=8  # 同時處理 8 個音頻樣本

# 顯存不足時調整
--per_device_train_batch_size=2  # 同時處理 2 個音頻樣本
```

**影響**：

- ✅ **降低值**：減少顯存使用，避免 OOM
- ❌ **降低值**：訓練可能不穩定，需要更多步數
- 📊 **顯存影響**：batch_size=8 vs batch_size=2 約減少 75% 顯存

#### `--per_device_eval_batch_size`（評估批次大小）

**定義**：評估時每個 GPU 上同時處理的樣本數量

```bash
# 預設值：8
--per_device_eval_batch_size=8

# 顯存不足時調整
--per_device_eval_batch_size=2
```

**特點**：

- 評估時不需要計算梯度，顯存需求較低
- 可以設定比訓練 batch size 稍大一些
- 主要影響評估速度，不影響訓練效果

### 2. 梯度累積參數

#### `--gradient_accumulation_steps`（梯度累積步數）

**定義**：累積多少個 mini-batch 的梯度後才進行一次參數更新

```bash
# 預設值：1（不累積）
--gradient_accumulation_steps=1

# 與 batch_size 降低配合使用
--gradient_accumulation_steps=4  # 累積 4 個 mini-batch
```

**工作原理**：

```
實際批次大小 = per_device_train_batch_size × gradient_accumulation_steps × GPU數量

範例：
- per_device_train_batch_size=2
- gradient_accumulation_steps=4
- GPU數量=1
- 實際批次大小 = 2 × 4 × 1 = 8
```

**優勢**：

- ✅ 保持相同的實際批次大小，訓練效果不變
- ✅ 顯存使用量只需要 mini-batch 的大小
- ✅ 避免小批次造成的訓練不穩定

## 參數調整策略

### 情況 1：輕微顯存不足

```bash
# 原始設定（可能 OOM）
--per_device_train_batch_size=8
--gradient_accumulation_steps=1

# 調整後（實際批次大小不變）
--per_device_train_batch_size=4
--gradient_accumulation_steps=2
```

### 情況 2：嚴重顯存不足

```bash
# 大幅減少顯存使用
--per_device_train_batch_size=2
--per_device_eval_batch_size=2
--gradient_accumulation_steps=4
```

### 情況 3：極限顯存不足

```bash
# 最小配置
--per_device_train_batch_size=1
--per_device_eval_batch_size=2
--gradient_accumulation_steps=8
--use_8bit=True
```

## 其他顯存優化參數

### 模型量化

#### `--use_8bit=True`

**定義**：將模型權重量化為 8 位元，大幅減少顯存

```bash
--use_8bit=True  # 約減少 50% 顯存使用
```

**效果**：

- 📉 顯存：whisper-large-v3 從 ~12GB 降到 ~6GB
- 🎯 準確度：輕微下降（通常可接受）
- ⚡ 速度：稍微變慢

### 音頻長度限制

#### `--max_audio_len`

**定義**：限制音頻的最大長度（秒）

```bash
# 預設值：30秒
--max_audio_len=30

# 減少顯存使用
--max_audio_len=20   # 約減少 33% 音頻相關顯存
--max_audio_len=15   # 約減少 50% 音頻相關顯存
```

**注意**：過短的音頻可能影響長音頻的辨識效果

### 精度設定

#### `--fp16=True`

**定義**：使用半精度浮點數（16 位元）替代全精度（32 位元）

```bash
--fp16=True  # 預設已啟用，約減少 50% 顯存
```

## 顯存需求參考表

| 模型             | 參數量 | 正常模式 | 8bit 模式 | batch_size=1 |
| ---------------- | ------ | -------- | --------- | ------------ |
| whisper-tiny     | 39M    | ~2GB     | ~1GB      | ~0.5GB       |
| whisper-base     | 74M    | ~3GB     | ~1.5GB    | ~1GB         |
| whisper-small    | 244M   | ~4GB     | ~2GB      | ~1.5GB       |
| whisper-medium   | 769M   | ~6GB     | ~3GB      | ~2GB         |
| whisper-large-v3 | 1550M  | ~12GB    | ~6GB      | ~4GB         |

## 推薦配置

### 8GB 顯存 GPU

```bash
# 推薦使用 whisper-small 或 whisper-medium
--base_model=openai/whisper-medium
--per_device_train_batch_size=4
--gradient_accumulation_steps=2
--use_8bit=True
--max_audio_len=25
```

### 12GB 顯存 GPU

```bash
# 可以使用 whisper-large-v3
--base_model=openai/whisper-large-v3
--per_device_train_batch_size=2
--gradient_accumulation_steps=4
--use_8bit=True
--max_audio_len=25
```

### 16GB+ 顯存 GPU

```bash
# 正常配置
--base_model=openai/whisper-large-v3
--per_device_train_batch_size=4
--gradient_accumulation_steps=2
--use_8bit=False
--max_audio_len=30
```

## 調試技巧

### 1. 逐步調整

```bash
# 步驟1：先啟用 8bit
--use_8bit=True

# 步驟2：如果還是 OOM，減少 batch size
--per_device_train_batch_size=4

# 步驟3：如果還是 OOM，進一步減少並增加累積
--per_device_train_batch_size=2
--gradient_accumulation_steps=4

# 步驟4：最後手段，減少音頻長度
--max_audio_len=20
```

### 2. 監控顯存使用

```bash
# 訓練前檢查顯存
nvidia-smi

# 訓練中監控
watch -n 1 nvidia-smi
```

### 3. 計算實際批次大小

```python
# 確保實際批次大小合理
actual_batch_size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus

# 範例：2 × 4 × 1 = 8（與原始 batch_size=8 相同）
```

## 常見問題

### Q: 減少 batch size 會影響訓練效果嗎？

A: 單純減少 batch size 可能會影響訓練穩定性，但配合 `gradient_accumulation_steps` 可以保持相同的實際批次大小，訓練效果基本不變。

### Q: 8bit 量化會大幅降低準確度嗎？

A: 通常準確度下降很小（<1%），但顯存節省顯著（~50%），是很好的權衡。

### Q: 如何選擇合適的模型大小？

A:

- 資料量少（<10 小時）：whisper-small 或 whisper-medium
- 資料量中等（10-50 小時）：whisper-medium 或 whisper-large-v3
- 資料量大（>50 小時）：whisper-large-v3

### Q: gradient_accumulation_steps 設定多大合適？

A: 一般 2-8 之間，確保 `actual_batch_size` 在 8-32 範圍內比較合適。

## 總結

顯存優化的核心思路是在**保持訓練效果**的前提下**減少顯存佔用**：

1. **優先調整**：`--use_8bit=True`（效果最明顯）
2. **核心策略**：減少 `per_device_train_batch_size` + 增加 `gradient_accumulation_steps`
3. **輔助調整**：適當減少 `max_audio_len`
4. **最後手段**：使用較小的模型

記住：**實際批次大小 = per_device_train_batch_size × gradient_accumulation_steps × GPU 數量**，保持這個值合理（8-32）即可維持良好的訓練效果。
