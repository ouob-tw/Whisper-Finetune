import os
import shutil
import torch
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple


# 保存模型时的回调函数
class SavePeftModelCallback(TrainerCallback):
    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs, ):
        if args.local_rank == 0 or args.local_rank == -1:
            # 保存效果最好的模型
            best_checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-best")
            # 因为只保存最新5个检查点，所以要确保不是之前的检查点
            if state.best_model_checkpoint is not None and os.path.exists(state.best_model_checkpoint):
                if os.path.exists(best_checkpoint_folder):
                    shutil.rmtree(best_checkpoint_folder)
                shutil.copytree(state.best_model_checkpoint, best_checkpoint_folder)
            print(f"效果最好的检查点为：{state.best_model_checkpoint}，评估结果为：{state.best_metric}")
        return control


# CER 評估回調函數
class CEREvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor, eval_loss_threshold=0.3, remove_pun=True, to_simple=True):
        """
        Args:
            eval_dataset: 評估數據集
            processor: Whisper processor
            eval_loss_threshold: 只有當 eval_loss 小於此值時才進行 CER 評估
            remove_pun: 是否移除標點符號
            to_simple: 是否轉換為簡體中文
        """
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.eval_loss_threshold = eval_loss_threshold
        self.remove_pun = remove_pun
        self.to_simple = to_simple
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        self.cer_log_file = None  # 將在 on_train_begin 中設定
    
    def on_train_begin(self, args, state, control, **kwargs):
        """訓練開始時設置 CER 日誌文件"""
        self.cer_log_file = os.path.join(args.output_dir, "cer_evaluation_log.json")
        
        # 初始化 CER 日誌文件
        initial_log = {
            "eval_loss_threshold": self.eval_loss_threshold,
            "remove_pun": self.remove_pun, 
            "to_simple": self.to_simple,
            "start_time": datetime.now().isoformat(),
            "evaluations": []
        }
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(self.cer_log_file, 'w', encoding='utf-8') as f:
            json.dump(initial_log, f, indent=2, ensure_ascii=False)
            
        print(f"📝 CER 評估日誌將保存到: {self.cer_log_file}")
        print(f"🎯 評估條件: eval_loss < {self.eval_loss_threshold}")
        print(f"📊 根據 eval_steps 設定，每 {getattr(args, 'eval_steps', 'N/A')} 步評估一次")
        
    def on_evaluate(self, args, state, control, logs=None, model=None, **kwargs):
        """在每次評估後檢查是否需要計算 CER"""
        if logs is None:
            return control
            
        eval_loss = logs.get('eval_loss', float('inf'))
        
        # 只有當 eval_loss 小於閾值時才進行 CER 評估
        if eval_loss < self.eval_loss_threshold:
            print(f"\n🎯 eval_loss ({eval_loss:.4f}) < {self.eval_loss_threshold}，開始計算 CER...")
            
            try:
                cer_result = self._compute_cer(model, args)
                logs['eval_cer'] = cer_result
                print(f"📊 CER 結果: {cer_result:.4f}")
                
                # 保存 CER 結果到日誌文件
                self._save_cer_result(state, eval_loss, cer_result)
                
                # 將 CER 結果記錄到 state 中（用於最佳模型選擇）
                if not hasattr(state, 'cer_history'):
                    state.cer_history = []
                state.cer_history.append({
                    'step': state.global_step,
                    'eval_loss': eval_loss,
                    'cer': cer_result
                })
                
            except Exception as e:
                print(f"❌ CER 計算失敗: {e}")
                # 即使失敗也要記錄
                self._save_cer_result(state, eval_loss, None, error=str(e))
        else:
            print(f"⏭️ eval_loss ({eval_loss:.4f}) >= {self.eval_loss_threshold}，跳過 CER 評估")
            # 記錄跳過的評估
            self._save_cer_result(state, eval_loss, None, skipped=True)
            
        return control
    
    def _save_cer_result(self, state, eval_loss, cer_result, error=None, skipped=False):
        """保存 CER 結果到日誌文件"""
        if self.cer_log_file is None:
            return
            
        try:
            # 讀取現有的日誌
            with open(self.cer_log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            # 準備新的評估記錄
            evaluation_record = {
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": eval_loss,
                "timestamp": datetime.now().isoformat(),
            }
            
            if skipped:
                evaluation_record["status"] = "skipped"
                evaluation_record["reason"] = f"eval_loss ({eval_loss:.4f}) >= threshold ({self.eval_loss_threshold})"
            elif error:
                evaluation_record["status"] = "error"
                evaluation_record["error"] = error
            else:
                evaluation_record["status"] = "completed"
                evaluation_record["cer"] = cer_result
                
            # 添加到評估列表
            log_data["evaluations"].append(evaluation_record)
            
            # 保存更新的日誌
            with open(self.cer_log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ 保存 CER 日誌時出錯: {e}")
    
    def _compute_cer(self, model, args):
        """計算 CER"""
        model.eval()
        
        # 創建 DataLoader
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=8,  # 使用較小的 batch size 避免記憶體問題
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=2
        )
        
        # 初始化 CER 計算器
        metric = evaluate.load("cer")
        
        print(f"🔄 開始 CER 評估，共 {len(eval_dataloader)} 個批次...")
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(eval_dataloader, desc="CER評估")):
                with torch.autocast(device_type="cuda"):
                    # 保持在 GPU 上進行運算
                    generated_tokens = model.generate(
                        input_features=batch["input_features"].cuda(),
                        decoder_input_ids=batch["labels"][:, :4].cuda(),
                        max_new_tokens=255)
                    labels = batch["labels"]
                    # 在 GPU 上處理 labels
                    labels = torch.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
                    
                    # 只在需要 decode 時才移到 CPU
                    generated_tokens = generated_tokens.cpu().numpy()
                    labels = labels.cpu().numpy()
                    
                    # 將預測和實際的 token 轉換為文本
                    decoded_preds = self.processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    decoded_labels = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                    
                    # 刪除標點符號
                    if self.remove_pun:
                        decoded_preds = remove_punctuation(decoded_preds)
                        decoded_labels = remove_punctuation(decoded_labels)
                    
                    # 轉換為簡體中文
                    if self.to_simple:
                        decoded_preds = to_simple(decoded_preds)
                        decoded_labels = to_simple(decoded_labels)
                    
                    metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                
                # 只評估前幾個批次以節省時間
                if step >= 20:  # 限制評估批次數量
                    print(f"⚡ 限制評估批次數量，只評估前 {step+1} 個批次")
                    break
        
        # 計算並返回 CER
        cer_result = metric.compute()
        model.train()  # 切回訓練模式
        return cer_result
