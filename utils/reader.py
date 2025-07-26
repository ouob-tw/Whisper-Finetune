import json
import os
import random
import sys
from typing import List

import librosa
import numpy as np
import soundfile
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.binary import DatasetReader


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 language=None,
                 timestamps=False,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30,
                 min_sentence=1,
                 max_sentence=200,
                 augment_config_path=None):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            mono: 是否将音频转换成单通道，这个必须是True
            language: 微调数据的语言
            timestamps: 微调时是否使用时间戳
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            min_sentence: 微调时最少的句子字数，默认1
            max_sentence: 微调时最多句子字数，默认200
            augment_config_path: 数据增强配置参数文件路径
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        assert min_sentence >= 1, f"min_sentence不能小于1，当前为：{min_sentence}"
        assert max_sentence <= 200, f"max_sentence不能大于200，当前为：{max_sentence}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.data_list_path = data_list_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sentence = min_sentence
        self.max_sentence = max_sentence
        self.vocab = self.processor.tokenizer.get_vocab()
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        if '<|nospeech|>' in self.vocab.keys():
            self.nospeech = self.vocab['<|nospeech|>']
            self.timestamp_begin = None
        else:
            # 兼容旧模型
            self.nospeech = self.vocab['<|nocaptions|>']
            self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()
        # 預覽語言分佈
        self._preview_language_distribution()
        # 設置自定義語言 token
        self._setup_custom_language_tokens()
        # 数据增强配置参数
        self.augment_configs = None
        self.noises_path = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)

    # 加载数据列表
    def _load_data_list(self):
        if self.data_list_path.endswith(".header"):
            # 获取二进制的数据列表
            self.dataset_reader = DatasetReader(data_header_path=self.data_list_path,
                                                min_duration=self.min_duration,
                                                max_duration=self.max_duration)
            self.data_list = self.dataset_reader.get_keys()
        else:
            # 获取数据列表
            with open(self.data_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in tqdm(lines, desc='读取数据列表'):
                if isinstance(line, str):
                    line = json.loads(line)
                if not isinstance(line, dict): continue
                # 跳过超出长度限制的音频
                if line["duration"] < self.min_duration:
                    continue
                if self.max_duration != -1 and line["duration"] > self.max_duration:
                    continue
                # 跳过超出句子字数限制的音频
                if 'sentence' in line.keys():
                    if len(line["sentence"]) < self.min_sentence or len(line["sentence"]) > self.max_sentence:
                        continue
                else:
                    sentence_len = 0
                    for s in line["sentences"]:
                        sentence_len += len(s['text'])
                    if sentence_len < self.min_sentence or sentence_len > self.max_sentence:
                        continue
                self.data_list.append(dict(line))

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            # 分割读取音频
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        # 转成单通道
        if self.mono:
            sample = librosa.to_mono(sample)
        # 数据增强
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)
        # 重采样
        if self.sample_rate != sample_rate:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
        return sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            if self.timestamp_begin is None:
                start = self.vocab[f'<|{start:.2f}|>']
            else:
                start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            if self.timestamp_begin is None:
                end = self.vocab[f'<|{end:.2f}|>']
            else:
                end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def _preview_language_distribution(self):
        """
        預覽資料集中的語言分佈情況
        統計每種語言的樣本數量和總時長
        """
        language_stats = {}
        total_samples = 0
        total_duration = 0.0
        
        print("\n" + "="*60)
        print("📊 資料集語言分佈預覽")
        print("="*60)
        
        for data in self.data_list:
            # 獲取語言標籤（如果沒有則使用全域設定）
            language = data.get('language', self.language)
            if language is None:
                language = 'Unknown'
            
            # 標準化語言名稱（轉小寫）
            language = language.lower()
            
            # 統計
            if language not in language_stats:
                language_stats[language] = {
                    'count': 0,
                    'duration': 0.0
                }
            
            language_stats[language]['count'] += 1
            language_stats[language]['duration'] += data.get('duration', 0.0)
            
            total_samples += 1
            total_duration += data.get('duration', 0.0)
        
        # 顯示統計結果
        print(f"📋 總樣本數：{total_samples:,}")
        print(f"⏱️  總時長：{total_duration:.2f} 小時 ({total_duration*60:.1f} 分鐘)")
        print(f"🌍 語言種類：{len(language_stats)} 種")
        print("\n📈 各語言詳細統計：")
        print("-" * 60)
        print(f"{'語言':<20} {'樣本數':<10} {'時長(小時)':<12} {'百分比':<8}")
        print("-" * 60)
        
        # 按樣本數排序顯示
        sorted_languages = sorted(language_stats.items(), 
                                key=lambda x: x[1]['count'], 
                                reverse=True)
        
        for language, stats in sorted_languages:
            count = stats['count']
            duration = stats['duration']
            percentage = (count / total_samples) * 100
            
            # 為客家話腔調添加特殊標記
            display_name = language
            if language.startswith('hakka_'):
                display_name = f"🗣️  {language}"
            
            print(f"{display_name:<20} {count:<10,} {duration:<12.2f} {percentage:<8.1f}%")
        
        print("-" * 60)
        print()

    def _setup_custom_language_tokens(self):
        """
        真正添加客家話語言 token 到 Whisper tokenizer 的詞彙表
        擴展語言支援而不是繞過驗證
        """
        hakka_languages = {
            'hakka_sixian': '<|hakka_sixian|>',
            'hakka_hailu': '<|hakka_hailu|>',
            'hakka_dapu': '<|hakka_dapu|>',
            'hakka_raoping': '<|hakka_raoping|>',
            'hakka_zhaoan': '<|hakka_zhaoan|>',
            'hakka_nansixian': '<|hakka_nansixian|>'
        }
        
        print(f"🔧 開始擴展 Whisper tokenizer 詞彙表")
        print(f"📋 要添加的客家話腔調：{list(hakka_languages.keys())}")
        
        tokenizer = self.processor.tokenizer
        
        # 1. 添加新的語言 token 到詞彙表
        new_tokens = []
        vocab = tokenizer.get_vocab()
        
        for lang_code, token in hakka_languages.items():
            if token not in vocab:
                new_tokens.append(token)
                print(f"   ➕ 添加語言 token：{token}")
        
        if new_tokens:
            # 添加特殊 token
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            print(f"✅ 成功添加 {len(new_tokens)} 個語言 token")
            
            # 重新獲取更新後的詞彙表
            updated_vocab = tokenizer.get_vocab()
            
            # 2. 修補語言驗證邏輯，讓新語言被接受
            if hasattr(tokenizer, '_get_language_id'):
                original_get_language_id = tokenizer._get_language_id
                
                def patched_get_language_id(language):
                    if language and language.lower() in hakka_languages:
                        # 返回對應的 token ID
                        token = hakka_languages[language.lower()]
                        token_id = updated_vocab.get(token)
                        print(f"🗣️ 客家話腔調 {language} -> token_id: {token_id}")
                        return token_id
                    else:
                        return original_get_language_id(language)
                
                tokenizer._get_language_id = patched_get_language_id
            
            # 3. 修補 prefix_tokens 屬性來處理新語言
            original_prefix_tokens_property = tokenizer.__class__.prefix_tokens
            
            def patched_prefix_tokens(self):
                # 檢查是否有客家話語言設定
                if hasattr(self, 'language') and self.language and self.language.lower() in hakka_languages:
                    lang_code = self.language.lower()
                    token = hakka_languages[lang_code]
                    token_id = updated_vocab.get(token)
                    
                    if token_id is not None:
                        # 只在第一次或每1000次時顯示，避免刷頻
                        if not hasattr(self, '_hakka_token_logged') or not hasattr(self, '_hakka_log_count'):
                            self._hakka_token_logged = set()
                            self._hakka_log_count = 0
                        
                        if token not in self._hakka_token_logged or self._hakka_log_count % 1000 == 0:
                            print(f"🎯 使用客家話 token：{token} (ID: {token_id})")
                            self._hakka_token_logged.add(token)
                        
                        self._hakka_log_count += 1
                        # 構建包含客家話語言 token 的前綴
                        prefix_tokens = [
                            updated_vocab['<|startoftranscript|>'],
                            token_id,  # 客家話語言 token
                            updated_vocab['<|transcribe|>'] if hasattr(self, 'task') and self.task == 'transcribe' else updated_vocab.get('<|translate|>', updated_vocab['<|transcribe|>'])
                        ]
                        return prefix_tokens
                
                # 其他情況使用原始邏輯
                return original_prefix_tokens_property.fget(self)
            
            # 應用修補
            tokenizer.__class__.prefix_tokens = property(patched_prefix_tokens)
            
            print("🎉 客家話語言 token 已成功整合到 Whisper tokenizer")
            print("✨ 現在每個客家話腔調都有獨立的語言識別 token")
            
        else:
            print("ℹ️  所有客家話語言 token 已存在，無需添加")
        
        return list(hakka_languages.keys())

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

    def __getitem__(self, idx):
        try:
            # 从数据列表里面获取音频数据、采样率和文本
            sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            # 可以为单独数据设置语言
            # 映射自定義語言到支援的語言
            mapped_language = self._map_custom_language(language if language is not None else self.language)
            self.processor.tokenizer.set_prefix_tokens(language=mapped_language)
            if len(transcript) > 0:
                # 加载带有时间戳的文本
                if self.timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # 从输入音频数组中计算log-Mel输入特征
                    data["input_features"] = self.processor(audio=sample, sampling_rate=self.sample_rate).input_features
                else:
                    # 获取log-Mel特征和标签ID
                    data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
            else:
                # 如果没有文本，则使用<|nospeech|>标记
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data['labels'] = [self.startoftranscript, self.nospeech, self.endoftext]
            return data
        except Exception as e:
            print(f'读取数据出错，序号：{idx}，错误信息：{e}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

    # 分割读取音频
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # 数据增强
    def augment(self, sample, sample_rate):
        for config in self.augment_configs:
            if config['type'] == 'speed' and random.random() < config['prob']:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = config['params']['min_speed_rate'], \
                        config['params']['max_speed_rate'], config['params']['num_rates']
                    self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                rate = random.choice(self.speed_rates)
                sample = self.change_speed(sample, speed_rate=rate)
            if config['type'] == 'shift' and random.random() < config['prob']:
                min_shift_ms, max_shift_ms = config['params']['min_shift_ms'], config['params']['max_shift_ms']
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config['type'] == 'volume' and random.random() < config['prob']:
                min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config['type'] == 'resample' and random.random() < config['prob']:
                new_sample_rates = config['params']['new_sample_rates']
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(sample, orig_sr=sample_rate, target_sr=new_sample_rate)
                sample_rate = new_sample_rate
            if config['type'] == 'noise' and random.random() < config['prob']:
                min_snr_dB, max_snr_dB = config['params']['min_snr_dB'], config['params']['max_snr_dB']
                if self.noises_path is None:
                    self.noises_path = []
                    noise_dir = config['params']['noise_dir']
                    if os.path.exists(noise_dir):
                        for file in os.listdir(noise_dir):
                            self.noises_path.append(os.path.join(noise_dir, file))
                noise_path = random.choice(self.noises_path)
                snr_dB = random.randint(min_snr_dB, max_snr_dB)
                sample = self.add_noise(sample, sample_rate, noise_path=noise_path, snr_dB=snr_dB)
        return sample, sample_rate

    # 改变语速
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # 音频偏移
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # 改变音量
    @staticmethod
    def volume(sample, gain):
        sample *= 10.**(gain / 20.)
        return sample

    # 声音重采样
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # 添加噪声
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # 标准化音频音量，保证噪声不会太大
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample *= 10. ** (gain / 20.)
        # 指定噪声音量
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample *= 10. ** (noise_gain_db / 20.)
        # 固定噪声长度
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - sample.shape[0])
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
        