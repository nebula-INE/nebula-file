import torch
import torchaudio
import os
import json
import numpy as np
from pathlib import Path
from modules.data.oto_parser import OtoParser
from modules.tools.batch_voice_optimizer import BatchVoiceOptimizer

class VoiceAdaptationEngine:
    def __init__(self, base_model_path: str = "models/bigvgan_base.onnx"):
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.oto_parser = OtoParser()
        self.optimizer = BatchVoiceOptimizer(target_sr=16000)  # 特徴抽出用

    def extract_acoustic_fingerprint(self, voice_dir: str) -> dict:
        """音源フォルダから音響指紋（統計量＋Embeddingベース）を抽出"""
        # 1. oto.ini パース
        self.oto_parser.load_voice_dir(voice_dir)
        aliases = self.oto_parser.all_aliases()
        
        # 2. 全WAVの特徴量を一括抽出（バッチ処理）
        wav_paths = []
        for alias in aliases:
            entry = self.oto_parser.get(alias)
            if entry:
                wav_paths.append(entry.wav_path)
        
        features_list = []
        for wav_path in wav_paths:
            features = self.optimizer._extract_acoustic_features(wav_path)  # 既存の関数を流用
            features_list.append(features)
        
        # 3. 統計量を集計（平均・分散・歪度など）
        fingerprint = {
            "f0_mean": np.mean([f.f0_mean for f in features_list]),
            "f0_std": np.std([f.f0_std for f in features_list]),
            "spectral_centroid_mean": np.mean([f.centroid_mean for f in features_list]),
            "zcr_mean": np.mean([f.zcr_mean for f in features_list]),
            "rms_mean": np.mean([f.rms_mean for f in features_list]),
            "num_samples": len(features_list),
            "oto_params": {
                alias: {
                    "preutterance": entry.preutterance,
                    "overlap": entry.overlap,
                    "fixed_range": entry.fixed_range
                } for alias, entry in self.oto_parser._db.items()
            }
        }
        return fingerprint

    def adapt_bigvgan_with_lora(self, voice_dir: str, fingerprint: dict, output_dir: str = ".vose_adapt"):
        """BigVGANにLoRAアダプターを追加学習させる"""
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import BigVGANModel  # 仮（実際にはBigVGANのPyTorch実装が必要）
        
        # 1. ベースBigVGANモデルをロード（PyTorch版）
        base_model = BigVGANModel.from_pretrained(self.base_model_path)
        base_model.to(self.device)
        
        # 2. LoRA設定（ターゲットは全線形層）
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8,  # ランク（小さいほど軽量）
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(base_model, lora_config)
        model.train()
        
        # 3. 学習データの準備（音源フォルダ内の全WAVからペアを作成）
        #    → WORLD分析（F0 + メルスペクトル）と元波形のペア
        train_loader = self._prepare_dataloader(voice_dir, fingerprint)
        
        # 4. 学習ループ（数エポック、短時間で収束させる）
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.L1Loss()  # 波形のL1 loss
        
        for epoch in range(3):  # 3エポックで十分
            for batch in train_loader:
                mel_spec, target_wav = batch
                mel_spec = mel_spec.to(self.device)
                target_wav = target_wav.to(self.device)
                
                # 推論
                pred_wav = model(mel_spec)
                
                # 損失計算
                loss = loss_fn(pred_wav, target_wav)
                
                # バックプロパゲーション
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 5. 学習済みアダプターを保存（LoRA重みのみ）
        os.makedirs(output_dir, exist_ok=True)
        voice_name = Path(voice_dir).name
        save_path = os.path.join(output_dir, f"{voice_name}_lora.pt")
        torch.save(model.state_dict(), save_path)
        
        # 6. ONNX変換（推論時にC++で読み込めるように）
        self._convert_lora_to_onnx(model, save_path.replace(".pt", ".onnx"))
        
        return save_path

    def _convert_lora_to_onnx(self, model, onnx_path):
        """LoRA適用済みモデルをONNXにエクスポート（C++エンジンで使うため）"""
        model.eval()
        dummy_input = torch.randn(1, 80, 256).to(self.device)  # メルスペクトル形状
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["mel_input"],
            output_names=["audio_output"],
            dynamic_axes={"mel_input": {2: "frames"}, "audio_output": {1: "samples"}},
            opset_version=14,
        )
