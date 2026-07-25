# modules/ai/adaptation_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from modules.ai.adaptation_dataset import UTAUAdaptationDataset
import os
import json
from tqdm import tqdm
import numpy as np

class AdaptationTrainer:
    def __init__(
        self,
        base_model_path: str,
        output_dir: str = ".vose_adapt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lora_r: int = 8,
        lr: float = 1e-4,
        epochs: int = 5,
    ):
        self.device = device
        self.output_dir = output_dir
        self.epochs = epochs
        self.lr = lr
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. ベースBigVGANのロード（HuggingFaceの実装を想定）
        #    例: from bigvgan import BigVGANModel
        #    ここでは仮のクラス名を使用
        try:
            from bigvgan import BigVGANModel
            self.base_model = BigVGANModel.from_pretrained(base_model_path)
        except ImportError:
            # フォールバック: ダミーモデル（実際は実装が必要）
            print("⚠️ BigVGAN PyTorch実装が見つかりません。ダミーモデルで動作確認します。")
            self.base_model = self._create_dummy_model()
        
        self.base_model.to(self.device)
        
        # 2. LoRA設定
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # BigVGANの線形層
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()  # 学習可能パラメータの数表示
    
    def _create_dummy_model(self):
        """BigVGANのモック（動作確認用）"""
        import torch.nn as nn
        class DummyBigVGAN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(80, 80)
                self.deconv = nn.ConvTranspose1d(80, 1, kernel_size=256, stride=256)
            def forward(self, mel):
                # mel: [B, T, 80] → [B, 80, T]
                x = mel.permute(0, 2, 1)
                x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self.deconv(x)  # [B, 1, T * 256]
                return x.squeeze(1)  # [B, samples]
        return DummyBigVGAN()
    
    def _mel_spectrogram_loss(self, pred_wav, target_wav, sample_rate=44100):
        """知覚損失（メルスペクトログラム領域でのL1損失）"""
        # 波形 → メルスペクトログラム（同じパラメータで変換）
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            power=1.0,
            norm="slaney",
        ).to(self.device)
        
        # 入力波形をメルに変換（チャンネル次元を追加）
        pred_mel = mel_transform(pred_wav.unsqueeze(1))  # [B, 80, T]
        target_mel = mel_transform(target_wav.unsqueeze(1))
        
        # log圧縮
        pred_mel = torch.log(torch.clamp(pred_mel, min=1e-5))
        target_mel = torch.log(torch.clamp(target_mel, min=1e-5))
        
        return nn.L1Loss()(pred_mel, target_mel)
    
    def train(self, voice_dir: str):
        """メイン学習ループ"""
        # 1. データセット＆ローダー
        dataset = UTAUAdaptationDataset(voice_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=2,  # 音源のサイズに応じて調整（GPUメモリに合わせる）
            shuffle=True,
            collate_fn=UTAUAdaptationDataset.collate_fn,
            num_workers=0,  # Windowsでのマルチプロセス問題回避
        )
        
        if len(dataset) == 0:
            raise ValueError(f"音源フォルダに有効なWAVがありません: {voice_dir}")
        
        print(f"📊 学習データ: {len(dataset)} ファイル")
        
        # 2. オプティマイザ＆スケジューラ
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # 3. 損失関数（波形L1 + メル損失のハイブリッド）
        l1_loss = nn.L1Loss()
        
        # 4. 学習ループ
        self.model.train()
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in pbar:
                mel = batch["mel"].to(self.device)           # [B, T_mel, 80]
                target = batch["target"].to(self.device)     # [B, T_wav]
                mel_lengths = batch["mel_lengths"]
                target_lengths = batch["target_lengths"]
                
                # ★★★★★ 順伝播 ★★★★★
                # BigVGANに入力（メルスペクトログラム）
                # 注: BigVGANによっては [B, 80, T] 形状が必要な場合があるため、必要に応じて転置
                pred = self.model(mel)  # [B, T_wav]
                
                # 有効長でマスク（パディング部分は損失に含めない）
                loss = 0.0
                for i in range(pred.shape[0]):
                    valid_len = target_lengths[i].item()
                    # 波形L1損失
                    loss += l1_loss(pred[i, :valid_len], target[i, :valid_len])
                    # メルスペクトログラム損失（知覚的品質向上）
                    loss += 0.5 * self._mel_spectrogram_loss(
                        pred[i, :valid_len].unsqueeze(0),
                        target[i, :valid_len].unsqueeze(0),
                    )
                loss = loss / pred.shape[0]
                
                # ★★★★★ 逆伝播 ★★★★★
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 勾配クリッピング
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            scheduler.step()
            avg_loss = epoch_loss / len(dataloader)
            print(f"✅ Epoch {epoch+1} 完了: 平均損失 = {avg_loss:.4f}")
            
            # ベストモデルを保存
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_adapter(epoch, is_best=True)
        
        # 最終モデル保存
        self._save_adapter(self.epochs, is_best=False)
        print(f"🎉 学習完了！ベスト損失: {best_loss:.4f}")
        
        return self.output_dir
    
    def _save_adapter(self, epoch, is_best=False):
        """LoRAアダプター重みを保存（.pt）"""
        save_name = "best_lora.pt" if is_best else f"lora_epoch_{epoch}.pt"
        save_path = os.path.join(self.output_dir, save_name)
        
        # アダプター重みのみ抽出
        adapter_weights = {
            k: v for k, v in self.model.state_dict().items()
            if "lora" in k
        }
        torch.save(adapter_weights, save_path)
        
        # メタデータも保存
        meta = {
            "epoch": epoch,
            "base_model": "bigvgan_base",
            "lora_r": 8,
            "voice_dir": self.voice_dir if hasattr(self, "voice_dir") else "unknown",
        }
        with open(os.path.join(self.output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        # ONNXエクスポート（ベストモデルのみ）
        if is_best:
            self._export_to_onnx(save_path)
    
    def _export_to_onnx(self, adapter_path: str):
        """マージ済みモデルをONNXにエクスポート（C++エンジン用）"""
        # 1. LoRA重みをベースモデルにマージ
        self.model.eval()
        merged_model = self.model.merge_and_unload()  # peftのメソッド
        merged_model.to("cpu")
        
        # 2. ダミー入力でONNXエクスポート
        dummy_mel = torch.randn(1, 256, 80)  # [B, T, n_mels]
        
        torch.onnx.export(
            merged_model,
            dummy_mel,
            os.path.join(self.output_dir, "adapted_bigvgan.onnx"),
            input_names=["mel_input"],
            output_names=["audio_output"],
            dynamic_axes={
                "mel_input": {1: "frames"},   # 時間方向可変
                "audio_output": {1: "samples"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"✅ ONNXエクスポート完了: {self.output_dir}/adapted_bigvgan.onnx")
