# VO-SE_vocal AI化 開発ロードマップ

> Project: VO-SE_vocal
> AI Architecture: Universal Singing Model + Voicebank Conditioning
> Primary Constraint: **任意のUTAU音源を1つのAIモデルで扱えること**

---

# 0. Vision

VO-SE_vocalを、単なるUTAU音源エディタから、

> **「任意のUTAU音源を、AIによって自然で表現力のある歌唱へ変換できる歌声制作環境」**

へ進化させる。

AIによって音源そのものを置き換えるのではなく、

```text
MIDI
Lyrics
Phonemes
Expression
       │
       ▼
┌────────────────────────┐
│   Universal Singing AI │
│                        │
│ Pitch                  │
│ Timing                 │
│ Dynamics               │
│ Vibrato                │
│ Expression             │
└───────────┬────────────┘
            │
            ▼
   Singing Control Data
            │
            ▼
┌────────────────────────┐
│     UTAU Voicebank     │
│                        │
│ oto.ini                │
│ Samples                │
│ PrefixMap              │
│ CV / VCV / CVVC        │
└───────────┬────────────┘
            │
            ▼
       VO-SE Engine
            │
            ▼
         Vocal
```

という構造を基本とする。

---

# 1. 最重要要件

## 1.1 Universal Model

AI化における最重要要件は、

> **音源ごとにAIモデルを用意しない。**

ことである。

目標構成：

```text
models/
└── vose_universal_singing.onnx
```

ユーザーが追加するUTAU音源：

```text
voicebanks/
├── Voice_A/
├── Voice_B/
├── Voice_C/
├── Voice_D/
└── ...
```

AIモデルは共通。

```text
                 ┌──────────────────┐
Voicebank ──────►│ Voicebank Adapter │
                 └────────┬─────────┘
                          │
                          ▼
                   Voice Embedding
                          │
                          ▼
MIDI ───────────────┐
Lyrics ─────────────┤
Phonemes ───────────┤
Expression ─────────┤
                    ▼
          Universal Singing Model
                    │
                    ▼
        Singing Control Parameters
                    │
                    ▼
             UTAU Renderer
```

---

# 2. AIの役割

AIは音源の「声そのもの」を生成するのではなく、まずは

**「どう歌わせるか」**

を生成する。

## AIが担当するもの

* Pitch Dynamics
* Pitch Correction
* Timing
* Note Transition
* Vibrato
* Vibrato Timing
* Vibrato Depth
* Dynamics
* Energy
* Attack
* Release
* Expression
* Breath Timing
* Phrase-level Dynamics
* Emotion-related controls

## UTAU / VO-SE Engineが担当するもの

* 音源サンプル
* 音素レンダリング
* oto.ini
* クロスフェード
* 原音パラメータ
* 音源固有の音色
* 実際の音声波形生成

つまり、

> **AI = 歌唱表現の頭脳**
> **UTAU = 声の素材**
> **VO-SE Engine = 音声生成**

という責務分離を行う。

---

# 3. Voicebank Universalization

## 3.1 Voicebank Adapter

UTAU音源ごとの違いをAIモデルへ直接ハードコードしない。

Voicebank Adapterを用意する。

```text
modules/
└── ai/
    ├── universal_model.py
    ├── voicebank_adapter.py
    ├── voicebank_analyzer.py
    ├── voice_embedding.py
    └── inference.py
```

---

# 4. Voicebank Analyzer

UTAU音源を読み込んだ際に解析する。

## 取得情報

### 基本情報

* Voicebank name
* Author
* Language
* Version
* Character name

### oto.ini

* alias
* WAV filename
* offset
* consonant
* cutoff
* preutterance
* overlap

### 音素情報

* CV
* VCV
* CVVC
* VC
* V
* C
* その他alias形式

### PrefixMap

* Prefix
* Suffix
* Note range
* Alias mapping

### 音域

* Minimum pitch
* Maximum pitch
* Recorded pitch
* Supported pitch regions

### 音源品質

可能な範囲で、

* sample length
* sample count
* phoneme coverage
* recording density
* phoneme balance

などを取得する。

---

# 5. Voicebank Embedding

Voicebank Analyzerの結果をAIへ渡せる固定長ベクトルに変換する。

```text
UTAU Voicebank
      │
      ▼
Voicebank Analyzer
      │
      ├── oto.ini
      ├── PrefixMap
      ├── phonemes
      ├── pitch range
      ├── sample metadata
      └── recording information
      │
      ▼
Voicebank Encoder
      │
      ▼
Voice Embedding
```

例：

```text
voice_embedding = [
    ...
]
```

Universal Modelはこのembeddingを条件として使用する。

---

# 6. Audio-based Voice Embedding

将来的には、metadataだけではなく実際の音源サンプルからVoice Embeddingを抽出できる構造にする。

理由：

`oto.ini`だけでは、

* 声質
* 明るさ
* 息成分
* 声の硬さ
* 声の太さ
* 子音の強さ

などを完全には表現できないため。

ただし、初期バージョンでは必須にしない。

## Phase 1

```text
oto.ini
PrefixMap
phoneme metadata
pitch range
```

## Phase 2

```text
sample audio
   ↓
Audio Encoder
   ↓
Voice Embedding
```

## Phase 3

```text
Metadata Embedding
        +
Audio Embedding
        ↓
Unified Voice Embedding
```

---

# 7. Universal Singing Model

## 入力

Universal Modelへの入力は以下を基本とする。

```text
note_pitch
note_duration
phoneme
phoneme_duration
previous_note
next_note
phrase_position
word_position
voice_embedding
expression_parameters
```

必要に応じて、

```text
tempo
accent
breath
language
```

などを追加する。

---

# 8. AI Output

AIは直接音声を出力するのではなく、まず歌唱制御データを出力する。

```text
Universal Singing Model
        │
        ├── F0 Curve
        ├── Timing
        ├── Dynamics
        ├── Vibrato
        ├── Expression
        └── Transition
```

例：

```text
{
    "pitch_curve": [...],
    "timing": [...],
    "energy": [...],
    "vibrato": {
        "rate": ...,
        "depth": ...,
        "onset": ...
    }
}
```

---

# 9. Human-in-the-loop

AIがすべてを決定する設計にはしない。

ユーザーがAI出力を編集できることを必須とする。

```text
AI Generated
     │
     ▼
Editable Control Data
     │
     ├── Pitch
     ├── Timing
     ├── Vibrato
     ├── Dynamics
     └── Expression
```

UI上では、

* AI生成
* AI強度
* AI再生成
* AI部分適用
* 手動修正
* Undo
* AI前後比較

を可能にする。

---

# 10. AI Strength

AIの出力を完全に適用するのではなく、強度を調整可能にする。

```text
AI Strength = 0.0
```

↓

完全に手動。

```text
AI Strength = 0.5
```

↓

AIを50%適用。

```text
AI Strength = 1.0
```

↓

AIを最大適用。

さらに将来的には、

```text
Pitch AI
Timing AI
Dynamics AI
Vibrato AI
Expression AI
```

それぞれ独立して強度を設定できるようにする。

---

# 11. Existing AuralAIEngine

現在存在する、

```text
modules/gui/aural_engine.py
```

の `AuralAIEngine` は、AI化の土台として再利用する。

現在の設計では、

```text
base_f0
   ↓
ONNX Model
   ↓
pitch delta
   ↓
final F0
```

という構造になっている。

これはUniversal Singing AIの

**Pitch Dynamics Module**

として発展させる。

---

# 12. AuralAIEngineの再設計

現在：

```python
get_baked_pitch(
    note_id,
    base_f0_array,
    strength=0.8
)
```

将来：

```python
generate_performance(
    notes,
    phonemes,
    voice_embedding,
    expression,
    strength
)
```

のような高レベルAPIへ移行する。

内部では、

```text
Performance Input
       │
       ▼
Feature Encoder
       │
       ▼
Universal Singing Model
       │
       ├── Pitch
       ├── Timing
       ├── Dynamics
       ├── Vibrato
       └── Expression
       │
       ▼
Performance Result
```

とする。

---

# 13. AI Model Architecture

初期段階では巨大な音声生成モデルを作らない。

優先するのは、

```text
Universal Performance Model
```

である。

候補構造：

```text
Input Encoder
     │
     ▼
Context Encoder
     │
     ▼
Sequence Model
     │
     ├── Pitch Head
     ├── Timing Head
     ├── Dynamics Head
     ├── Vibrato Head
     └── Expression Head
```

複数の出力Headを持つMulti-task Modelを基本候補とする。

---

# 14. なぜ一つのモデルにするのか

VO-SE_vocalは任意のUTAU音源を読み込むことを前提とする。

したがって、

```text
Voice A → model_A
Voice B → model_B
Voice C → model_C
```

という設計は採用しない。

代わりに、

```text
Voice A ─┐
Voice B ─┤
Voice C ─┼─► Universal Model
Voice D ─┤
Voice E ─┘
```

とする。

---

# 15. Training Dataset

Universal Modelを成立させるため、単一音源だけで学習しない。

複数の、

* 声質
* 性別
* 音域
* 歌唱スタイル
* 録音品質
* UTAU形式

を含むデータセットを構築する。

---

# 16. Training Data

基本的な学習データ：

```text
MIDI
Lyrics
Phonemes
F0
Energy
Timing
Vibrato
Expression
Voice Identity
```

可能なら、

```text
note onset
note duration
phoneme onset
phoneme duration
phrase boundary
breath
accent
```

も保存する。

---

# 17. Dataset Format

内部的な学習データフォーマットを統一する。

例：

```json
{
    "voice_id": "voice_001",
    "notes": [],
    "phonemes": [],
    "f0": [],
    "energy": [],
    "timing": [],
    "vibrato": [],
    "expression": []
}
```

重要なのは、

> UTAUのファイル形式そのものをAIの学習フォーマットにしない

ことである。

UTAU → Dataset Format

という変換層を作る。

---

# 18. Voicebank Type Compatibility

最低限、以下を設計対象とする。

### CV

```text
ka
ki
ku
ke
ko
```

### VCV

```text
a ka
a ki
a ku
```

### CVVC

```text
ka
a k
```

### Prefix/Suffix

```text
C4ka
C5ka
ka_R
```

Voicebank Adapterが内部表現へ正規化する。

---

# 19. Internal Phoneme Representation

AI内部ではUTAU aliasをそのまま使用しない。

```text
UTAU Alias
    ↓
Alias Resolver
    ↓
Canonical Phoneme
```

例：

```text
"あ"
" a"
"a"
"あー"
```

などを必要に応じて統一表現へ変換する。

---

# 20. Unknown Voicebank

未知のUTAU音源を完全に拒否しない。

```text
Known Voicebank
      ↓
Full Conditioning
```

未知音源：

```text
Unknown Voicebank
      ↓
Generic Voice Embedding
      ↓
Universal Model
```

というFallbackを用意する。

---

# 21. Unknown Voicebank Compatibility

新しい音源を読み込んだ際、

```text
Voicebank Analyzer
       │
       ├── oto.ini
       ├── PrefixMap
       ├── aliases
       └── sample metadata
       │
       ▼
Voice Embedding
       │
       ▼
Universal Model
```

とする。

モデル再学習なしで利用可能であることを目標とする。

---

# 22. AI Rendering Pipeline

最終的な処理系：

```text
Project
   │
   ├── MIDI
   ├── Lyrics
   ├── Notes
   └── Expression
   │
   ▼
Phoneme Generation
   │
   ▼
Voicebank Adapter
   │
   ▼
Voice Embedding
   │
   ▼
Universal Singing AI
   │
   ├── Pitch
   ├── Timing
   ├── Dynamics
   ├── Vibrato
   └── Expression
   │
   ▼
UTAU Rendering
   │
   ▼
VO-SE Audio Engine
   │
   ▼
Audio
```

---

# 23. AI Rendering Modes

## Manual

```text
MIDI
 ↓
UTAU
```

AIなし。

## Assisted

```text
MIDI
 ↓
AI
 ↓
Editable Parameters
 ↓
UTAU
```

## Full AI

```text
MIDI
 ↓
AI Performance
 ↓
UTAU
```

3モードを用意する。

---

# 24. Real-time Preview

AI推論は可能な限りリアルタイム編集を阻害しない。

基本方針：

```text
Editor
   │
   ├── Fast Preview
   │
   └── Full Render
```

## Fast Preview

* 軽量モデル
* CPU inference
* 部分推論
* キャッシュ利用

## Full Render

* 全フレーズ推論
* 高品質設定
* オフラインレンダリング

---

# 25. ONNX Deployment

現在のONNX Runtime基盤を継続利用する。

```text
Training
   ↓
PyTorch
   ↓
ONNX Export
   ↓
VO-SE_vocal
   ↓
ONNX Runtime
```

配布時には、

```text
models/
└── vose_universal_singing.onnx
```

を基本とする。

---

# 26. Model Versioning

AIモデルとアプリ本体を独立してバージョン管理する。

例：

```text
VO-SE_vocal
  1.2.0

Universal Singing Model
  0.1.0
```

モデルmetadata：

```json
{
    "model_name": "vose_universal_singing",
    "version": "0.1.0",
    "input_schema": "1",
    "voice_embedding_version": "1",
    "phoneme_schema": "1"
}
```

---

# 27. Model Compatibility

モデル更新によって旧プロジェクトが壊れないようにする。

```text
Project
   │
   ├── model_version
   ├── voicebank_id
   └── AI parameters
```

を保存する。

---

# 28. Cache

現在の `AuralAIEngine` にあるcache機構を発展させる。

キャッシュキー：

```text
project_id
+
note_id
+
voicebank_id
+
model_version
+
AI parameters
```

同じ条件なら再推論しない。

---

# 29. AI UI

AIを別アプリのように扱わない。

既存の歌唱編集画面へ統合する。

基本UI：

```text
┌─────────────────────────────┐
│ AI Performance              │
├─────────────────────────────┤
│ Pitch        [────●────]    │
│ Timing       [────●────]    │
│ Dynamics     [────●────]    │
│ Vibrato      [────●────]    │
│ Expression   [────●────]    │
│                             │
│ [ Generate AI ]             │
│ [ Apply ] [ Reset ]         │
└─────────────────────────────┘
```

---

# 30. AI Explainability

AIが変更した箇所を確認できるようにする。

例えば、

```text
AI Modified:
Pitch      +12 cents
Timing     -8 ms
Dynamics   +6%
Vibrato    Added
```

などを表示可能にする。

---

# 31. AI安全性

AI生成結果が極端にならないよう制限する。

例：

```text
max pitch deviation
max timing deviation
max dynamics deviation
max vibrato depth
```

を設定する。

AIが暴走した場合でも、

```text
AI Output
   ↓
Constraint
   ↓
Safe Performance
```

となるようにする。

---

# 32. Training / Inference Separation

アプリ本体に学習処理を入れない。

```text
Training Environment
        │
        ▼
Model
        │
        ▼
ONNX
        │
        ▼
VO-SE_vocal
```

VO-SE_vocalは原則としてInference専用とする。

---

# 33. Repository Structure

将来的に以下へ整理する。

```text
modules/
├── ai/
│   ├── universal_model.py
│   ├── inference.py
│   ├── voicebank_adapter.py
│   ├── voicebank_analyzer.py
│   ├── voice_embedding.py
│   ├── phoneme_normalizer.py
│   └── model_manager.py
│
├── audio/
├── backend/
├── bridge/
├── data/
├── ffi/
├── gui/
├── talk/
└── ...
```

学習側はアプリ本体から分離する。

```text
ai_training/
├── dataset/
├── preprocessing/
├── models/
├── training/
├── evaluation/
└── export/
```

---

# 34. Phase 0 — Architecture Freeze

最初にAI実装を始める前に仕様を固定する。

### Tasks

* [ ] Universal Model方式を確定
* [ ] Voicebank Adapter仕様決定
* [ ] Voice Embedding仕様決定
* [ ] Canonical Phoneme仕様決定
* [ ] AI Output Schema決定
* [ ] Model metadata仕様決定
* [ ] Project保存仕様決定

### 完了条件

UTAU音源をAIへ渡すまでのデータフローが確定している。

---

# 35. Phase 1 — Voicebank Analyzer

### Tasks

* [ ] oto.ini parser
* [ ] PrefixMap parser
* [ ] alias parser
* [ ] CV detector
* [ ] VCV detector
* [ ] CVVC detector
* [ ] pitch range analyzer
* [ ] sample metadata analyzer
* [ ] Voicebank profile生成

### Output

```json
{
    "voicebank": "...",
    "phoneme_type": "VCV",
    "pitch_range": [],
    "aliases": [],
    "sample_count": 0
}
```

---

# 36. Phase 2 — Canonical Phoneme System

### Tasks

* [ ] UTAU alias normalization
* [ ] phoneme tokenizer
* [ ] CV normalization
* [ ] VCV normalization
* [ ] CVVC normalization
* [ ] Prefix/Suffix resolution
* [ ] unknown alias fallback

### 完了条件

異なるUTAU音源でもAI内部では共通のphoneme representationを使用できる。

---

# 37. Phase 3 — Voice Embedding

### Tasks

* [ ] metadata encoder
* [ ] Voicebank profile encoder
* [ ] embedding format
* [ ] embedding cache
* [ ] unknown voicebank fallback

初期段階ではaudio encoderを必須としない。

---

# 38. Phase 4 — Universal Pitch AI

最初のAIモデルはPitchから開始する。

```text
MIDI
 ↓
Base F0
 ↓
Voice Embedding
 ↓
Universal Pitch Model
 ↓
Pitch Delta
 ↓
Final F0
```

現在の `AuralAIEngine` をこのPhaseへ移行する。

### Tasks

* [ ] training dataset preparation
* [ ] multi-voice training
* [ ] voice conditioning
* [ ] F0 model
* [ ] ONNX export
* [ ] inference integration
* [ ] pitch cache

---

# 39. Phase 5 — Dynamics AI

Pitch AIの次にDynamicsを実装する。

```text
Input
 ↓
Universal Model
 ↓
Energy / Dynamics
```

対象：

* note attack
* note release
* phrase dynamics
* intensity
* accent

---

# 40. Phase 6 — Timing AI

歌唱タイミングをモデル化する。

```text
MIDI Timing
     ↓
AI Timing Offset
     ↓
Actual Timing
```

対象：

* phoneme timing
* consonant timing
* vowel onset
* note transition
* phrase timing

---

# 41. Phase 7 — Vibrato AI

AIが、

* vibrato onset
* vibrato rate
* vibrato depth
* vibrato duration

を予測する。

```text
Note
 ↓
Vibrato Prediction
 ↓
Editable Vibrato Curve
```

完全自動だけではなく手動編集可能にする。

---

# 42. Phase 8 — Expression AI

最終的に、

* emotion
* phrase expression
* dynamics
* attack
* release
* breath
* emphasis

などを扱う。

ただし「Emotion」を直接音声に変換するブラックボックス化は避け、

```text
Emotion
 ↓
Performance Parameters
 ↓
UTAU
```

とする。

---

# 43. Phase 9 — Universal Model Integration

各AIを統合する。

```text
                ┌── Pitch
                ├── Timing
Input ──────────┼── Dynamics
                ├── Vibrato
                └── Expression
                         │
                         ▼
                 Performance Data
```

最終的には単一Universal Model、または共有Encoder＋複数Head構造を採用する。

---

# 44. Phase 10 — Web / Desktop Integration

Desktop版とWeb版でAI仕様を分裂させない。

共通：

```text
Project Schema
AI Parameter Schema
Voicebank Profile
AI Output Schema
```

を使用する。

Desktop：

```text
ONNX Runtime
```

Web：

```text
WebAssembly / WebGPU / Server Inference
```

など、実行環境だけを差し替える。

---

# 45. Phase 11 — Mobile

iPad / iPhoneでは、重いAI推論を必ずしも端末上で行わない。

候補：

```text
Mobile UI
   │
   ├── Local lightweight inference
   │
   └── Remote inference
```

ただしプロジェクト編集自体はオフラインでも成立する設計を維持する。

---

# 46. Evaluation

Universal Modelでは、

> 「一つの音源だけで綺麗に動く」

ことを成功条件にしない。

複数Voicebankで評価する。

## 評価軸

### Voicebank diversity

* CV
* VCV
* CVVC
* male
* female
* low voice
* high voice
* soft voice
* powerful voice
* multilingual

### Singing quality

* pitch accuracy
* timing accuracy
* vibrato naturalness
* dynamics
* phrase expression
* consonant timing

### Generalization

```text
Training Voicebank
        vs
Unseen Voicebank
```

未知音源でも破綻しないことを重要視する。

---

# 47. Evaluation Dataset

最低でも、

```text
Training Voices
Validation Voices
Unseen Voices
```

に分離する。

特に、

> **学習に一度も登場していないUTAU音源**

をテストする。

これによってUniversal Modelとしての汎化性能を確認する。

---

# 48. Failure Handling

AIが利用できない場合でもVO-SE_vocal自体は動作する。

```text
AI Available
    ↓
AI Rendering

AI Unavailable
    ↓
Normal UTAU Rendering
```

ONNX Runtimeがない場合：

```text
Manual / Classic Mode
```

へFallbackする。

モデルが壊れている場合も同様。

---

# 49. Performance Requirements

目標：

### CPU

一般的なデスクトップCPUで実用的な速度。

### Memory

モデルサイズを可能な限り抑える。

### GPU

存在する場合は利用可能。

### Mobile

将来的に軽量モデルを用意。

---

# 50. Model Compression

Universal Modelは巨大化しすぎないようにする。

候補：

* FP32
* FP16
* INT8
* Quantization
* Distillation

ただし、品質低下とのトレードオフを評価して決定する。

---

# 51. Model Distribution

基本モデル：

```text
vose_universal_singing.onnx
```

必要に応じて：

```text
vose_universal_singing_int8.onnx
vose_universal_singing_fp16.onnx
```

などを用意する。

---

# 52. AI Presets

ユーザーがAIを細かく設定しなくても使えるPresetを提供する。

例：

```text
Natural
Soft
Power
Ballad
Pop
Rock
Smooth
Manual Assist
```

ただしPresetはAIモデルそのものではなく、

```text
AI Parameters
```

の組み合わせとして扱う。

---

# 53. Non-destructive AI

AI処理は原則として非破壊編集。

```text
Original Performance
        +
AI Layer
        +
Manual Layer
```

として保存する。

例：

```text
Pitch:
    base
    ai_delta
    user_delta
```

最終値：

```text
final = base + ai_delta + user_delta
```

---

# 54. AI Regeneration

ユーザーが一部だけAIを再生成できるようにする。

例：

```text
Measure 1-4
   AI Generate

Measure 5-8
   Manual

Measure 9-12
   AI Generate
```

これによりAIを完全自動作曲機能ではなく、

**歌唱編集アシスタント**

として利用できる。

---

# 55. Important Design Rule

AIがUTAU音源そのものを置き換えない。

避ける：

```text
MIDI
 ↓
AI
 ↓
完全生成音声
```

初期VO-SEでは採用しない。

推奨：

```text
MIDI
 ↓
AI Performance
 ↓
UTAU Voicebank
 ↓
VO-SE Engine
 ↓
Audio
```

これにより既存のUTAU資産をそのまま活用できる。

---

# 56. Future Generative Singing Engine

将来的に必要になった場合のみ、

```text
Universal Singing AI
        ↓
Acoustic / Neural Vocoder
        ↓
Generated Voice
```

を研究対象にする。

ただしこれは現在のAI化とは別Phaseとする。

---

# 57. Security / Licensing

UTAU音源にはそれぞれ異なる利用規約が存在する。

AI学習に使用する音源については、

* 再配布条件
* 学習利用条件
* 商用利用条件
* 二次利用条件
* 作者の許可

を確認する。

特に、

> **ユーザーが所有しているUTAU音源を、そのまま学習データとしてクラウドへ送信する**

ような仕様は初期段階では採用しない。

---

# 58. Development Priority

優先順位：

```text
1. Voicebank Adapter
2. Canonical Phoneme
3. Voicebank Analyzer
4. Voice Embedding
5. Universal Pitch AI
6. Dynamics AI
7. Timing AI
8. Vibrato AI
9. Expression AI
10. Unified Universal Model
11. Web Integration
12. Mobile Integration
```

---

# 59. MVP

最初のAI化完成ラインは以下。

```text
UTAU Voicebank
       ↓
Voicebank Analyzer
       ↓
Voice Embedding
       ↓
Universal Pitch Model
       ↓
AI Pitch Curve
       ↓
UTAU Rendering
```

条件：

* [ ] 複数UTAU音源で動作
* [ ] 学習済みモデルは1つ
* [ ] 音源ごとのモデル不要
* [ ] CV/VCVの基本対応
* [ ] AI強度調整
* [ ] AI結果を手動編集可能
* [ ] ONNX Runtimeで推論可能
* [ ] AIなしでも従来レンダリング可能

---

# 60. Final Architecture

最終目標：

```text
                         VO-SE_vocal
                              │
             ┌────────────────┴────────────────┐
             │                                 │
        Project Data                       Voicebank
             │                                 │
             │                         Voicebank Analyzer
             │                                 │
             │                         Voicebank Adapter
             │                                 │
             │                         Voice Embedding
             │                                 │
             └──────────────┬──────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ Universal Singing AI │
                │                      │
                │ Pitch                │
                │ Timing               │
                │ Dynamics             │
                │ Vibrato              │
                │ Expression           │
                └──────────┬───────────┘
                           │
                           ▼
                  Editable Performance
                           │
                           ▼
                    UTAU Voicebank
                           │
                           ▼
                     VO-SE Engine
                           │
                           ▼
                         Audio
```

---

# 61. Definition of Done

VO-SE_vocalのAI化を完了と判断する基準：

* [ ] 1つのUniversal Modelで複数UTAU音源を処理できる
* [ ] 新しいUTAU音源を追加するだけでAIを利用できる
* [ ] 音源ごとのAIモデルを必要としない
* [ ] oto.iniを自動解析できる
* [ ] PrefixMapを処理できる
* [ ] CV / VCV / CVVCを正規化できる
* [ ] Voicebank Embeddingを生成できる
* [ ] Pitch AIが動作する
* [ ] Timing AIが動作する
* [ ] Dynamics AIが動作する
* [ ] Vibrato AIが動作する
* [ ] Expression AIが動作する
* [ ] AI出力をユーザーが編集できる
* [ ] AI処理を非破壊で保存できる
* [ ] ONNXで推論できる
* [ ] AIなしでも従来のUTAUレンダリングが動作する
* [ ] 未学習Voicebankでの汎化テストを実施する
* [ ] Desktop / Webで共通のAIデータ仕様を利用する
* [ ] モデルのバージョン管理ができる

---

# 62. Core Principle

VO-SE_vocal AI化の中心思想：

> **「AIが声を作る」のではなく、「AIが音源の歌わせ方を理解する」。**

そして、

> **「音源ごとにAIを作る」のではなく、「一つのUniversal Modelが音源を理解する」。**

これをVO-SE_vocal AI Architectureの基本原則とする。
