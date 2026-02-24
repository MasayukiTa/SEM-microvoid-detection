# SEM Microvoid / Striation-Void Detection

SEM画像に対して、以下を1本のスクリプトで実行するためのリポジトリです。

- 物体検出（void / crack）
- レビューGUIでのラベル修正（void / crack / other）
- annotations.json からのレビュー再開
- レビュー結果を使ったファインチューニング

対象スクリプト: `SEM_cudaopenvinofinetuning.py`

## 特徴

- CUDA + OpenVINO の併用による推論ワークフロー
- レビューGUIのキーバインド対応（V/C/A, Ctrl+A など）
- ファインチューニング中に、設定を別GUIから随時変更可能
- 中断時の保存・再開フローを用意

## 動作環境

- OS: Windows 10/11 (推奨)
- Python: 3.10 以上
- 任意: NVIDIA GPU + CUDA 対応 PyTorch（高速化用）

## インストール

### 1) 仮想環境（推奨）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 必要パッケージ

```powershell
pip install -r requirements.txt
```

`requirements.txt` には以下を含めています。

- `numpy`
- `Pillow`
- `openvino`
- `torch`
- `torchvision`

### 3) CUDA版PyTorchを使う場合（任意）

GPU利用時は、環境に合わせてPyTorchを上書きインストールしてください（例: CUDA 12.1）。

```powershell
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 起動方法

同梱のランチャーを使う方法（推奨）:

```powershell
.\run_sem.ps1
```

または:

```bat
run_sem.bat
```

直接実行:

```powershell
python .\SEM_cudaopenvinofinetuning.py
```

## モード概要

起動後、GUIで以下3モードを選択します。

1. detect + review
2. review resume
3. finetune

### 1) detect + review

- `.pth` モデルと画像フォルダを選択
- 検出結果を `_outputs_finetune` に保存
- レビューGUIでラベル調整

主な出力:

- `_outputs_finetune/output_*.jpg`（可視化画像）
- `_outputs_finetune/summary.csv`
- `_outputs_finetune/annotations.json`

### 2) review resume

- 保存済み `annotations.json` を読み込み
- レビュー作業を再開

### 3) finetune

- ベースモデル（`.pth`）と `annotations.json` を選択
- 学習中に設定コントロールGUIで各種パラメータを更新可能
- 学習完了後、`*_finetuned.pth` を出力

## ファインチューニング中のGUI変更

本スクリプトは「学習中でも設定を変更できる」運用を重視しています。

変更可能項目と反映タイミング:

- `input_size`: 次batch
- `freeze_mode`: 次batch（optimizer再構築は次epoch）
- `batch_size`: 次epoch
- `num_workers`: 次epoch
- `lr`: 次epoch
- `epochs`: 次epoch

## レビューGUIの主なショートカット

- `V`: 選択BBoxを `void`
- `C`: 選択BBoxを `crack`
- `A`: 選択BBoxを `other`
- `Ctrl+A`: 現在画像の全BBoxを `other`
- `B`: BBox作成モード ON/OFF
- `Ctrl+Z / Ctrl+Y`: Undo / Redo
- `Space` or `Enter`: 確定して次画像
- `H`: ヘルプ表示

## Git運用メモ

`.gitignore` で以下のモデル成果物は除外設定済みです。

- `*.pth`, `*.pt`, `*.ckpt`
- `*.onnx`
- OpenVINO IR: `*.xml`, `*.bin`

## トラブルシュート

- PowerShell実行ポリシーで `run_sem.ps1` がブロックされる場合:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_sem.ps1
```

- GPUが使えない場合:
  - PyTorchのCUDA版インストールを確認
  - 自動的にCPU実行にフォールバックする箇所があります

## ライセンス

`LICENSE` を参照してください。
