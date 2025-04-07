# Prompt Image Editor
`prompt-image-editor`は、SDSロスによる潜在空間操作を使用し、テキスト入力での FashionMNIST の画像編集を行うPythonベースのプロジェクトです。

<div align="center">
<img width="421" alt="image" src="assets/output.gif">
</div>

## インストール

1. リポジトリをクローンします:
   ```bash
   git clone https://github.com/your-username/prompt-image-editor.git
   cd prompt-image-editor
   ```

2. 必要な依存関係をインストールします:
	```
   conda create -n prompt-image-editor python=3.10 -y
   conda activate prompt-image-editor

	pip install -r requirements.txt
   pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
	```

## 使用方法
### トレーニング
モデルをトレーニングするには、train.pyスクリプトを使用します:
```
python train.py
```

### 画像編集
画像編集パイプラインを実行します:
```
python main.py --input assets/image-1.jpg --output_dir output/image-1 --prompt "Long T-shirts"
```
