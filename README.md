# Pretrain-RoBERTa-ja
cpu上でRoBERTa事前学習を試してみる．\
モデルの構築にPytorchを使用．\
データセットに青空文庫から太宰治の[走れメロス](https://www.aozora.gr.jp/cards/000035/card1567.html "走れメロス")をお借りして使用しました．
学習にかかる時間：およそ8分．\
実際にはデータセットやプログラムの工夫が必要．
# 実行環境
Windows11 メモリ16GB\
python==3.8.18\
pytorch : [公式サイト](https://pytorch.org/ "pytorch")からダウンロード\
その他，必要なパッケージ
```
pip install -r requirements.txt
```