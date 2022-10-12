# Open Problems - Multimodal Single-Cell Integration

![image](https://user-images.githubusercontent.com/1638500/195366837-9048d24c-86ca-42d6-99a8-414a019a5048.png)

## Solution

### Preprocess

#### Citeseq

- PCA で 240 次元に削減
- ターゲットの列名を含む インプットの列名のデータを温存 (200次元)
- [ivis](https://bering-ivis.readthedocs.io/en/latest/index.html) の教師なし学習で 240 次元の特徴量を生成
- 合わせてミトコンドリアのRNAの細胞ごとの和を特徴量に追加 (6次元)
- metadata の cell type (1次元)

#### Multiome

- 列名の接頭文字が同じグルーごとに PCA で およそ 各 100 次元に削減
    - さらに [ivis](https://bering-ivis.readthedocs.io/en/latest/index.html) の教師なし学習で 240 次元の特徴量を生成
- metadata の cell type (1次元)
    - test data は 事前学習したモデルでの推論結果を使用


### Training

#### Common

- Adversarial training で誤判断された細胞 1.3 万件ほどを正当ラベルとした StratifiedKFold
- Pearson Loss
- TabNet の pre-training

#### Citeseq


##### Model (CV Score, Num of Feature, Ensemble weight)

- TabNet (0.90175, 680)
- TabNet (0.90170, 681)
- TabNet (0.90186, 686)
- TabNet (0.90188, 687)
- TabNet (0.90213, 687, x2, pre-training with training data)
- TabNet (0.90216, 687, x2, pre-training with all data)
- Simple MLP (0.90098, 686)
- Simple MLP (0.90107, 686, tf-like initialization)

#### Multiome


##### Model (CV Score, Num of Feature, Ensemble weight)

- TabNet (0.66899, 3104)
- TabNet (0.66904, 3105)
- TabNet (0.66900, 3105, seed 1440)
- TabNet (0.66918, 3105, pre-training with training data)
- TabNet (0.66902, 3105, pre-training with all data)


### Postprocess

#### Citeseq



#### Multiome




### Ensemble

- 各推論を Standardize してからアンサンブルする
    - OOF と 推論結果をまとめて Standardize する

## Score

CV weight -> cite:multi = 0.712:0.288

- CV: 0.83599 (cite: 0.90327, multi: 0.66967), LB: 0.811
