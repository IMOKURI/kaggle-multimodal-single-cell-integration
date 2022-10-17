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
- [targets (ENSG番号) が含まれる inputsの配列名を抽出](https://github.com/vfr800hu/Multimodal_Single-Cell_Integration/blob/main/preprocess/gff3/MULTIOME%E3%82%BF%E3%83%BC%E3%82%B2%E3%83%83%E3%83%88%E9%81%BA%E4%BC%9D%E5%AD%90%E3%81%8C%E5%AD%98%E5%9C%A8%E3%81%99%E3%82%8B%E9%85%8D%E5%88%97.ipynb)
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
- MLP with Dropout (0.90199, 686)
- MLP with Dropout (0.90198, 686, tf-like initialization)
- ResNet (0.90139, 686)
- ResNet (0.90147, 686, tf-like initialization)
- 1D CNN (0.90189, 686)
- 1D CNN (0.90175, 686)

#### Multiome


##### Model (CV Score, Num of Feature, Ensemble weight)

- TabNet (0.66899, 3104)
- TabNet (0.66904, 3105)
- TabNet (0.66900, 3105, seed 1440)
- TabNet (0.66918, 3105, pre-training with training data)
- TabNet (0.66902, 3105, pre-training with all data)
- TabNet (0.66902, 3345, pre-training with all data)
- TabNet (0.66903, 3345, pre-training with all data, seed 2440)
- MLP with Dropout (0.66791, 3344)
- MLP with Dropout (0.66786, 3344, tf-like initialization)
- 1D CNN (0.67085, 3344)
- 1D CNN (0.67086, 3344, tf-like initialization)


### Postprocess

#### Citeseq



#### Multiome




### Ensemble

- 各推論を Standardize してからアンサンブルする
    - OOF と 推論結果をまとめて Standardize する

## Score

CV weight -> cite:multi = 0.712:0.288

- CV: 0.83647 (cite: 0.90370, multi: 0.67029), LB: 0.811
