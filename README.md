# Open Problems - Multimodal Single-Cell Integration

Predict how DNA, RNA & protein measurements co-vary in single cells


## Solution

### Preprocess

#### Citeseq

- PCA で 240 次元に削減
- ターゲットの列名を含む インプットの列名のデータを温存
- metadata の cell_type を特徴量に追加

#### Multiome

- 列名の接頭文字が同じグルーごとに PCA で およそ 各 100 次元に削減
- metadata の cell_type を特徴量に追加


### Training

### Citeseq

- Tabnet Pre-train
    - 学習データ・テストデータで教師なし学習（穴埋め）
- Tabnet (no tuning yet)
    - 学習データ 5 fold

### Multiome

- Tabnet Pre-train
    - 学習データ・テストデータで教師なし学習（穴埋め）
- Tabnet (no tuning yet)
    - 学習データ 5 fold


### Postprocess

#### Citeseq

- Publicテストリーク対応

#### Multiome

- 推論結果を 非負 にする



#### Ensemble

- 各推論を Standardize してからアンサンブルする
