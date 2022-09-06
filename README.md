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
- Tabnet

### Multiome

- Tabnet Pre-train
- Tabnet


### Postprocess

#### Citeseq

- Publicテストリーク対応

#### Multiome

- 推論結果を 非負 にする



#### Ensemble

- 各推論を Standardize してからアンサンブルする
