# Open Problems - Multimodal Single-Cell Integration

Predict how DNA, RNA & protein measurements co-vary in single cells


## Solution

### Preprocess

#### Citeseq

- PCA で 240 次元に削減
- ターゲットの列名を含む インプットの列名のデータを温存
- [ivis](https://bering-ivis.readthedocs.io/en/latest/index.html) の教師なし学習で 240 次元の特徴量を生成
- 合わせてミトコンドリアのRNAの細胞ごとの和を特徴量に追加

#### Multiome

- 列名の接頭文字が同じグルーごとに PCA で およそ 各 100 次元に削減
    - さらに [ivis](https://bering-ivis.readthedocs.io/en/latest/index.html) の教師なし学習で 240 次元の特徴量を生成


### Training (CV Score, Num of Feature)

#### Common

- Adversarial training で誤判断された細胞 1.3 万件ほどを正当ラベルとした StratifiedKFold

#### Citeseq

- Ridge (0.89373, 680)
- Tabnet (0.89482, 680)
- Tabnet (0.89468, 686)
- XGBoost (0.89416, 926)

#### Multiome

- Tabnet (0.66752, 3104)
- Tabnet (0.66712, 3104, seed 1440)


### Postprocess

#### Citeseq



#### Multiome

- 推論結果を 非負 にする



### Ensemble

- 各推論を Standardize してからアンサンブルする
    - OOF と 推論結果をまとめて Standardize する

## Score

- CV: 0.83319 (cite: 0.89983, multi: 0.66843), LB: 0.809
