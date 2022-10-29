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

- Adversarial training で誤判断された細胞 3 万件ほどを正当ラベルとした StratifiedKFold
- Pearson Loss
- TabNet の pre-training

#### Citeseq


##### Model (CV Score, Num of Feature, Ensemble weight)

- TabNet
- MLP
- ResNet
- 1D CNN
- XGBoost


#### Multiome


##### Model (CV Score, Num of Feature, Ensemble weight)

- 1D CNN のみ


### Postprocess

#### Citeseq



#### Multiome




### Ensemble

- 各推論を Standardize してからアンサンブルする
    - OOF と 推論結果をまとめて Standardize する
- Optuna でアンサンブルの重み最適化する
    - 評価指標には、Adversarial training で誤判断された細胞 3 万件ほど使用
- Public submission と 上原さんの Submission をアンサンブルする
    - Public LB の状況見て weight を決めている

## Score

### LB

- Public: 0.812

### CV

Citeseq, Multiome weight: `0.712:0.288`

```
../output/2022-10-29_00-49-37

2022-10-29 00:50:00,033 [INFO][load_data] Load CITEseq inference data.
2022-10-29 00:50:00,034 [INFO][load_data]   -> 2022-10-15_01-02-59: 0.45
2022-10-29 00:50:00,503 [INFO][load_data]   -> 2022-10-15_01-16-43: 0.51
2022-10-29 00:50:00,953 [INFO][load_data]   -> 2022-10-15_01-34-34: 0.3
2022-10-29 00:50:01,409 [INFO][load_data]   -> 2022-10-15_01-44-29: 0.18
2022-10-29 00:50:01,866 [INFO][load_data]   -> 2022-10-15_12-39-39: 0.2
2022-10-29 00:50:02,317 [INFO][load_data]   -> 2022-10-15_12-50-55: 0.3
2022-10-29 00:50:02,777 [INFO][load_data]   -> 2022-10-14_03-16-10: 0.44
2022-10-29 00:50:03,067 [INFO][load_data]   -> 2022-10-18_00-16-49: 0.96
2022-10-29 00:50:03,353 [INFO][load_data]   -> 2022-10-27_12-50-48: 0.35
2022-10-29 00:50:03,636 [INFO][load_data]   -> 2022-10-27_12-55-11: 0.87
2022-10-29 00:50:03,895 [INFO][load_data]   -> 2022-10-28_14-34-19: 0.92
2022-10-29 00:50:04,182 [INFO][load_data]   -> 2022-10-28_14-46-38: 0.97
2022-10-29 00:50:04,439 [INFO][load_data]   -> 2022-10-28_14-52-22: 0.98
2022-10-29 00:50:04,691 [INFO][load_data]   -> 2022-10-28_14-54-39: 0.91
2022-10-29 00:50:04,947 [INFO][load_data] Load Multiome inference data.
2022-10-29 00:50:04,948 [INFO][load_data]   -> 2022-10-16_13-08-46: 1
2022-10-29 00:51:26,581 [INFO][load_data]   -> 2022-10-16_13-30-35: 1
2022-10-29 00:52:48,508 [INFO][load_data]   -> 2022-10-17_00-25-56: 1
2022-10-29 00:54:19,961 [INFO][load_data]   -> 2022-10-17_13-19-49: 1
2022-10-29 00:56:21,169 [INFO][load_data] Load public submission.
2022-10-29 00:56:21,170 [INFO][load_data]   -> 5-5-msci22-ensembling-citeseq: 1
2022-10-29 00:57:22,080 [INFO][load_data]   -> all-in-one-citeseq-multiome-with-keras: 1
2022-10-29 00:58:22,835 [INFO][load_data]   -> uehara-san-2022-10-28-0749: 1
2022-10-29 01:05:42,322 [INFO][postprocess] All training data CV: 0.83726 (cite: 0.90441, multi: 0.67126)
2022-10-29 01:06:03,016 [INFO][postprocess] training data that similar test data CV: 0.83448 (cite: 0.90213(size: 13112), multi: 0.66723(size: 16702))
2022-10-29 01:06:48,531 [INFO][postprocess] Main submission weight: 2
```
