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

## Score

CV weight -> cite:multi = 0.712:0.288

```
2022-10-26 21:58:31,975 [INFO][load_data] Load CITEseq inference data.
2022-10-26 21:58:31,976 [INFO][load_data]   -> 2022-10-11_01-48-10: 0.19653
2022-10-26 21:58:32,265 [INFO][load_data]   -> 2022-10-18_10-55-21: 0.94984
2022-10-26 21:58:32,539 [INFO][load_data]   -> 2022-10-18_11-33-29: 0.87416
2022-10-26 21:58:32,814 [INFO][load_data]   -> 2022-10-15_01-02-59: 0.41997
2022-10-26 21:58:33,247 [INFO][load_data]   -> 2022-10-15_01-16-43: 0.3216
2022-10-26 21:58:33,687 [INFO][load_data]   -> 2022-10-15_01-34-34: 0.13372
2022-10-26 21:58:34,125 [INFO][load_data]   -> 2022-10-15_01-44-29: 0.16316
2022-10-26 21:58:34,565 [INFO][load_data]   -> 2022-10-15_12-39-39: 0.31211
2022-10-26 21:58:35,004 [INFO][load_data]   -> 2022-10-15_12-50-55: 0.27789
2022-10-26 21:58:35,439 [INFO][load_data]   -> 2022-10-14_03-16-10: 0.15874
2022-10-26 21:58:35,686 [INFO][load_data]   -> 2022-10-18_00-16-49: 0.74503
2022-10-26 21:58:35,955 [INFO][load_data] Load Multiome inference data.
2022-10-26 21:58:35,955 [INFO][load_data]   -> 2022-10-16_13-08-46: 1
2022-10-26 22:00:18,263 [INFO][load_data]   -> 2022-10-16_13-30-35: 1
2022-10-26 22:02:05,304 [INFO][load_data]   -> 2022-10-17_00-25-56: 1
2022-10-26 22:04:10,814 [INFO][load_data]   -> 2022-10-17_13-19-49: 1
2022-10-26 22:05:45,760 [INFO][load_data] Load public submission.
2022-10-26 22:05:45,762 [INFO][load_data]   -> 5-5-msci22-ensembling-citeseq: 1
2022-10-26 22:06:46,478 [INFO][load_data]   -> all-in-one-citeseq-multiome-with-keras: 1
2022-10-26 22:07:46,988 [INFO][load_data]   -> uehara-san-2022-10-24-1906: 1.5
2022-10-26 22:13:57,347 [INFO][postprocess] All training data CV: 0.83707 (cite: 0.90414, multi: 0.67126)
2022-10-26 22:14:17,358 [INFO][postprocess] training data that similar test data CV: 0.83422 (cite: 0.90177(size: 13112), multi: 0.66723(size: 16702))
2022-10-26 22:14:50,765 [INFO][postprocess] Main submission weight: 2
2022-10-26 22:54:27,678 [INFO][postprocess] Done.
```
