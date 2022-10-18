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

To be updated...

#### Multiome


##### Model (CV Score, Num of Feature, Ensemble weight)

To be updated...


### Postprocess

#### Citeseq



#### Multiome




### Ensemble

- 各推論を Standardize してからアンサンブルする
    - OOF と 推論結果をまとめて Standardize する
- Optuna でアンサンブルの重み最適化
    - 評価指標には、Adversarial training で誤判断された細胞 3 万件ほど使用

## Score

CV weight -> cite:multi = 0.712:0.288

```
2022-10-17 12:20:04,729 [INFO][load_data] Load CITEseq inference data.
2022-10-17 12:20:04,730 [INFO][load_data]   -> 2022-10-03_12-37-37
2022-10-17 12:20:05,026 [INFO][load_data]   -> 2022-10-06_04-34-58
2022-10-17 12:20:05,306 [INFO][load_data]   -> 2022-10-03_12-58-59
2022-10-17 12:20:05,590 [INFO][load_data]   -> 2022-10-06_05-01-11
2022-10-17 12:20:05,870 [INFO][load_data]   -> 2022-10-11_01-09-07
2022-10-17 12:20:06,149 [INFO][load_data]   -> 2022-10-11_01-48-10
2022-10-17 12:20:06,429 [INFO][load_data]   -> 2022-10-12_00-50-18
2022-10-17 12:20:06,873 [INFO][load_data]   -> 2022-10-12_11-11-39
2022-10-17 12:20:07,327 [INFO][load_data]   -> 2022-10-15_01-02-59
2022-10-17 12:20:07,778 [INFO][load_data]   -> 2022-10-15_01-16-43
2022-10-17 12:20:08,205 [INFO][load_data]   -> 2022-10-15_01-34-34
2022-10-17 12:20:08,655 [INFO][load_data]   -> 2022-10-15_01-44-29
2022-10-17 12:20:09,080 [INFO][load_data]   -> 2022-10-15_12-39-39
2022-10-17 12:20:09,504 [INFO][load_data]   -> 2022-10-15_12-50-55
2022-10-17 12:20:09,930 [INFO][load_data]   -> 2022-10-17_00-13-14
2022-10-17 12:20:10,357 [INFO][load_data]   -> 2022-10-14_03-16-10
2022-10-17 12:20:10,603 [INFO][load_data] Load Multiome inference data.
2022-10-17 12:20:10,604 [INFO][load_data]   -> 2022-10-16_13-08-46
2022-10-17 12:22:02,761 [INFO][load_data]   -> 2022-10-16_13-30-35
2022-10-17 12:23:42,025 [INFO][load_data]   -> 2022-10-17_00-25-56
2022-10-17 12:29:00,501 [INFO][postprocess] All training data CV: 0.83684 (cite: 0.90382, multi: 0.67123)
2022-10-17 12:29:20,723 [INFO][postprocess] training data that similar test data CV: 0.83392 (cite: 0.90136(size: 13112), multi: 0.66720(size: 16702))
2022-10-17 12:29:20,724 [INFO][postprocess] Optimize cite ensemble weight.
2022-10-17 12:36:13,997 [INFO][postprocess] cite optimization result. CV: 0.90150, weight: [0.007376409412677783, 0.1166870161126573, 0.7801007217121276, 0.10170586956956369, 0.09139796603729966, 0.694426794195483, 0.6115378192689488, 0.20512529286503275, 0.35754757164433537, 0.32747650873586587, 0.42382351693118503, 0.48927543570485044, 0.9574686713449889, 0.8123202475805951, 0.11832344635996042, 0.9984121079364743]
2022-10-17 12:36:13,998 [INFO][postprocess] training data that similar test data optimized CV: 0.83402
2022-10-17 13:16:10,969 [INFO][postprocess] Done.
```
