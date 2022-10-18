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
2022-10-18 12:15:57,983 [INFO][load_data] Load CITEseq inference data.
2022-10-18 12:15:57,984 [INFO][load_data]   -> 2022-10-18_10-55-21
2022-10-18 12:15:58,306 [INFO][load_data]   -> 2022-10-18_11-33-29
2022-10-18 12:15:58,599 [INFO][load_data]   -> 2022-10-12_00-50-18
2022-10-18 12:15:59,181 [INFO][load_data]   -> 2022-10-12_11-11-39
2022-10-18 12:15:59,770 [INFO][load_data]   -> 2022-10-15_01-02-59
2022-10-18 12:16:00,342 [INFO][load_data]   -> 2022-10-15_01-16-43
2022-10-18 12:16:00,900 [INFO][load_data]   -> 2022-10-15_01-34-34
2022-10-18 12:16:01,468 [INFO][load_data]   -> 2022-10-15_01-44-29
2022-10-18 12:16:01,993 [INFO][load_data]   -> 2022-10-15_12-50-55
2022-10-18 12:16:02,589 [INFO][load_data]   -> 2022-10-17_00-13-14
2022-10-18 12:16:03,166 [INFO][load_data]   -> 2022-10-14_03-16-10
2022-10-18 12:16:03,522 [INFO][load_data] Load Multiome inference data.
2022-10-18 12:16:03,522 [INFO][load_data]   -> 2022-10-16_13-08-46
2022-10-18 12:17:57,379 [INFO][load_data]   -> 2022-10-16_13-30-35
2022-10-18 12:20:06,484 [INFO][load_data]   -> 2022-10-17_00-25-56
2022-10-18 12:22:44,774 [INFO][load_data]   -> 2022-10-17_13-19-49
2022-10-18 12:31:37,825 [INFO][postprocess] All training data CV: 0.83690 (cite: 0.90391, multi: 0.67126)
2022-10-18 12:31:58,183 [INFO][postprocess] training data that similar test data CV: 0.83402 (cite: 0.90149(size: 13112), multi: 0.66723(size: 16702))
2022-10-18 12:31:58,185 [INFO][postprocess] Optimize cite ensemble weight.
2022-10-18 12:42:55,117 [INFO][postprocess] cite optimization result. CV: 0.90161, weight: [0.2640474387440314, 0.946477353882143, 0.019359167217375066, 0.07263067920479108, 0.8004125030010443, 0.23299235688015946, 0.14623894627162803, 0.08619918775342417, 0.19411397275938902, 0.1314448260365039, 0.816985969895111]
2022-10-18 12:42:55,118 [INFO][postprocess] training data that similar test data optimized CV: 0.83411
```
