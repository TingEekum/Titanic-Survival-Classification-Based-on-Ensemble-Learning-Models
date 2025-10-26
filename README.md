# 基於集成學習模型的鐵達尼號生存分類研究
本研究採用了[鐵達尼號資料集](https://www.kaggle.com/competitions/titanic)作為分析、預測的對象，並使用了集成學習演算法進行模型建構，包括隨機森林、梯度提升、極限梯度提升(XGBoost)及旋轉森林，在最終比較這四者的績效表現。
<!-- This study uses the Titanic dataset and applies ensemble learning models, including Random Forest, Gradient Boosting, XGBoost, and Rotation Forest, for classification prediction, while comparing their performance. -->
## 資料組成
本資料集含有train.csv, test.csv, gender_submission.csv三個檔案。train.csv包含了第1到第891號乘客的資料與是否倖存結果，在此將作為訓練集訓練模型；test.csv, gender_submission.csv則是分別包含了第892到第1309號乘客的資料及是否倖存結果，在此將作為測試集預測結果並查看模型績效。<br/>
訓練集包含了12個欄位，分別是:<br/>
| 參數 | 定義 |
| ---- | ---- |
| PassengerId | 乘客編號 |
| Survived | 是否倖存 |
| Pclass | 船票等級 |
| Name | 乘客姓名 |
| Sex | 乘客性別 |
| Age | 乘客年齡 |
| SibSp | 乘客在船上的配偶/手足人數 |
| Parch | 乘客在船上的父母/子女人數 |
| Ticket | 船票號碼 |
| Fare | 票價 |
| Cabin | 艙位號碼 |
| Embarked | 登船的港口 |

## 資料前處理
### 考慮資料欄位的去留
<img width="1149" height="405" alt="ori_data" src="https://github.com/user-attachments/assets/8050e553-33c2-4135-93b5-f8b2bfdbffc8" />
在觀察完資料及其組成後，可以發現PassengerId、Ticket、Cabin對於標籤Survived是較無意義的，故先將這三個欄位刪除。<br/><br/>
接著從Name一欄可以觀察到，每個人名中都有包含了稱謂，我們可以將這些稱謂(包括: Mr, Mrs, Miss, Master, Don, Rev, Dr, Mme, Ms, Major, Lady, Sir, Mlle, Col, Capt, the Countess, Jonkheer)提取出來，作為新的資料欄位Title，並把稱謂合併成六個種類: Mr, Mrs(包括原先的Mrs, Mme), Miss(包括原先的Miss, Ms, Mlle), Master, Rare_Male(包括原先的Don, Rev, Dr, Major, Sir, Col, Capt, Jonkheer), Rare_Female(包括原先的Lady, the Countess)，最後將Name欄位刪除。<br/>

### 處理空值
<img width="341" height="376" alt="data_info" src="https://github.com/user-attachments/assets/9b6f1a2d-cb66-46cb-847d-fbff61315515" /><br/>
由上圖可以觀察到欄位Age及Embarked含有空值，因此將Age的空值以該稱謂的中位數填入，避免出現稱謂為Master年齡卻為30的情況；Embarked的空值則以眾數(出現次數最多的類別)填入。<br/>

### 類別欄位進行編碼
<img width="315" height="325" alt="data_type" src="https://github.com/user-attachments/assets/04f7d86b-e1fc-44eb-9234-8e23c7c32620" /><br/>
因Sex, Embarked, Title皆為類別欄位，需先將它們編碼，轉換成數字形式的資料後才能夠使用。

### 特徵選取
本次使用了[相互資訊(Mutual Information)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)來作為特徵篩選的依據。相互資訊可用於衡量變數之間的依賴關係，其數值越大依賴關係越強。

## 模型建構與資料預測
本次研究此用了四種集成學習模型，包括隨機森林、梯度提升、極限梯度提升及旋轉森林，並搭配使用交叉驗證取得模型最佳參數。
以下為測試集的預測結果的混淆矩陣:
| 隨機森林(Random Forest) | 梯度提升(Gradient Boost) | 極限梯度提升(XGBoost) | 旋轉森林(Rotation Forest) |
| :---: | :---: | :---: | :---: |
| <img width="150" height="130" alt="cm_rf" src="https://github.com/user-attachments/assets/5a3dabde-fe0b-44cf-b3ea-2750b06b23ea" /> | <img width="150" height="130" alt="cm_gb" src="https://github.com/user-attachments/assets/729c734d-8365-4fd7-981d-4246736299c1" /> | <img width="150" height="130" alt="cm_xgb" src="https://github.com/user-attachments/assets/653c6aaf-2abe-4ca4-a02a-8c6b5086a63e" /> | <img width="150" height="130" alt="cm_rotf" src="https://github.com/user-attachments/assets/02c88ca6-69c6-41ef-83dc-987eb6cd8e59" /> |<br/>

接著，以下是訓練集與測試集的預測結果的準確率、精確率、召回率及f1-score:
| 準確率(Accuracy) | 精確率(Precision) |
| :---: | :---: |
| <img width="790" height="490" alt="acc" src="https://github.com/user-attachments/assets/b1470f98-0c37-42dd-9f10-1c01e1b49588" /> | <img width="790" height="490" alt="precision" src="https://github.com/user-attachments/assets/92598360-07ea-4a41-9896-e38fb5623ce7" /> |
| **召回率(Recall)** | **f1-score** |
| <img width="790" height="490" alt="recall" src="https://github.com/user-attachments/assets/95101e2b-420d-4393-a505-2e5bca32692a" /> | <img width="790" height="490" alt="f1-score" src="https://github.com/user-attachments/assets/ea1d4962-64f4-4901-a570-5166ca8dcd2b" /> |<br/>

以結果來看，四個模型在四種指標中都有相當好的表現，其中旋轉森林的訓練集與測試集績效相差較大，可能會有過擬合的情形。<br/><br/>
最後，以下是四個模型的ROC curve及AUC:
| 隨機森林(Random Forest | 梯度提升(Gradient Boost) |
| :---: | :---: |
| <img width="567" height="455" alt="roc_rf" src="https://github.com/user-attachments/assets/5ccc66da-3705-4b82-9fac-15ad1db45197" /> | <img width="567" height="455" alt="roc_gb" src="https://github.com/user-attachments/assets/524613d8-9529-47e3-918d-995a5ce46254" /> |
| **極限梯度提升(XGBoost)** | **旋轉森林(Rotation Forest)** |
| <img width="567" height="455" alt="roc_xgb" src="https://github.com/user-attachments/assets/dddd2d6b-cd4c-4cd1-910e-2c47b968cee3" /> | <img width="567" height="455" alt="roc_rotf" src="https://github.com/user-attachments/assets/119f390a-eb44-45ac-8307-d1352c94d89d" /> |<br/>

四個模型的AUC皆接近1，驗證了模型分類成效佳的說法。
