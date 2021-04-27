# Problem 2
> 經管碩二 0753732 陳弘承
## Description

Please read [this](https://github.com/Hong-CC/ECM9042_Deep_Learning/blob/main/Problem1/DL_HW1.pdf).

## Dataset

The original dataset comes from Eden Social Welfare Foundation.

Provided and cleaned by TAs.

### i.

這邊我做的資料處理只有：
1. 使用助教給的人臉位置的資料，將照片上的人臉切下來。
2. 將切下來的人臉照片處理成80x80的大小。
3. 將圖片的特徵值(像素)都除以225，讓特徵值全都介於0~1之間。

最後呈現出來的圖片大概像底下這樣：

![](https://i.imgur.com/1qAuNDT.png)


題目中問到的是，因為每一張照片都有不同的解析度，還有臉部的大小不一樣，有沒有因此做任何的資料處理，這邊我並沒有特別針對這兩點做處理，只有將資料大小全部變成80x80的尺寸，因為畫質比較差，或是針對照片拿掉偶數行，奇數列之類的方式並不會影響對圖片的片讀，像是在Max pooling，就是在特定大小的像素內挑出最大的像素，雖然有些特徵不見了，但不影響我們對圖片的判讀，還可以大幅度的降低資料的維度。另外將圖片的特徵值都除以225是因為像素的數字界於0到225之間，如果同除225可以讓數值降為0~1之間，這個做法像是Normalize，這樣可以讓CNN運算得更快。

### ii.
此小題使用的架構如下：
* channel number = [32, 32]
* stride size = 1
* hidden layer = [1024, 512]

我們比較的標的是兩個不同的filter size，一個選用3x3的大小，另一個則選用9x9的大小，結果如下圖所示：


|Accuracy rate|Training loss|
|:-----:|:-----:|
| ![](https://i.imgur.com/OAcxh0w.png) | ![](https://i.imgur.com/BjoUnqK.png) |

這邊沒有附上filter是9x9的training loss的原因是，filter loss的training loss就只是一條水平的線躺在15000的位置，如果畫在一起會影響filter為3x3的圖形，另外畫也沒有太大的意義。


可以發現，filter size為9x9的train and test accuracy rate都沒有變過，從confustion matrix可以發現，這個模型把所有的結果都預測為good，所以表現得比較不好，這四相比之下，filter size為3x3的模型表現的就還不錯，至少在預測上相對的是比較精準的，只是發現有overfitting的問題，test data的accuracy rate在約莫20個epoch的時候預測準確率最高，來到0.944，但後面幾個epoch的accuracy rate卻一路往下掉，推測原因是train data的預測準確率已經來到1，後面的epoch再更新參數只會讓模型更難預測沒看過的資料。

|9x9 test data|3x3 test data|
|:-----:|:-----:|
| ![](https://i.imgur.com/zLJr1iC.png) |![](https://i.imgur.com/6qMUMpB.png)|

### iii.
#### (1)
這邊先提供前一小提filter大小為3x3各種不同標籤的預測準確率：

![](https://i.imgur.com/Xt40om0.png)


從上表可以看到none (wrongly wearing mask)的效果特別的差，只有不到一半的準確率，none預測成good與bad的機率大概一半一半，並沒有把none都預測成某個特定類別的情況。
#### (2)
這個小題使用了三種方法，一是dropout；二是weighted cross entropy；三是綜合前兩種方法。
- 方法一：Dropout
經由第 (ii) 小題的實作，可以發現模型似乎有overfitting的問題，也許這是讓最後的預測沒那麼準確的原因之一，所以決定先嘗試透過dropout來看看能不能解決這個問題。
dropout是常用來處理overfitting的正則化方法，其運作的方式是在訓練的時候，每次迭代都用一定的機率drop掉一些neuron，這邊我們設定neuron被drop的機率是0.5，而被drop掉的neuron就無法傳遞訊息，因此可以在訓練的時候不會讓架構過於依賴某個neuron，達到對抗overfitting的效果。而最後的結果顯示在下面：


|Accuracy rate|Accuracy of each class of test data|
|:-----------:| -------- |
|![](https://i.imgur.com/ViLJzuS.png)|![](https://i.imgur.com/FBRGuhp.png)|

- 方法二：Weighted cross entropy
透過觀察資料的數量可以知道類別的數量是不平衡的，絕大多數的資料都是good，占了整體資料的80%，bad只佔了17%，none更只有3%，所以其實CNN就算完全忽略none只要能分對good跟bad還是有97%的預測正確率，所以這邊希望可以提升深度學習對於none的預測效果，而weighted cross entropy則是針對cross entropy進行權重的調整，比較一般的cross entropy式子和balanced cross entropy就能明顯的看到差別：
一般的cross entropy：
$$H(p,q) = -\sum p \cdot log(q) $$
weighted cross entropy:
$$WCE = -\sum w \cdot p \cdot log(q)$$
因為我們在計算的目標是最小化loss function，這邊我使用的$w$是$1/class\_size$，也就是資料筆數比較少的，$w$會比較大，也就是當錯誤預測數量比較少的類別的時候，會有比較大的逞罰項，讓loss變大，以達到讓NN重視資料筆數較少的類別。
而結果顯示如下：

|Accuracy rate| Accuracy of each class of test data|
|:------:|:------:|
|![](https://i.imgur.com/4uk32Pf.png)|![](https://i.imgur.com/Hbf9992.png)|

- 方法三：Dropout + Weighted cross entropy
結果顯示如下：

|Accuracy rate|Accuracy of each class of test data|
|:--------:|:--------:|
|![](https://i.imgur.com/WrTpTC5.png)|![](https://i.imgur.com/5G4j03O.png)|

#### (3)

再放一次各個方法的比較：
|Original|Dropout|Weighted CE|Dropout + Weighted CE|
|:------:|:------:|:--------:|:----:|
|![](https://i.imgur.com/Xt40om0.png)|![](https://i.imgur.com/FBRGuhp.png)|![](https://i.imgur.com/Hbf9992.png)|![](https://i.imgur.com/5G4j03O.png)|

在我嘗試的方法裡面，針對loss function進行調整的weight CE在none這個類別表現的比較好，從原本的36%進步到50%，但相對的可以看到，在good這一項，預測的準確率就有點下降了。
