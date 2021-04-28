# Problem3
> 經管碩二 0753732 陳弘承
## Description

Please read [this](https://github.com/Hong-CC/ECM9042_Deep_Learning/blob/main/Problem1/DL_HW2.pdf).

## Dataset

The original dataset is provided and cleaned by TAs.

You can find [here](https://drive.google.com/file/d/1A4W3Zr6XjQ9epZwn0gbg6OKIERinYinh/view)

### i.

列出前10筆的相關係數矩陣：
![](https://i.imgur.com/ynDl7K2.png)


### ii. Data process

- Step 1: 一般研究會將0.7設定為高度相關的閥值，所以第一步先將所有相關係數絕對值 > 0.7的國家拿出來當作樣本放到dataset C。
- Step 2: 設定start index = I與sequence的長度 = L
- Step 3: 把第一步驟得到的C從第I天開始，照長度L排列，舉例來說，如果設定I = 2, L=10，那每個國家的資料都會被分成，第2天到第11天為第一個sequence、第12天到第21天為第二個sequence，以此類推。

### iii.
建立一個兩層的RNN，neuron個數分別為16, 16，再加上一層neuron各數=8的全連階層，最後output layer使用sigmoid當作activation function，在訓練100個epoch後結果如下：
(使用的start index = 1，L=2)
![](https://i.imgur.com/QNYLc0C.png)
![](https://i.imgur.com/X4sCrnX.png)


### iv.
比較不同天數的LSTM，這邊使用的LSTM架構跟前一小題RNN的neuron個數還有activaiton都相同。
![](https://i.imgur.com/RZzlZrt.png)
![](https://i.imgur.com/liaM3m3.png)

結果跟想像的不太一樣，本來以為會是時間區間維10天的資料會預測的比較準，畢竟一次給電腦比較多天的資料，但因為我選擇的資料處理的方式，在L=10的情況下data point總共只會剩下大概1000筆，而且其中有92%的資料的label都是0所以會比較難訓練，反而倒是L=2的預測小果比較好。

### v.
利用RNN給的結果畫出世界地圖如下
![](https://i.imgur.com/pJeEibq.png)
可以發現一些跟想像不太一樣的地方，像是疫情比較嚴重的歐美地區，都是分類到下降，儘管顏色比他國家深，代表算出來的機率值在0.5以下一點的地方，但還是被歸類到下降的。

### vi.
這一小題似乎因為資料處理的方式不夠細膩，所以儘管在訓練的時候可以得到還不錯的效果，但在預測上會顯得有點不夠力，像是資料筆數會因為選擇的L變大而導致資料筆數下降這個問題，但在練習的過程中，有試著用滾動的方式取樣，假設L=10取樣的結果會像是，第1到第11天一組，第2到第12天一組，雖然這樣能有效地防止資料筆數變少，但預測的決果卻沒有很理想預測準確率都大概在0.8左右而且有蠻嚴重的overfitting的問題。

