# Problem4
> 經管碩二 0753732 陳弘承
## Description

Please read [this](https://github.com/Hong-CC/ECM9042_Deep_Learning/blob/main/Problem1/DL_HW2.pdf).

## Dataset

You can find [here](https://github.com/bchao1/Anime-Face-Dataset)

### i.
資料及內給的圖片大小為64x64，但在訓練的時候會出現resourceexhaustederror，所以選擇把資料resize成32x32的大小，只能折衷把圖片的解析度調降。
模型的設計，
encoding的部分是：
- Step 1: Convolution
filters = 64、kernel size = 5x5、strides = 2
- Step 3: Convolution
filter = 32、Kernel size = 3x3、strides =2
- Step 5: 將結果放到全連階層變成一個50維的output

dncoding的部分則是按照相反的順序完成deonvolution，最後再將資料變回32x32，channel=3的資料大小。

### ii.
Learning curve of negative ELBO using VAE
![](https://i.imgur.com/Mb5FRNx.png)


### iii.
| Real samples in dataset | Reconstruction samples using VAE |
| -------- |:--------:|
|![](https://i.imgur.com/c2XapjW.png)|![](https://i.imgur.com/4HKhVjF.png)|

這邊多嘗試使用在convolution layer後加上max pooling layer得到的結果是：

| Real samples in dataset | Reconstruction samples using VAE |
| -------- |:--------:|
|![](https://i.imgur.com/c2XapjW.png)|![](https://i.imgur.com/h529IUm.png)|

兩這看起來結果沒有差太多，但是在encoder的地方加上max pooling layer可以比較快地降低照片的維度。


### iv. Synthesized samples drawn from VAE
$p(z) \sim normal$其中平均數跟變異數，都是利用前面train出來的z值計算來的。
![](https://i.imgur.com/27bkfaj.png)

### v.Show the synthesized images based on the interpolation of two latent codes z between two real samples.

![](https://i.imgur.com/1o5kKU0.png)

### vi. Multiply the Kullback-Leibler (KL) term in ELBO by 100

##### negative ELBO
![](https://i.imgur.com/vGbD9AD.png)

| Real samples in dataset | Reconstruction samples using VAE |
| -------- |:--------:|
|![](https://i.imgur.com/c2XapjW.png)|![](https://i.imgur.com/mA6AWsD.png)|

##### Synthesized samples drawn from VAE
![](https://i.imgur.com/0CVWCKi.png)

##### Show the synthesized images based on the interpolation of two latent codes z between two real samples.

![](https://i.imgur.com/3jUxzQz.png)


### vii. Multiply the KL term by 0 in ELBO

##### negative ELBO
![](https://i.imgur.com/KnO9Csp.png)


| Real samples in dataset | Reconstruction samples using VAE |
| -------- |:--------:|
|![](https://i.imgur.com/c2XapjW.png)|![](https://i.imgur.com/ZBiNRaw.png)|

##### Synthesized samples drawn from VAE
![](https://i.imgur.com/KNFIf7l.png)

##### Show the synthesized images based on the interpolation of two latent codes z between two real samples.

![](https://i.imgur.com/5cXWwTA.png)

### viii.

比較明顯能看出差異的地方是，當KL term乘100的時候，所有生成的圖片都長得差不多，而當KL term乘0的時候，一開始在第iii小題，利用既有的圖片生成時，除了細部會有一些奇怪的色塊(像是隨機生成latent code生成的照片那樣色塊)但再利用隨機抽取的latent code時結果就爆掉了。
會有上述的結果，推測是機器在學習的時候，原本的KL term像是對照片加上一個noise tern，讓NN再判斷的時候，不因為input有一點點的差異而欲撤出完全不同的結果，所以當我們對KL term乘100時，這個時候noise會變得太大，導致所有input出來的結果都差不多，而當我們對KL term乘0時，這時候就會因為input有一點點不一樣而生出非常不同的圖，就像我們用p(z)抽latent code的結果時產生奇怪的結果。

