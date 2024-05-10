# 转置卷积

​		也被称为反卷积，不会使用预先设定的插值方法，它具有可学习的参数。

​		转置卷积的步长与padding都是作用于输出结果上的，常规卷积是作用于输入数据上。

https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Transpose_Convolution.html

https://zh.d2l.ai/chapter_computer-vision/transposed-conv.html

# 可分离卷积

​		空间可分离，在空间维度将标准卷积运算进行拆分，将标准卷积核拆分成多个小卷积核。一个3x3拆成3x1和1x3的组合，由原来的9次乘法变成2个需要3次乘法。

​		深度可分离，将卷积核分成两个单独的小卷积核，分别进行2种卷积运算：深度卷积运算和逐点卷积运算。

​		a.深度卷积：对于12×12×3的输入图像而言，使用3个 5×5×1 的卷积核分别提取输入图像中3个 channel 的特征，每个卷积核计算完成后，会得到3个 8×8×1的输出特征图，将这些特征图堆叠在一起就可以得到大小为 8×8×3的最终输出特征图。这一点不同于常规卷积，常规卷积使用5x5x3输出一个8x8x1，要想得到多个8x8就必须使用多个5x5x3。深度卷积运算缺少通道间的特征融合 ，并且运算前后通道数无法改变。

​		b.逐点卷积：逐点卷积其实就是1x1卷积，因为其会遍历每个点，所以称之为逐点卷积。使用一个1x1x3的卷积对8x8x3特征图计算后得到8x8x1，这样就实现了通道的特征融合。

https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Separable_Convolution.html

# 膨胀卷积-空洞卷积

​		空洞卷积(Dilated Convolution)，在某些文献中也被称为扩张卷积（Atrous Deconvolution），是针对图像语义分割问题中下采样带来的图像分辨率降低、信息丢失问题而提出的一种新的卷积思路。引入扩张率（Dilation Rate）使得同样尺寸的卷积核获得更大的感受野。相应地，也可以使得在相同感受野大小的前提下，空洞卷积比普通卷积的参数量更少。

https://paddlepedia.readthedocs.io/en/latest/tutorials/CNN/convolution_operator/Dilated_Convolution.html

# dual-softmax

​		CAMoE(Improving Video-Text Retrieval by Multi-Stream Corpus Alignment and Dual Softmax Loss)中提出，先按列求softmax，再按行求softmax。减少以一个文本同时被多个视频检索。

# MNN-mutual-nearest-neighbor

MNN主要实现步骤：
假如我们有两个批次：批次1（m个细胞）和批次2（n个细胞）的scRNA-seq的基因表达数据
（1）将不同批次的基因表达谱信息按细胞进行余弦标准化（cosine normalization）；
（2）依次计算批次1中每个细胞B1i到批次2中所有细胞的欧式距离，其实际等同于表达数据标准化前的余弦距离。这样我们就得到m个向量存放欧式距离，每个向量里存放了n个欧式距离，再保存每个向量中ki个具有最小欧式距离的细胞对（nearest neighbor，NN）。比如批次1中细胞1，计算出n个欧式距离，里面有10个细胞具有最小欧式距离，我们就保存这10个细胞对（NNs）。再依次保存剩下的m-1个中具有最小欧式距离的细胞对（k1_1, k1_2, k1_3, k1_i..., k1_m）。k1_i表示每个欧式距离向量中具有最小的欧式距离的细胞对的数量。
（3）接下来，反过来对批次2，执行相同的步骤（2）。计算批次2中每个细胞到批次1中所有细胞的欧式距离，得到n个向量，每个向量里存放了m个欧式距离。然后，再保存每个欧式距离向量中具有最小欧式距离的细胞对(k2_1, k2_2, k2_3, k2_i, ..., k2_n)。k2_i表示具有最小的欧式距离的细胞对的数量。
（4）这样，我们比较这些配对的细胞，如果发现批次1和批次2中细胞互相配对的时候，那么，嘿嘿嘿，我们就保存这种细胞对，也称作互为邻接对（MNNs）。这种MNN的细胞，在本文中就被认为是同一类型的细胞了。
（5）利用MNN细胞对的表达信息，计算两两细胞间的基因表达差值，得到表达差异向量，也称为配对特异的批次效应校正向量（pair-specific batch convection vector）。同一种细胞，基因的表达模式应该相同或接近，那么这种表达差异向量就源于批次效应了。
（6）计算出来的所有的pair-specific 批次效应校正向量，利用高斯核函数，计算它们的加权平均数作为最后的批次效应校正向量，该向量就是唯一一个，长度为基因的个数。最后将其应用到批次2的所有细胞（不管属不属于MNNs的细胞）中进行批次效应的校正。

# 图像边界效应

​		边界效应是指当图像被放大或裁剪时，图像边缘的像素值会发生变化，导致图像质量下降。