# 分割常用指标评价

令 i 为要预测的类别，j为其他类别

n_ii 为类别 i 被预测为类别 i 的像素数量，指 True Positive

n_ij 为类别 i 被预测为类别 j 的像素数量，指 False Negative

n_ji 为类别 j 被预测为类别 i的像素数量，指 False Positive

n_cl 为所有的类别总数 (分类+背景)

t_i 是指所有 n_ij 的总数+ n_ii 的总数，也就是真正的类别 i 的总像素数量

Pixel accuracy： 预测正确像素数量占类别i总像素数量的比例

mean accuracy：基于Pixel accuracy 所改进的指标，先计算每个类别的预测准确率再求平均

mean IU (mIOU)：计算所有像素类别的平均IOU

frequency weighted IU：基于mIOU所改进的指标，会依据每个类别出现的频率设置权重

# 2019

## APCNet

Adaptive Pyramid Context Network for Semantic Segmentation

​		仍然围绕上下文信息，作者目的在于找到最优的上下文向量，并且这个向量尽可能紧凑以减少无关信息的干扰。

​		核心点在于作者提出的ACM(Adaptive Contect Module)，ACM有两个分支，第一个分支学习Affinity亲和参数(理解为相关性)，第二个分支就是常规的adaptive average pooling操作，最终对两个分支的特征进行matrix product。

## DANet

Dual Attention Network for Scene Segmentation

​		使用双注意力网络来集成局部特征和全局依赖关系。以往的上下文融合的方法并没有从全局来考虑不同目标特征之间的关系。

​		核心在于提出的position attention和channel attention(每个高级特征的通道映射都可以看作是一个类特定的响应，不同的语义响应相互关联。通过挖掘通道映射之间的相互依赖性，强调相互依赖的特征映射，改善特定语义的特征表示)。

## Semantic FPN

Panoptic Feature Pyramid Networks

​		全局分割的概念：一张图中，有固定形态的物体认为是“thing”，使用instance segmentation割出来作为各类前景；对于无固定形态的物体认为是“stuff”，使用semantic segmentation割成各类背景。要实现全景分割，必须完成semantic / instance segmentation这两个任务。

​		将语义分割和实例分割的FCN和Mask R-CNN结合起来，设计了 Panoptic FPN。

## DMNet

Dynamic Multi-scale Filters for Semantic Segmentation

​		APCNet同一个作者，对结构做了一些改动(甚至图都是一样)。ACM变为DCM，同样的两个分支，只是第二个分支经过Adaptive pooling后再经过1*1卷积得到[K, K]大小的特征图，后面将这个作为卷积与第一个分支的输出进行深度可分离卷积的计算。

## CCNet

CCNet: Criss-Cross Attention for Semantic Segmentation

​		Non-Local需要的计算量太大，复杂度从(HxW)x(HxW)降为(HxW)x(H+W-1)。整幅图的attention变成了求十字路径的attention。

## EMANet

Expectation-Maximization Attention Networks for Semantic Segmentation

​		自注意力机制在分割中非常有帮助，但是计算量太大。作者提出EM算法来迭代一组紧凑的基，在这个基上使用注意力，从而减少计算量。

​		最主要的是这种思想：基当做参数，attention map当做隐变量；随机值作为初始的基，E步计算出隐变量的概率分布，M步通过隐变量的概率来迭代参数。

## ANN

Asymmetric Non-local Neural Networks for Semantic Segmentation

​		同样是从Non-Local计算量太大的角度出发，第一步**APNB** Asymmetric Pyramid Non-local Block减少self-attention时key和value的通道数，降低通道采用金字塔采样，对于输入的特征图使用pool得到4个尺度(1x1 3x3  6x6 8x8)，展平成1D再堆叠到一起；第二步**AFNB** Asymmetric Fusion Non-local Block利用高层特征H和低层特征L，L经过卷积得到K和V，H经过卷积得到Q，QK相乘再和V相乘起到特征融合的作用。

## GCNet

GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond

​		Non-Local对于不同的查询点起attention map是一样的，他的全局依赖是位置无关的(任务不一样。语义分割需要对每个像素都输出，所以要“雨露均沾”。分类只需关注最重要的概念就OK；而检测正例数量远远小于反例。只focus正例)。SENet则只考虑了通道间的依赖关系。结合二者使得model能够有长距离的长下文信息还能够比较轻量。

​		核心思想简化了non-local block，本身就是不受位置依赖，那么直接将key经过softmax加到value上。1x1卷积用bottleneck transform模块来取代，降低参数量。两层bottleneck transform增加了优化难度，在ReLU前面增加一个layer normalization层(降低优化难度且作为正则提高了泛化性)。

## FastFCN

FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

​		空洞卷积能获得更大的感受野同时不减小特征图分辨率，提高语义分割精确度，但空洞卷积输出使得分辨率变大，增加了计算开销。作者 提出了Joint Pyramid Upsampling(JPU) 来提取高分辨率特征图。

## Fast-SCNN(实时性友好)

Fast-SCNN for Semantic Segmentation

![scnn](./imgs/fastscnn.png)



# 2018

## BiSeNetV1

BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

![bisenet](./imgs/BiSeNet.png)

## PSANet

PSANet: Point-wise Spatial Attention Network for Scene Parsing

有一种空间KxL和通道(2H-1)x(2W-1)注意力的目的。

![psa](./imgs/psanet.png)

## ICNet

ICNet for Real-Time Semantic Segmentation on High-Resolution Images

![icnet](./imgs/icnet.png)

## UPerNet

Unified Perceptual Parsing for Scene Understanding

![upernet](./imgs/upernet.png)

## DeepLabV3+

Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

![v3plus](./imgs/v3plus.png)

## EncNet

Context Encoding for Semantic Segmentation

![encnet](./imgs/encnet.png)

## Non-Local Net

Non-local Neural Networks

![non](./imgs/noloc.png)

# 2017

## ERFNet

ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation

空间可分离卷积

![2](./imgs/ERFNet.png)

## PSPNet

Pyramid Scene Parsing Network

![3](./imgs/pspnet.png)

## DeepLabV3

Rethinking Atrous Convolution for Semantic Image Segmentation

![4](./imgs/deeplabv3.png)

# 2016

## UNet

U-Net: Convolutional Networks for Biomedical Image Segmentation

![1](./imgs/unet.png)

相关部署仓库-https://github.com/cagery/unet-onnx

# 2015

## FCN

Fully Convolutional Networks for Semantic Segmentation

​		使用转置卷积(双线性插值核)来替代全连接层。暴力的上采样必定丢失较多细节，而且在对各个像素进行分类时，没有考虑像素与像素之间的关系。
