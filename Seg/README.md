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

​		使用稀疏连接来代替Non-Local的密集连接。

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
