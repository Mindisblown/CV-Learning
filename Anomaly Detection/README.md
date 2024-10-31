# Anomaly Detection流派

## Reconstruction



## Teacher-Student

​		在仅包含正常样本的数据集上，让pretrained的teacher模型去教没有pretrain的student模型，使得teacher模型和student模型输出的embedding尽可能一致。那么在inference时，由于teacher只教过student如何embed正常样本，所以正常样本上teacher模型和student模型输出的embedding会比较相似，但异常样本上两者输出的embedding差异会比较大；



## Normalizing Flows

​		本质上是用一个简单的分布来表示真实分布。具体实现通过一些列change of variable(简单变化，必须可逆且方便计算det)组合。一次变化称为一个flow。因为可逆我就可以从简单分布sample一个点来生成真实分布的数据，必须可以知道这个图片发生的概率有多大。至于det是反应每个变化时物体体积的变化。



## Memory Bank

​		本质上要构建一个空间存取所有的特征信息，后续来的数据映射到这个空间上算特征距离来判断是否异常。



# One-CLass

## 2023-SimpleNet: A Simple Network for Image Anomaly Detection and Localization

​		不直接使用预训练的特征，使用feature adaptor来生成有目标倾向的特征，以此来减少domain bias；不直接合成数据，而是在正常样本的特征空间添加noise，需要对噪声的尺度进行校准，才能获取边界紧密的特征空间；不使用统计学的方式，训练一个简单的辨别器来识别异常区域。

# 大模型

## 2024-RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection

​		Strength-controllable Diffusion Anomaly Synthesis (SDAS)扩散策略生成不同异常的样本；Anomaly-aware Features Selection (AFS)选择具有代表性和区分性的预训练特征子集的方法，消除预训练的偏差，以提高异常检测性能；Reconstruction Residuals Selection (RRS)，一种自适应选择具有区分性残差的策略，用于全面识别多个粒度级别上的异常区域。AFS在每一层内选择最具判别性的特征子集，而RRS侧重于层的选择。不同的层适用于检测不同尺度和语义的异常区域。例如，低层特征能有效检测小面积的纹理异常，而对于大面积的功能性异常可能造成漏检或检测区域不连续。同理，高层特征也无法有效检测低级纹理异常。RRS仅保留包含最多异常信息的部分重建误差用来生成最终的anomaly map，以最大程度缓解异常漏检。

## 2024-AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models

​		语言模型与CV结合。

## 2023-Segment Any Anomaly without Training via Hybrid Prompt Regularization

多模态的工作，采用clip的思想使用语言模型。

## 2024-Text-Guided Variational Image Generation for Industrial Anomaly Detection and Segmentation

多模态clip

## 2024-Real-IAD: A Real-World Multi-view Dataset for Benchmarking Versatile Industrial Anomaly Detection

开源数据集

# RE

## 2021-Draem-a discriminatively trained reconstruction embedding for surface anomaly detection

​		生成模型通过重建正常区域来识别异常情况，只在normal的数据上训练，需要手工设计后处理来定位异常区域；没看见过异常数据那么就没有良好的特征学习，因此无法发挥最大的检测性能。将异常检测视为一个discriminative问题(辨别问题，辨别OK或者NG？)。

​		一种通用的方法是将整张图片看做异常或非异常，但是异常只占很小的一部分时，他们的分布是非常接近的。在工业的质量管控中，异常的数据很少，不能来一种就标记，这很耗费时间。

​		重建子网：原图经过破坏得到I_a，他要去重建出没破坏之前的I；之前使用的L2将相邻的像素看做独立是不对的，因此使用基于patch的SSIM loss。那么优化目标就变为最小化原始图像I和重建图像I_r之间的差异，ssim越大表示越相似那么优化目标就变为1-ssim。

​		辨别子网：I_r与I进行通道堆叠为I_c，预测一个异常map，二值图为M_a，计算与GT的mask图M的损失。



网络三大模块：异常生成模块、重构模块、判断模块。

异常生成模块：因为只有正常数据，通过将异常区域做到正常图里达到模拟异常图的效果。

重构模块：将产生的异常图进行重构得到重构图，可以产生损失Lr。

判别模块：重构图与异常图一同输入，通过该模块的层级结构对比产生分割真值图

​		推理时两个子网，时间不一定扛得住？本质上学习好坏之间的差异，从而知道好的样品是什么样子，在部署的时候能不能直接砍掉重建网络，拿原图和一张好的模板图concat调用辨别网络。

​		由于模型的泛化能力，在训练时解码器获得了很好的特征提取能力，编码器也获得了很好的重建能力。这导致测试时模型不仅能很好地重构正常样本，异常样本也能被很好的重构出来。这样，正常样本和异常样本的重构误差就没有那么泾渭分明了，从而导致混淆。

## ~~2022-DSR: A dual subspace re-projection network for surface anomaly detection~~

矢量量化(Codebook)：将一个向量空间中的点用一个有限子集来进行编码。最直接的例子就是PQ算法：

1.将高维向量分解为若干子向量；

2.分别在子向量中进行聚类，子向量空间中聚类中心的集合我们称为codebook；

3.计算子向量与聚类中心的距离，为每个子向量分配对应的codebook索引；

4.将子向量的索引组合起来，即可代表高维向量。

Quantized latent space encoder

​		输入图像经过ResNet下采样不同的倍数，在4x和8x的时候使用不同的codebook来将特征进行量化表示，得到两个不同层级的量化表示空间；

General object appearance decoder

​		低分辨率的量化表示进行上采样与高分辨率的量化表示进行concat得到重建图像；

Object-specific appearance decoder

说白了codebook其实就是embedding sapce，作者在特征层级来生成异常数据。

# TS

## 2022-Anomaly Detection via Reverse Distillation from One-Class Embedding

​		L_t为非异常数据，L_q为query集合包含正常和异常样本。目标在于识别定位L_q。两个数据中的正常样本分布是一致的，ood( Out-of-distribution)样本则为异常。

​		在无监督的AD任务中，查询样本为异常时，s模型期望输出与t模型不一样的表征，但是这种差异有时会消失，导致检测失败。

​		因为teacher和student网络相似，student学的非常好，有了和teacher很接近的能力，导致在推理阶段输入异常样本，student网络也是有很大的可能重建的和teacher很接近，那么最后的Loss也会比较小，这样对异常检测不利。而且传统知识蒸馏中的student网络都要比teacher小，小网络的重建性能是需要得到质疑的，如果重建能力不是那么好，不管正常还是异常都会产生问题。

​		针对上面提出的问题，作者提出了反转的知识蒸馏方案，将teacher作为encoder，student作为decoder,并且在teacher和student之间加入了一个one-class bottleneck embedding (OCBE) 模块。

​		在OCBE模块中，包含了多尺度特征融合模块MFF和一类特征提取模块OCE，使用多尺度融合大多数理由都差不多：低维度的特征包含了很多纹理、边缘等信息，高维度的特征包含了很多语义结构等信息。然后多尺度融合其实也进行了一个压缩特征的作用，去掉冗余的信息。

​		Reverse Distillation—教师网络作为Encoder，学生网络为Decoder，对齐每一个阶段两个网络的特征，通过cosine similarity来计算两个阶段特征之间的差异。

​		One-Class Bottleneck Embedding—E网络最后一个block的输出作为D的输入，但是这样仍然存在问题。E网络特征提取的能力很强，维度过高，会带来过多的冗余信息给D，而我的D只需要去处理基础的特征(例如PCA，我只需要处理主成分就行)。并且最后一个block的层很深，携带的都是高级的语义特征，那么D网络怎么拿着这些高级特征来重建一些低级的特征呢？有一个可能就是添加skip path，但是在推理异常样本时，无疑泄露了特征信息给到D。

​		第一个问题作者用one-class embedding block将高级特征映射到低维空间。将异常问题看做正常模式下存在扰动的话，紧凑的嵌入可以看做信息瓶颈，使得一些扰动不会流入D中。就是高维到低维。

​		第二个问题是MFF block在one-class嵌入前堆叠多尺度特征表示；OCBE由MFF和OCE组成。

​		推理的时候T模型能够反应异常，S模型反应不了，因为S模型只从one-class嵌入学习到了正常样本的表示。那么对于异常样本S和T的特征必定存在差异。 SAL 中的最大值定义为样本级异常分数。

# NF

## 2022-CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows

​		ImageNet预训练好的CNN对patch做encoder，decoder对encoder之后的特征进行一系列变换(flow)来逼近标准的正太分布，那么我的优化目标就转变为最大化正太分布，也就是通过最大化似然估计来实现。转变为正太分布之后，其实异常样本的概率密度的位置会很低。

# MB

## 2022-Towards Total Recall in Industrial Anomaly Detection

​		工业缺陷检测是一个冷启动问题，我们有很多正常的数据，但是手机缺陷数据确实一个比较困难的问题。如果只有OK数据来训练对于NG数据的话是存在分布偏移的，这种偏移你无法确认100%网络能识别成NG。

​		



包含3个模块：1.局部patch特征聚合为memory bank；2.coreset-reduction increase efficient；3.构成一个算法达到检测的目的。



Locally aware patch features

​		ImageNet预训练模型作为H，H_ij表示第i张图片的第j层特征。多级特征很好理解，只用最后一层的话丢弃了太多有用的信息。并且这种深层语义特征在ImageNet预训练，和工业检测存在较大的区别；

​		因此提出patch-level的特征memory-bank M，包含了中级特征的表达，这样避免在ImageNet上深层语义的偏置带来的影响；

​		patch的思想就必须考虑context信息，更多的pool和conv虽然增大了感受野，但是特征更加倾向ImageNet而与AD检测无关。提聚合邻域特征向量的方法，使用adaptive average pooling。这种patch特征放到一个大的空间中构成memory bank(一幅图像有很多patch集合，每一个patch构成一个feature)；

 Coreset-reduced patch-feature memory bank

​		使用minimax facility location从大的集合M中找到一个核心子集M_C(要求大的集合所有特征距离子集最近特征的最大距离最小，本质上在拉进两个集合之间的距离)；一种稀疏化的思想。



# Few-Shot

## 2022-Registration based Few-Shot Anomaly Detection

​		anomaly是一个模棱两可的定义，只要不是normal都可以成为异常。人通过对比正常样本来判断异常，为了模仿人这种行为，文章提出注册，各种不同的图片经过处理转换到同一个坐标系(为了便于比较)。 注册其实是类别无关的，那么可以跨类别泛化。

​		Proxy Task自监督中用的比较多，自监督学习也要有一个目的，而proxy task就是设计好一个子任务，让模型最终的目的是去做好这个子任务，子任务做好了也能学习到对应的特征。

​		Spatial Transformer Networks来给特征施加变换，使得模型学习到不同角度的特征。 并且在特征空间来对齐两个相同类别样本的特征。训练时是同个类别normal的样本对。在测试阶段，新的类别的normal数据拿到特征，计算均值方差获取他的高斯分布，对于新数据的abnormal特征映射到这个高斯分布，判断是否属于异常。



# 大模型

## CLIP-Learning Transferable B Visual Models From Natural Language Supervision

​		图片+文字多模态特征，输入网络的是图片和文本对；数据量要求很大；

​		没有分类头如何做预测呢？—prompt template，一个类别生成1个句子(A photo of a (object))，预训练的时候是句子，保持一致；

​		categorical label，类别无关，因为没有具体分类头；

​		prompt engineer + prompt ensemble：prompt起到一个提示作用，给模型做一个引导作用；



​		Contrastive Language-Image Pre-training(CLIP);

​		方法：自然语言处理的监督信号拿来做预训练，WIT WebImage Text数据；逐字逐句预测文本很难，训练效率也会很慢，因此使用对比学习来预测图片和文本是不是一对；

​		zero-shot transfer：自监督或无监督的方法主要研究特征学习的方法，让网络学习到更好的特征，但是用于下游任务时还是会存在很多问题，比如要下游任务的有标签数据，数据之间的分布偏移等；如何推理呢？图像编码器和文本编码器生成对应特征，然后计算文本特征和图像特征的相似性。



## Mamba

可以替代Transformer；

SSM state space model-状态空间模型，与时间无关linear time-invariant；

选择性SSM



引入非线性，delta t；选择性能力，选择和记住关键信息，引入时变；

有RNN的思想，那么只要控制我要观察多长时间的状态，那不就在调整注意力的窗口大小。



# Vision Prompt

## T-Rex: Counting by Visual Prompting



## Segment Any Anomaly without Training via Hybrid Prompt Regularization

多个工作的拼凑，整篇文章感觉像是prompt工程

Anomaly Region Generator：利用GroundingDINO提供一些Prompt，比如“Anomaly”；得到bbox-level region set与它对应的confidence score set；整个生成器就是要获取这可能存在缺陷的区域和他的置信度，和RPN差不多；

Anomaly Region Refiner：bbox作为prompt输入SAM微调mask区域；

## AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2

感觉与patchcore没区别，就是使用dinov2更强大的特征学习能力；