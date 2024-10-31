# 2024

## Adversarially Robust Few-shot Learning via Parameter Co-distillation of Similarity and Class Concept Learners

​		通过similarity学习的方式其实没有用到label的信息，类似使用特征进行聚类，而一般的监督学习使用类别，将特征映射到N个类别的标签空间中；

​		Similarity-concept Co-distillatiion： 有一个unified embedding模块参数是θu，另外两个学习器θs和θc，也就是前面的相似性和类别，通过EMA的方式θu会引入两个学习器的权重，为了方式两个学习器发散的太快，每隔m次就把u的参数分配到s和c(参数不会产生非常大的偏差，这也是防止了过拟合的产生）；

​		 Cross-branch Class-wise Global Adversarial Initialization Perturbations： unified框架的核心在于能够分辨两个学习器不同的知识，但是当adversaries过大时，co-distillation就丢失了robustness(理解为对抗太多网络学习差的太多，那么unified很难学习到有用且具有辨别性的知识，要么就是不断地过拟合)；

​		Branch Robustness Harmonization Module

## Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation for Cross-Domain Few-Shot Learning

​		Task-agnostic representation learning： ViT的基础模型，但是使用MIM的模式来训练；

​		Task-specific representation learning：给特征加一个缩放和平移的变换，y=ax+b，a是缩放b是平移；目的在于调整特征的均值和方差 ，其实也就是调整特征的分布，使用两个参数就能使得预训练的特征契合目标data的分布；

​		Varying the tuning depth： 许多方法直接微调整个网络，而文章是选择性的微调预训练的representation；

​		Discriminative sample-guided feature adaptation： 提出了可辨别的样本感知loss，每个类别分配一个Prototype，但是Prototype作为锚点，sample可以被吸引或排斥，那么就可以学习到适应任务的嵌入空间。

# 2020

## Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions

​		缩写为FEAT(Few-Shot Embedding Adaptation with Transformer)，针对特征提取阶段做的改进，之前的方法多是使用预训练提取特征然后计算相似性，但是这样嵌入的特征是与任务无关的，为了使得与任务相关，作者设计了自适应的Transformer，和AD中的SimpleNet提出的Feature Adaptation一个思路。

## Prototype Rectification for Few-Shot Learning

​		intra-class bias与cross-class bias，引入bias diminishing；预期的prototype应该是所有样本构成但是很难达到这个要求，那么少样本的情况下prototype是存在bias的，为了减少这种intra bias，使用伪标签的形式来对support set进行增强(考虑可能有分错的情况，使用X的加权和)；cross bias则是通过query set像support set的方向移动，其实求的是support set的平均样本特征与query set的平均样本特征间的偏差。

# 2019

## SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning

​		对嵌入特征做centering和L2 normalization，中心化就是encoder编码的基础上减去对应类的特征中心。

## Baseline-A CLOSER LOOK AT FEW-SHOT CLASSIFICATION

​		本质山是一个理论探讨性的实验，作者将数据集划分为base classes大量样本和novel classes少量样本，两者的label空间不相交，在base上就正常训练分类模型，在novel上进行fine-tuning，固定前面的backbone，然后更新分类head(使用cosine相似度作为度量)。迁移学习的思想，让网络学习更好的representation。

# 2018

## RelationNet-Learning to Compare: Relation Network for Few-Shot Learning

​		3个数据集合：training set、support set、testing set；通过训练集上的episode来模拟少样本学习，训练好了之后还可以根据需要再support set上进行fine-tuned；

​		网络由embedding module和relation module组成，假设训练集中挑选出了ABCDE五个类别的数据，再提供一个query sample，都经过embedding得到对应特征，query sample的特征和ABCDE的特征直接在depth维度上concat，得到5个concat的特征，在经过relation计算生成一个0-1之间的相似性得分。这是1shot的情况，如果有Kshot，那么需要将嵌入之后的特征进行element-wise融合得到对应类别的特征。

​		文中使用的MSE均方误差作为损失函数，看上去文章在做一个分类任务，标签只有0-1，但其实预测的是pair之间的关系分数。这样的好处是模型可以学习样本空间更加细微的关系，而不是直接预测一个hard label。有种label smooth的感觉。

# 2017

## ProtoNet-Prototypical Networks for Few-shot Learning

​		什么是Prototypical？可以理解为将图像映射到一个嵌入空间，对于每个类别在这个空间都是簇状的，这个簇的中心就可以称为原型。

​		Few-shot本质上在解决数据量少的时候引发的过拟合问题。对于数据中某个类别的样本很少，甚至只有一个，直接训练的话必定导致过拟合。

​		文章的主要思想就是通过CNN将图像映射到一个嵌入空间，那么属于一个类别的数据视为support set，所有support  set的mean vector构成了Prototype。预测的时候，将需要分类的样本也映射到嵌入空间中，然后计算到每个Prototype的距离，选取最小距离的原型作为这个类别。

​		Matching Networks使用有标签的样本来学习一个嵌入空间，对于没有标注的数据直接映射到这个嵌入空间中(有标签和无标签又可以称为support set和query set)，可以看做基于嵌入空间的最近邻分类器。在训练的时候，每次采样的mini-batch称为episodes，不断的进行子抽样来模拟few-shot任务。

​		作者的要解决一个核心的问题：过拟合。数据集非常有限，那么分类器只能有非常简单的归纳偏置。每个类别在嵌入空间中都是簇拥在1个prototype representation附近。利用CNN来学习一个非线性映射，讲输入映射到嵌入空间中，support set的平均称为class's  prototype，对于分类只需计算query point距离哪个原型网络最近。

​		1个原型是所有support set的平均值(映射到嵌入空间的特征向量)；如果1个类别有多个原型呢？是能够做到的，但是需要多个不同阶段再把这些support set聚成一个类别；



​		Bregman divergence是一种衡量两个概率分布之间差异的度量，通过规范化处理，可以找到两个域之间的公共子空间。 在这个公共子空间内，源域和目标域的相似性得以增强，有助于提取出两域之间的共享特征；query样本特征在support样本分布上的位置来衡量。5Way 1Shot采样数据包含5个类别，每个类别中有1个图像做support sample(整个way和shot都是针对support set来说)。



# Inductive与Transductive

​		Inductive学习：使用大量的手写数字图像来训练一个卷积神经网络模型，该模型能够学习到一般的手写数字特征，以便于在新的未知图像上进行分类预测。

​		Transductive：使用一小部分手写数字图像和它们对应的标签来训练模型。使用支持向量机等模型来对一组未知的手写数字图像进行分类预测，该模型只使用了已知的图像和标签，而没有从其他的手写数字图像中推断出一般性规律。

​		因此，Inductive学习更侧重于广泛的数据集训练，以便于在新的未知数据上具有很好的泛化能力；而Transductive学习更侧重于特定数据集的准确性，以便于对该特定数据集进行预测。

# Exponential Moving Average(EMA)

​		指数移动平均，平滑模型权重带来更好的泛化性能；本质上有点类似Momentum的思想，都是会考虑一个惯性，将前面的权重和当前权重通过a和(1-a)的比例还选取，并且越往前的权重考虑到的越小。