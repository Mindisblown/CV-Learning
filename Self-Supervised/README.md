# inductive bias

cnn携带了locality的先验，认为图片相邻之间特征相关性就越强，平移等边形(无论先做平移还是先做卷积，结果是一样的)，位置不敏感。

# 利用负样本

## InstDIsc-Unsupervised Feature Learning via Non-Parametric Instance Discrimination

判别式代理任务

个体判别任务；每一张图片看做一个类别。一个CNN将所有图像编码成特征，在特征空间中，这些特征能够区分开。从memory bank中抽负样本，进行对比学习。

## InvaSpread-Unsupervised Embedding Learning via Invariant and Spreading Instance Feature(SimCLR前身)

判别式代理任务

相似物体特征应保持不变形，不相似的物体应该尽可能分开。假设batch是256，经过数据增强生成增强批同样是256，那么进行对比学习时它的负样本有(256-1)x2。共享同一个CNN。

## CPC-Representation Learning with Contrastive Predictive Coding

预测代理任务

自回归模型LSTM，前一刻t的输出预测下一个t+1时刻的输出

## CMC-Contrastive Multiview Coding

多视角下的互信息，多视角的数据集NYU RGBD；不同的输入来自不同的模态，但是对应的事同一个物体，他们应该是正样本(多个视角)，其余的图片则为负样本(不配对的视角)。多视角下的对比学习，但是不同视角需要的编码器结构不一样，计算资源比较大。

## MoCo-Momentum Contrast for Unsupervised Visual Representation Learning

队列解决字典问题，动量解决字典特征不一致的问题。InstDisc的改进。

## SimCLR-A Simple Framework for Contrastive Learning of Visual Representation

正负样本和InvaSpread一样，CNN之后添加了一个projection(MLP)，下游任务时这个是丢弃掉；

## MoCoV2-Improved Baselines with Momentum Contrastive Learning

simlcr的组件引入到MoCoV1

## SimCLRV2-Big Self-Supervised Models are Strong Semi-Supervised Learners

simclrv1自监督得到一个模型，在小部分有标签数据上进行微调，利用微调后的模型在大量无标签的数据生成伪标签，利用伪标签再进行自监督训练。

## SwAV-Unsupervised Learning of Visual Features by Contrastive Cluster Assignments

聚类算法引入无监督，3000个cluster center，特征Z与prototype C；Multi-crop：增加crop(也就是view)，2个160+4个96。

# 不用负样本的对比学习

## BYOL-Bootstrap Your Own Latent A New Approach to Self-Supervised Learning

Latent、Embedding、Hidden都是特征的意思；model collapse或learning collapse—不给负样本去约束，模型只会预测0，那么网络就相当于懒得更新，每次都预测那个值就可以；在特征z输出之后添加了一个predictor(MLP)，mlp预测尽可能去另外一个特征尽可能相似。

BatchNorm存在信息泄露，会释放一些隐式的负样本特征。—第二篇文章对这个做出了回应。

## BYOL Works even without batch statistics

 BN提升模型训练的稳健性，使用GN与weight standard。

## SimSiam-Exploring Simple Siamese Representation Learning

​		不需要负样本，不需要大的batchsize，不需要动量编码器；stop gradient



# Transformer

## MoCoV3-An Empirical Study of Training Self-Supervised Vision Transformers

freeze patch projection；

## DINO-Emerging Properties in Self-Supervised Vision Transformers

Self-distillation with no labels(DINO)

student的网络去预测teacher输出的结果

## MAE-Masked Autoencoders Are Scalable Vision Learners