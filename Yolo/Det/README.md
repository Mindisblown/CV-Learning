# DETR-End2End Object Detection with Transformers

​		预测box看做一个集合预测的问题，但是现有的方法都没有直接去做这个集合预测，反而是做一个替代预测(分类+回归)；作者绕过这些代理任务，直接端到端；

​		CNN抽特征拉直送到Transformer encoder decoder来捕捉全局特征(每个特征与图片上其他的特征有交互)并生成box，box与gt做matching loss；

​		DETR最终输出是一个固定的集合，也就是说无论图片多大最终输出的维度一直都是固定的(文中是100)；那么如何将预测的100分配到哪一个GT上呢？二分图匹配，cost矩阵存放loss(预测的对不对+回归的好不好)

​		object queries：learnable embdeding



说白了patch的特征还是会存在歧义，多义性；我这个特征单独看是某种缺陷，如果结合看是不是不一样呢？

特征抽出一个子网络 专门预测在哪一块作为image patch prompt





匈牙利匹配算法：每一行每一列减去各自对应的最小值，保证每一行每一列都有一个0；进行迭代：第一步用最少的线覆盖矩阵中所有的0元素，第二步判断是否需要终止循环，第三步创造尽可能多的0；