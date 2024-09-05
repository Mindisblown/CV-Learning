# 2017

## Protonet

Prototypical Networks for Few-shot Learning

​		什么是Prototypical？可以理解为将图像映射到一个嵌入空间，对于每个类别在这个空间都是簇状的，这个簇的中心就可以称为原型。

​		Few-shot本质上在解决数据量少的时候引发的过拟合问题。对于数据中某个类别的样本很少，甚至只有一个，直接训练的话必定导致过拟合。

​		文章的主要思想就是通过CNN将图像映射到一个嵌入空间，那么属于一个类别的数据视为support set，所有support  set的mean vector构成了Prototype。预测的时候，将需要分类的样本也映射到嵌入空间中，然后计算到每个Prototype的距离，选取最小距离的原型作为这个类别。