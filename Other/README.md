# 2016

## Learning Deep Features for Discriminative Localization

​		由于CNN中卷积的存在，哪怕是分类任务也是有定位物体的能力。使用全连接层进行分类会直接丢失这种能力。

​		分类时使用global average pooling来替代全连接层，不仅能够减少参数防止过拟合，还可以建立特征图与类别之间的关联。

​		CAM(Class Activation Mapping)