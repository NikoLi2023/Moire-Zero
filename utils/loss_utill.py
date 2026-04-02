import torch.nn.functional as F
import torch
import torchvision
from torchvision import models as tv



class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam #通常表示权重系数
        self.lam_p = lam_p #lam_p = lambda perceptual（感知损失的权重）
    def forward(self, pred, out1, out2, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.lam_p*self.loss_fn(pred, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(pred, gt1)
        loss2 = self.lam_p*self.loss_fn(out1, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt2)
        loss3 = self.lam_p*self.loss_fn(out2, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt3)
        return loss1+loss2+loss3
#Vgg16
# 索引|层类型|输入通道	|输出通道|卷积核	|说明
# Block 1
# 0	Conv2d	3	64	3x3	第1个卷积
# 1	ReLU	-	-	-	激活函数
# 2	Conv2d	64	64	3x3	第2个卷积
# 3	ReLU	-	-	-	激活函数
# 4	MaxPool2d	64	64	2x2	池化，尺寸减半
# Block 2
# 5	Conv2d	64	128	3x3	第3个卷积
# 6	ReLU	-	-	-	激活函数
# 7	Conv2d	128	128	3x3	第4个卷积
# 8	ReLU	-	-	-	激活函数
# 9	MaxPool2d	128	128	2x2	池化，尺寸减半
# Block 3
# 10	Conv2d	128	256	3x3	第5个卷积
# 11	ReLU	-	-	-	激活函数
# 12	Conv2d	256	256	3x3	第6个卷积
# 13	ReLU	-	-	-	激活函数
# 14	Conv2d	256	256	3x3	第7个卷积
# 15	ReLU	-	-	-	激活函数
# 16	MaxPool2d	256	256	2x2	池化，尺寸减半
# Block 4
# 17	Conv2d	256	512	3x3	第8个卷积
# 18	ReLU	-	-	-	激活函数
# 19	Conv2d	512	512	3x3	第9个卷积
# 20	ReLU	-	-	-	激活函数
# 21	Conv2d	512	512	3x3	第10个卷积
# 22	ReLU	-	-	-	激活函数
# 23	MaxPool2d	512	512	2x2	池化，尺寸减半
# Block 5
# 24	Conv2d	512	512	3x3	第11个卷积
# 25	ReLU	-	-	-	激活函数
# 26	Conv2d	512	512	3x3	第12个卷积
# 27	ReLU	-	-	-	激活函数
# 28	Conv2d	512	512	3x3	第13个卷积
# 29	ReLU	-	-	-	激活函数
# 30	MaxPool2d	512	512	2x2	池化，尺寸减半
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=tv.VGG16_Weights.DEFAULT).features[:4].eval()) ## [0,1,2,3] → 输出 112×112
        blocks.append(torchvision.models.vgg16(weights=tv.VGG16_Weights.DEFAULT).features[4:9].eval()) # [4,5,6,7,8] → 输出 56×56 ;包含第一个池化 + 第二个卷积组
        blocks.append(torchvision.models.vgg16(weights=tv.VGG16_Weights.DEFAULT).features[9:16].eval()) # [9,10,11,12,13,14,15] → 输出 28×28 ;包含第二个池化 + 第三个卷积组
        blocks.append(torchvision.models.vgg16(weights=tv.VGG16_Weights.DEFAULT).features[16:23].eval()) #[16,17,18,19,20,21,22] → 输出14x14 ;包含第三个池化 + 第四个卷积组 14×14
        for bl in blocks:
            for p in bl:
                p.requires_grad = False #对于卷积层，这会冻结 weight 和 bias  ；对于 ReLU、MaxPool2d 等没有参数的层，这行代码没有实际效果
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate #作用：保存插值函数，用于图像缩放
        # ImageNet
        # 数据集中，RGB
        # 三个通道的统计值不同，反映了自然图像的色彩分布：
        # R(红色)
        # 0.485
        # 0.229
        # 红色通道略亮
        # G(绿色)
        # 0.456
        # 0.224
        # 绿色通道稍暗
        # B(蓝色)
        # 0.406
        # 0.225
        # 蓝色通道最暗
        # 自然图像中，天空、水面等蓝色物体较少
        # 植被（绿色）和土壤 / 建筑（红色）更常见
        # 这反映了
        # ImageNet
        # 数据集的真实色彩分布
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) #作用：定义图像归一化的均值
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)) #作用：定义图像归一化的标准差
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss