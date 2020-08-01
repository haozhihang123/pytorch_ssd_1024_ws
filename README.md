库版本信息： `PyTorch 1.4` in `Python 3.6`.

代码说明：

代码源自对pytorco_ssd的修改，pytorch_ssd是对SSD的最原始的复现，输入图像为300*300，由于要检测ws，图像为遥感图像，尺寸比较大，故增大输入样本的尺寸到1024*1024

300*300 map：0.859

1024*1024map：0.779

问题及解决：

２．版本问题：model.py中的520行及附近中的suppress变量会由于版本不同导致报错，只需要修改一下类型即可（在dtype=torch.bool和dtype=torch.uint8之间转换，下面的~suppress变换为1-suppress或反变换）

本程序仅仅是本人学习SSD并应用于自身项目中的记录。
