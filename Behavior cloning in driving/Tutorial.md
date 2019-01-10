# 实验三：Behavior Clone in  Driving


## 1. Introduction
	以下是实验步骤
		1. 利用模拟器收集数据
		2. 用keras搭建一个神经网络，图片作为输入，steering angles是输出。
		3. 在模拟器中测试你的模型。

## 2. 模拟器
	打开 beta_simulator_linux/beta_simulator.x86_64 文件
![](figures/simulator.png)
![](figures/simulator1.png)

## 3. 读取数据
	1. lib/model.utils.py 下调用函数 read_csv_data，会生成训练集和validation集合，分别保存在 self.train_data和 self.valid_data下
	2. 在model.py的main函数，改下log_dir

## 4. keras
	不懂的可以参考目录 KerasTutorial，需要pip安装keras
或者参考[Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)
## 5. model.py
	1. 在main函数改log_dir。
	2. 在build_and_train_model（）下，搭建你的神经网络，并且训练。
	3. 建议的神经网络结构： https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
![](figures/train.png)
![](figures/cnn_model.png)

## 6.生成mp4文件
	0. 利用CarND-Behavioral-Cloning-P3目录下的 drive.py和video.py
	1. model.save( filepath )
	2. python drive.py model.h5 run1
	3. python video.py run1
	4.注意，运行以上可能会出现 no module name 'xxx', 用pip安装即可
[原github链接](https://github.com/udacity/CarND-Behavioral-Cloning-P3)

## 7.提交
	把mp4文件和代码压缩,命名“学号+姓名+实验三”，发送到yuyuli@mail.ustc.edu.cn
_______________
[参考论文](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
[原github链接](https://github.com/udacity/CarND-Behavioral-Cloning-P3)