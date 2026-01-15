# **Real-time Stereo Visual Odometry with Local Non-Linear Least Square Optimisation**

## **About**

在源代码https://github.com/sakshamjindal/Stereo-Visual-SLAM-Odometry.git基础上修改的，用以支持我们自己的双目数据和应用场景。在源代码的基础上修改了前后帧的tracking方式，将原来的光流tracking改为了关键点匹配；增加了aruco二维码的四个角点作关键点的特征提取以及匹配方式。原始代码库中没有位姿图优化，当然也就没有保存共视关系，只是依次估计相邻帧的位姿并累积。原始代码的结构框架较好，因此用来二次开发。适合用于小范围大步长的位姿估计。

相机每次移动距离较长,通过光流进行tracking效果不太好，将tracking改为了matching，matching就需要先检测当前帧的关键点，因此在_process_continuous_frame()中要把_update_stereo_state()函数的位置提前，_update_stereo_state()代码位置提前后导致_process_second_frame()函数和_process_continuous_frame()就相同了，应此将这两个函数合并成一个了，用_process_continuous_frame()来表示。

将params_cec.yaml中的my_draw_matches设置为True就可以显示关键点检测和匹配的图像。有两次匹配结果，一次是双目图像的匹配；另一次是相邻两帧的左相机图像的匹配。每次显示结果会暂停程序，按下esc键后会关闭图像窗口并继续执行程序。

## **Installation**

```bash
$ conda env create -f setup/environment.yml
$ pip install -e .
```

## **Usage**

For simulation of visual odometry, run the followig command

```bash
$ python main.py --config_path configs/params.yaml
```

The `params.yaml` needs to be edited to configure the sequence to run the simulation.
