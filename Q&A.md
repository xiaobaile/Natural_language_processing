# 问题及解决办法 
- Q:在pycharm中导入自己写的模块时，得不到智能提示，并在模块名下出现下红线，但是代码可以执行。
- A:因为文件目录设置的问题，pycharm中的最上层文件夹是项目文件夹，在项目中导包默认是从这个目录下寻找，当在其中再次建立目录，目录内的py文件如果要导入当前目录内的其他文件，单纯的使用import导入，是得不到智能提示的，这是pycharm设置的问题，并非导入错误。
- M: `rom .auto_encoder import AutoEncoder`


- Q:通过pip install 安装第三方包时经常会出现超时的现象
- A:换个网络连接重新下载,
- M:`pip --default-timeout=100 install jieba -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com`