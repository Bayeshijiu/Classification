# 1.ipynb文件转为py文件
方法一：  
在xxx.ipynb所在目录下，打开终端，并输入命令:  
ipynb转换为python  
jupyter nbconvert --to python my_file.ipynb  
ipynb转换为md  
jupyter nbconvert --to md my_file.ipynb  
ipynb转为html  
jupyter nbconvert --to html my_file.ipynb  
ipython转换为pdf  
jupyter nbconvert --to pdf my_file.ipynb  
其他格式转换请参考  
jupyter nbconvert --help  
其中my_file.ipynb是要被转换的文件，转换后在该目录下出现xxx.py文件。  

方法二：  
jupyter notebook 界面  --download as--python file  

# 2.jupyter notebookn内加载py文件(即转为ipynb文件)  
 
In [ ]:%run file.py  
 加载了file.py文件，相当于导包。  
In [ ]:%load file.py  
 把flie.py的代码显示出来。  
