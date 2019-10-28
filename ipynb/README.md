# 1.ipynb文件转为py文件
  
在xxx.ipynb所在目录下，打开终端，并输入命令:  
ipynb转换为python  
jupyter nbconvert --to python my_file.ipynb  
其中my_file.ipynb是要被转换的文件，转换后在该目录下出现xxx.py文件。  
ipynb转换为md  
jupyter nbconvert --to md my_file.ipynb   
其他格式转换请参考  
jupyter nbconvert --help  


# 2.jupyter notebookn内加载py文件(即转为ipynb文件)  
 
In [ ]:%run file.py  
 加载了file.py文件，相当于导包。  
In [ ]:%load file.py  
 把flie.py的代码显示出来。  
