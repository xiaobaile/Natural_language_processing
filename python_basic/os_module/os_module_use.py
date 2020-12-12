import os

# 
""" 获取当前文件所在的路径:
"/Users/shiluyou/Desktop/Natural_language_processing/python_basic/os_module/os_module_use.py"
"""
res = os.path.curdir
res = os.path.abspath(res)
print(res)