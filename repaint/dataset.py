import os

folder2 = r"/home/yingying/project/conditional_diffusion/dataset/train/ASTER"
folder1 = r"/home/yingying/project/conditional_diffusion/dataset/train/tf_mask"

# 获取文件夹1中的文件名
files1 = os.listdir(folder1)
for file in files1:
    names=file.split("_")
    new_name=names[0]+"_"+names[1]+"_"+names[2].split(".")[0]+"_tfmask.tif"
    os.rename(os.path.join(folder1,file),os.path.join(folder1,new_name))

