import os


folder=r'/home/yingying/project/conditional_diffusion/dataset/test/test_dataset/mask/multi16'
file_list=os.listdir(folder)
for i in range(len(file_list)):
    mask=file_list[i]
    new_name='mask_'+str(i+1)+'.tif'
    path1=os.path.join(folder,mask)
    path2=os.path.join(folder,new_name)
    os.rename(path1,path2)
