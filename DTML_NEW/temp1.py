import glob
import os
from posixpath import dirname
from PIL import Image
from tqdm import tqdm


import os
print(os.getcwd())

base_dir = r'D:/cmc_/'
parent_dir = dirname(dirname(base_dir))


def read_img(path):
    return Image.open(path)

def move2dir(img, file_name):
    new_path = parent_dir+'/result/'+str(img.size)
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    print(new_path+'/'+file_name)
    img.save(new_path+'/'+file_name)

def devide():
    file_list = os.listdir(base_dir)
    for file_name in tqdm(file_list, desc='Moving..'):
        img = read_img(base_dir+file_name)
        move2dir(img, file_name)

devide()