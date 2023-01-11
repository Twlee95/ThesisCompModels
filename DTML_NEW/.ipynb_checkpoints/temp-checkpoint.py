import cv2
import os
import glob
import shutil
import numpy as np

reference_root= r'C:\Users\USER\Desktop\210630 CMC_seg_태원'
remove_root= r"C:\Users\USER\Desktop\ori"
remove_image_path_list=[image_path.replace('_polyp','').replace('_outline','').replace('_inline','').replace('tif','jpg').replace(reference_root,remove_root)  for image_path in glob.glob(reference_root+'/*')]
remove_image_path_set=set(remove_image_path_list)

def imread_kor ( filePath, mode=cv2.IMREAD_UNCHANGED ) :
    stream = open( filePath.encode("utf-8") , "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , mode)
def imwrite_kor(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
                return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path
for candidate_path in glob.glob(remove_root+'/*'):
    if candidate_path in remove_image_path_set:
        pass
    else:
        os.remove(candidate_path)
        print(candidate_path)
for image_path in glob.glob(reference_root+'/*'):
    os.rename(image_path,image_path.replace('tif','jpg'))
for image_path in glob.glob(reference_root+'/*'):
    origin=image_path.replace(reference_root, remove_root).replace('_polyp','').replace('_outline','').replace('_inline','')
    shutil.copy(origin,image_path.replace('.jpg','_.jpg'))
for image_path in glob.glob(reference_root+'/*'):
    if not '_.jpg' in image_path:
        target_shape = imread_kor(image_path.replace('.jpg','_.jpg')).shape[:2]
        resized_image = cv2.resize(imread_kor(image_path),(target_shape[1],target_shape[0]))
        imwrite_kor(image_path,resized_image)
def divide_into_shape(root):
    for image_path in glob.glob(root+'/*'):
        image=imread_kor(image_path)
        image_root=root+'/'+str(image.shape[:2])
        create_directory(image_root)
        os.rename(image_path,image_path.replace(root,image_root))
divide_into_shape(reference_root)