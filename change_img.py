# -*- coding: utf-8 -*-
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
path = '/资料/good.data'
datagen = ImageDataGenerator(
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

for file in os.listdir(path):
    # this is a PIL image, please replace to your own file path
    img = load_img(os.path.join(path,file))
    # this is a Numpy array with shape (3, 150, 150)
    x = img_to_array(img)
    # this is a Numpy array with shape (1, 3, 150, 150)
    x = x.reshape((1,) + x.shape)
    if not os.path.exists('/资料/pre_test/'+file.split('-')[0]):
        os.mkdir('/资料/pre_test/'+file.split('-')[0])
    print(file)
    i = 0
    for batch in datagen.flow(x,
                              batch_size=128,
                              save_to_dir='/资料/pre_test/'+file.split('-')[0],  # 生成后的图像保存路径
                              save_prefix=file.split('-')[0],
                              save_format='png'):
        i += 1
        if i > 500:  # 这个20指出要扩增多少个数据
            break  # otherwise the generator would loop indefinitely
