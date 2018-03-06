import os
import tensorflow as tf
from PIL import Image

cwd = '/资料/pre_test/'

'''
此处我加载的数据目录如下：
0 -- img1.jpg
     img2.jpg
     img3.jpg
     ...
1 -- img1.jpg
     img2.jpg
     ...
2 -- ...
 这里的0， 1， 2...就是类别，也就是下文中的classes
 classes是我根据自己数据类型定义的一个列表，大家可以根据自己的数据情况灵活运用
...
'''
def dict_gen():
    classes = os.listdir(cwd)
    for index, name in enumerate(classes):
        write_dict(str(index)+':'+name, 'dict.txt')

def write_dict(txt, result_file_name):
    with open(result_file_name, 'a') as f:
        # s = str(result_list).encode("utf-8")
        f.writelines(txt+'\n')
        f.close()

classes = os.listdir(cwd)
writer = tf.python_io.TFRecordWriter("train.tfrecords")
count = 0
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        count += 1
        img_path = class_path + img_name
        print(img_path)
        img = Image.open(img_path)
        img = img.resize((32, 32))
        # img = img.convert("L")
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()
write_dict('num:'+str(count),'dict.txt')

dict_gen()