import os
from PIL import Image

texture_files = os.listdir("./")
texture_files = [f for f in texture_files if f.lower().endswith('jpeg') or f.lower().endswith('jpg') or f.lower().endswith('png') or f.lower().endswith('webp')]

for tf in texture_files:
    file_name, im_type = tf.split('.')
    if im_type=='png':
        continue
    tf_png = tf.replace(tf, file_name + '.png')
    im = Image.open(tf).convert('RGB')
    im.save(tf_png, 'png')
