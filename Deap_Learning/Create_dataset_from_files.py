import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

imgdir_path = pathlib.Path('cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])



#fig = plt.figure(figsize=(10,5))
#for i, file in enumerate(file_list):
 #   img_raw = tf.io.read_file(file)
  #  img = tf.image.decode_image(img_raw)
   # print('Форма изображения: ', img.shape)
    #ax = fig.add_subplot(2,3,i+1)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.imshow(img)
    #ax.set_title(os.path.basename(file), size=15)
#plt.tight_layout()
#plt.show()

labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
print(labels)


ds_files_labels = tf.data.Dataset.from_tensor_slices((file_list, labels))

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0
    return image, label

img_height, img_width = 80, 120
ds_images_labels = ds_files_labels.map(load_and_preprocess)

fig = plt.figure(figsize=(10,6))
for i, example in enumerate(ds_images_labels):
    ax = fig.add_subplot(2,3, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()), size=15)

plt.tight_layout()
plt.show()



