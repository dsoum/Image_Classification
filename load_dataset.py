# images will be size of image_size with batch, and labels will be one_hot_encoded
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

def load_dataset(image_size = (180,180),batch_size = 32,validation_split=0.2):
    # to download the dataset
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,fname = "flower_photos",untar = True)
    data_dir = pathlib.Path(data_dir)
    #print(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir,image_size=image_size,label_mode='categorical',validation_split=validation_split,seed = 23,batch_size=batch_size,subset='training')
    valid_ds = tf.keras.utils.image_dataset_from_directory(data_dir,image_size=image_size,label_mode='categorical',validation_split=validation_split,seed = 23,batch_size=batch_size,subset='validation')

    return train_ds,valid_ds

if __name__=='__main__':
    train_ds,valid_ds= load_dataset()
    for image,label in train_ds.take(1):
        print(train_ds.class_names)
        plt.imshow(image[1].numpy().astype('uint8'))
        plt.title(str(label[1]))
        plt.show()
        break


