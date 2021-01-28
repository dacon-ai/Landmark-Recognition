import tensorflow as tf
import pandas as pd

def preprocess_input(image, target_size, augment=False):
    
    image = tf.image.resize(
        image, target_size, method='bilinear')
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def create_dataset(df, training, batch_size, input_size):

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)

    image_paths, labels, probs = df.path, df.label, df.prob

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, probs))
    dataset = dataset.map(
        lambda x, y, p: (read_image(x), y, p),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        lambda x, y, p: (preprocess_input(x, input_size[:2], training), y),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset