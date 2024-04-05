import tensorflow as tf
from fashion_input import prepare_df, load_data_numpy
from simple_resnet import ResNet50
import os
import numpy as np
from hyper_parameters import get_arguments
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

args = get_arguments()

TRAIN_DIR = 'logs_' + args.version + '/'
TRAIN_LOG_PATH = args.version + '_error.csv'

# Assuming you have a function to generate your dataset
def get_dataset(df, batch_size):
    """Assumes `load_data_numpy` returns suitable numpy arrays for x and y."""
    images, labels, _ = load_data_numpy(df)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset


def train():
    train_df = prepare_df(args.train_path, usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'])
    vali_df = prepare_df(args.vali_path, usecols=['image_path', 'category', 'x1', 'y1', 'x2', 'y2'])

    train_dataset = get_dataset(train_df, args.batch_size)
    val_dataset = get_dataset(vali_df, args.batch_size) 

    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset, verbose=1)


    # Save the model
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    model.save(os.path.join(TRAIN_DIR, 'final_model.h5'))

if __name__ == "__main__":
    train()
