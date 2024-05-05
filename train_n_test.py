import tensorflow as tf
from fashion_input import prepare_df, load_data_numpy
from simple_resnet import ResNet50
import os
import numpy as np
from hyper_parameters import get_arguments
# from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
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
    train_df = prepare_df(args.train_path, usecols=[
        'image_path', 'category', 'x1', 'y1', 'x2', 'y2'])
    vali_df = prepare_df(args.vali_path, usecols=[
        'image_path', 'category', 'x1', 'y1', 'x2', 'y2'])

    train_dataset = get_dataset(train_df, args.batch_size)
    val_dataset = get_dataset(vali_df, args.batch_size)

    model = ResNet50(input_shape=(64, 64, 3), classes=6)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # model.fit(train_dataset, epochs=args.epochs, validation_data=val_dataset)
    log_dir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # # Early Stopping callback based on validation loss
    # early_stopping_loss = EarlyStopping(
    #     monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='min')

    early_stopping_loss = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,         # Adjust minimum change threshold
        patience=7,              # Increase patience
        verbose=1, 
        mode='min'               # Monitor for loss improvement
    )

    early_stopping_acc = EarlyStopping(
        monitor='val_accuracy',  # Monitor validation accuracy
        min_delta=0.001,         # Adjust minimum change threshold
        patience=7,              # Increase patience
        verbose=1, 
        mode='max'               # Monitor for accuracy improvement
    )

    # # Early Stopping callback based on validation accuracy
    # early_stopping_acc = EarlyStopping(
    #     monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1, mode='max')

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001
    )

    model_checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True)

    callbacks_list = [
        tensorboard_callback,
        early_stopping_loss,
        early_stopping_acc,
        reduce_lr,
        model_checkpoint
    ]

    # # Include the desired Early Stopping in the callbacks list. Uncomment the one you prefer.
    # callbacks_list = [
    #     tensorboard_callback,
    #     # early_stopping_loss,  # Uncomment to enable early stopping based on loss
    #     early_stopping_acc    # Uncomment to enable early stopping based on accuracy
    # ]

    model.fit(train_dataset, epochs=args.epochs,
              validation_data=val_dataset, verbose=1, callbacks=callbacks_list)

    # # Save the model
    # if not os.path.exists(TRAIN_DIR):
    #     os.makedirs(TRAIN_DIR)
    # model.save(os.path.join(TRAIN_DIR, 'final_model.h5'))

    # Save the model
    TRAIN_DIR = 'logs_' + args.version + '/'
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    model.save(os.path.join(TRAIN_DIR, 'final_model.keras'))


if __name__ == "__main__":
    train()
