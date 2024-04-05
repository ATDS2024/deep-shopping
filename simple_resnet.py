import tensorflow as tf
from hyper_parameters import get_arguments

args = get_arguments()

def ResNetBlock(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A standard ResNet block."""
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = tf.keras.layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
    return x

def ResNet50(input_shape=(64, 64, 3), classes=6):
    """Builds ResNet50 with TensorFlow 2.x."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', name='conv1_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(axis=3, name='conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1_pool')(x)

    x = ResNetBlock(x, 64, name='conv2_block1')
    x = ResNetBlock(x, 64, conv_shortcut=False, name='conv2_block2')
    x = ResNetBlock(x, 64, conv_shortcut=False, name='conv2_block3')

    x = ResNetBlock(x, 128, stride=2, name='conv3_block1')
    # Add more blocks as needed

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = tf.keras.Model(inputs, x, name='resnet50')
    return model
