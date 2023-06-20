import os
import time
from argparse import ArgumentParser

import keras
import keras.layers as kl
import keras.models as km
import keras.optimizers
import keras.utils as ku
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color as skc
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imsave, imshow
from skimage.transform import resize
from tqdm import tqdm

# PATHS TO DATA
TRAIN_PATH = '/app/data/imagenet_data/train/'
TENSORBOARD_PATH = '/app/tensorboard/tensorboard_logs_throwaway'
# HIST_DATAFRAME_PATH = '/app/output/vgg19_hist_dataframe_full_2.csv'
MODEL_SAVE_PATH = '/app/models/decoder_throwaway.h5'
# CHECKPOINT_PATH = "/app/checkpoints/cp.ckpt"

# FUNCTIONS


def create_XY(data: list):
    """
    Use the images from the data generator to create the X and Y arrays, where X is the grayscale image and Y is the ab channels.
    We also need to reshape X to have a 4th dimension for the channel to be compatible with the VGG19 model.
    """
    X = []
    Y = []
    for img in data:
        try:
            lab = skc.rgb2lab(img)
            X.append(lab[:, :, 0])
            Y.append(lab[:, :, 1:] / 128)
        except Exception as ex:
            print(ex)

    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape + (1,))
    return X, Y


def out_vgg(X, vgg_model):
    """
    Use the VGG19 model to extract the features from the grayscale images.
    """
    vgg_features = []
    for i, sample in enumerate(X):
        sample = skc.gray2rgb(sample)  # VGG19 model takes 3 channels as input
        sample = sample.reshape((1, 224, 224, 3))
        prediction = vgg_model.predict(sample, verbose=0)
        prediction = prediction.reshape((14, 14, 512))
        vgg_features.append(prediction)
    vgg_features = np.array(vgg_features)
    return vgg_features


def create_decoder_model():
    """
    Create the decoder model that will take the features from the VGG19 model and output the ab channels.
    """
    decoder_model = km.Sequential()
    decoder_model.add(kl.Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(14, 14, 512)))
    decoder_model.add(kl.Conv2D(128, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(64, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(64, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(16, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.Conv2D(2, (3, 3), activation='tanh', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))

    adam = keras.optimizers.Adam(learning_rate=0.00001)

    decoder_model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    return decoder_model


def run_encoder_vgg(data, vgg_model):
    """
    Run the encoder model on the data to extract the features from the grayscale images.
    """
    X, Y = create_XY(data)
    vgg_features = out_vgg(X, vgg_model)
    return vgg_features, Y


def predict_grayscale_to_rgb(file_paths, encoder_model, decoder_model):
    """
    Predict the rgb images from the grayscale images.
    """
    rgb_images = []
    for file in tqdm(file_paths):
        test = ku.img_to_array(ku.load_img(file))
        test = resize(test, (224, 224), anti_aliasing=True)
        test *= 1.0 / 255
        lab = skc.rgb2lab(test)
        l = lab[:, :, 0]
        L = skc.gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))
        vggpred = encoder_model.predict(L, verbose=0)
        ab = decoder_model.predict(vggpred, verbose=0)
        ab = ab * 128

        cur = np.zeros((224, 224, 3))
        cur[:, :, 0] = l
        cur[:, :, 1:] = ab

        rgb_img = skc.lab2rgb(cur)
        rgb_img = (rgb_img * 256).astype(np.uint8)
        rgb_images.append(rgb_img)

    return rgb_images


def train(*, train_path: str, tensorboard_path: str, model_save_path: str, percentage: float, epochs: int, save_checkpoints: bool, tqdm_on: bool = False):
    encoder_path = '/app/models/encoder/vgg19_encoder.h5'
    if os.path.isfile(encoder_path):
        encoder_model = km.load_model(encoder_path)
    else:
        # download VGG19 model
        vgg_model = VGG19()

        # Base VGG19 convolutional layers until (14, 14, 512)
        encoder_model = km.Sequential(
            vgg_model.layers[:-5]
        )

    # freeze layers in the encoder model
    for layer in encoder_model.layers:
        layer.trainable = False

    # decoder model
    if os.path.isfile(model_save_path):
        decoder_model = km.load_model(model_save_path)
        print('Loaded decoder model from file.')
        adam = keras.optimizers.Adam(learning_rate=0.00001)
        decoder_model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    else:
        decoder_model = create_decoder_model()

    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=64 * 5,
        shuffle=True,
        class_mode=None,
    )

    tensorboard_callback = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True)

    model_checkpoint_dir = os.path.splitext(model_save_path)[0]
    model_name = os.path.splitext(os.path.basename(model_save_path))[0]
    if not os.path.isdir(model_checkpoint_dir) and save_checkpoints:
        os.mkdir(model_checkpoint_dir)
    
    base_path = os.path.relpath(os.path.dirname(model_save_path))
    csv_path = os.path.join(base_path, f'{model_name}_hist.csv')

    # train model
    batches = int((train.n // train.batch_size) * percentage)
    iterator = tqdm(range(batches)) if tqdm_on else range(batches)

    for i in iterator:
        vgg_features, Y = run_encoder_vgg(train[i], encoder_model)
        hist = decoder_model.fit(vgg_features, Y, validation_split=0.1, epochs=epochs, batch_size=32, verbose=0, callbacks=[tensorboard_callback])
        df_hist = pd.DataFrame(hist.history)
        if not os.path.isfile(csv_path):
            df_hist.to_csv(csv_path, index=False)
        else:
            df_hist.to_csv(csv_path, mode='a', header=False, index=False)


        if not tqdm_on:
            print(f'Cycle done: {i+1}/{batches}')

        if (i + 1) % 10 == 0 and save_checkpoints:
            decoder_model.save(os.path.join(model_checkpoint_dir, f'{model_name}_{i}.h5'))

    # save model
    decoder_model.save(model_save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', help='Path to the data folder', required=True)
    parser.add_argument('-t', '--tensorboard', help='Path to the tensorboard folder', required=True)
    parser.add_argument('-s', '--save', help='Path where the model file will be saved', required=True)
    parser.add_argument('-c', '--checkpoints', help='If set save checkpoints. Default: True', default=True, type=bool)
    parser.add_argument('-e', '--epochs', help='Number of epochs to train the model', default=40, type=int)
    parser.add_argument('-p', '--percentage', help='Percentage of the training data to use', default=1.0, type=float)
    args = parser.parse_args()

    TRAIN_PATH = args.data
    TENSORBOARD_PATH = args.tensorboard
    MODEL_SAVE_PATH = args.save

    args = parser.parse_args()
    start = time.perf_counter()
    train(
        train_path=TRAIN_PATH,
        tensorboard_path=TENSORBOARD_PATH,
        model_save_path=MODEL_SAVE_PATH,
        epochs=args.epochs,
        save_checkpoints=args.checkpoints,
        percentage=args.percentage
    )
    end = time.perf_counter()
    print(f'Took: {end-start:.2f} seconds')
