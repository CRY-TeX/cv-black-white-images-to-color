import os
import time

import keras
import keras.layers as kl
import keras.models as km
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
TENSORBOARD_PATH = '/app/tensorboard/vgg19_tensorboard_logs_full'
HIST_DATAFRAME_PATH = '/app/output/vgg19_hist_dataframe_full.csv'
MODEL_SAVE_PATH = '/app/models/decoder_vgg19_full.h5'

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
        sample = skc.gray2rgb(sample) # VGG19 model takes 3 channels as input
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
    decoder_model.add(kl.UpSampling2D((2, 2)))
    decoder_model.add(kl.Conv2D(32, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))
    decoder_model.add(kl.Conv2D(16, (3, 3), activation='relu', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))
    decoder_model.add(kl.Conv2D(2, (3, 3), activation='tanh', padding='same'))
    decoder_model.add(kl.UpSampling2D((2, 2)))

    decoder_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return decoder_model

def run_encoder_vgg(data, vgg_model):
    """
    Run the encoder model on the data to extract the features from the grayscale images.
    """
    X, Y = create_XY(data)
    vgg_features = out_vgg(X, vgg_model)
    return vgg_features, Y

def predict_grayscal2rgb(file_paths, encoder_model, decoder_model):
    """
    Predict the rgb images from the grayscale images.
    """
    rgb_images = []
    for file in tqdm(file_paths):
        test = ku.img_to_array(ku.load_img(os.path.join(TEST_PATH, file)))
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
        rgb_img = ( rgb_img * 256 ).astype(np.uint8)
        rgb_images.append(rgb_img)

    return rgb_images


def main():
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
    decoder_model = create_decoder_model()

    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255)
    train = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(224, 224),
        batch_size=128,
        class_mode=None
    )

    tensorboard_callback = TensorBoard(log_dir=TENSORBOARD_PATH, histogram_freq=0, write_graph=True, write_images=True)

    # train model
    hists = []
    batches = train.n // train.batch_size
    for i in tqdm(range(batches)):
        vgg_features, Y = run_encoder_vgg(train[i], encoder_model)
        hist = decoder_model.fit(vgg_features, Y, validation_split=0.1, epochs=100, batch_size=32, verbose=0, callbacks=[tensorboard_callback])
        hists.append(hist)

    # save model
    decoder_model.save(MODEL_SAVE_PATH)

    # save histories to csv
    concatted_histories = []
    for hist in hists:
        concatted_histories.append(pd.DataFrame(hist.history))

    df_hist = pd.concat(concatted_histories)
    df_hist.to_csv(HIST_DATAFRAME_PATH)

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f'Took: {end-start:.2f} seconds')
