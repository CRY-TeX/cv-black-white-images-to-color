import enum
import os
from typing import List

import keras
import keras.utils
import numpy as np
import pkg_resources
import skimage.color
import skimage.transform
import tensorflow as tf

try:
    ENCODER_MODEL_PATH = pkg_resources.resource_filename("colorizer", "models/encoder/vgg19_encoder.h5")
    DECODER_PATH = pkg_resources.resource_filename("colorizer", "models/")
except ModuleNotFoundError:
    ENCODER_MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models/encoder/vgg19_encoder.h5")
    DECODER_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")


class Decoder(enum.Enum):
    Decoder_vgg19_full = 'decoder_vgg19_full.h5'
    Decoder_vgg19_full_2 = 'decoder_vgg19_full_2.h5'
    Decoder_vgg19_full_3 = 'decoder_vgg19_full_3.h5'
    Decoder_vgg19_full_4 = 'decoder_vgg19_full_4.h5'
    Decoder_model_vgg19_5 = 'decoder_model_vgg19_5.h5'



class Colorizer:
    def __init__(self, decoder_model: Decoder = Decoder.Decoder_model_vgg19_5) -> None:
        self._encoder: keras.Model = keras.models.load_model(ENCODER_MODEL_PATH)  # type: ignore
        if self._encoder is None:
            raise ValueError(f'Could not load encoder model from "{ENCODER_MODEL_PATH}"')

        self._decoder: keras.Model = keras.models.load_model(os.path.join(DECODER_PATH, decoder_model.value))  # type: ignore
        if self._decoder is None:
            raise ValueError(f'Could not load decoder model "{decoder_model.value}" from "{DECODER_PATH}"')

    def colorize(self, file_path: str):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File "{file_path}" could not be found')

        # load source image from file path
        image = keras.utils.img_to_array(keras.utils.load_img(file_path))
        image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
        image *= 1.0 / 255

        # convert to LAB color space
        lab = skimage.color.rgb2lab(image)
        l = lab[:, :, 0]

        L = skimage.color.gray2rgb(l)
        L = L.reshape((1, 224, 224, 3))

        # make prediction
        vggpred = self._encoder.predict(L, verbose=0)
        ab = self._decoder.predict(vggpred, verbose=0)
        ab = ab * 128

        cur = np.zeros((224, 224, 3))
        cur[:, :, 0] = l
        cur[:, :, 1:] = ab

        # convert back to rgb image
        rgb_img = skimage.color.lab2rgb(cur)
        rgb_img = (rgb_img * 256).astype(np.uint8)
        return rgb_img

    def _run_vgg_encoder(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Use the VGG19 model to extract the features from the grayscale images.
        """
        vgg_features = []
        for sample in X:
            sample = skimage.color.gray2rgb(sample)  # VGG19 model takes 3 channels as input
            sample = sample.reshape((1, 224, 224, 3))
            prediction = self._encoder.predict(sample, verbose=0)
            prediction = prediction.reshape((14, 14, 512))
            vgg_features.append(prediction)
        vgg_features = np.array(vgg_features)
        return vgg_features

    def evaluate_decoder(self, file_paths: List[str]):
        if any(not os.path.isfile(file_path) for file_path in file_paths):
            raise FileNotFoundError(f'One or more files could not be found')

        # load source image from file path
        X = []
        Y = []
        for file in file_paths:
            image = keras.utils.img_to_array(keras.utils.load_img(file))
            image = skimage.transform.resize(image, (224, 224), anti_aliasing=True)
            image *= 1.0 / 255

            lab = skimage.color.rgb2lab(image)
            l = lab[:, :, 0]
            ab = lab[:, :, 1:] / 128

            X.append(l)
            Y.append(ab)

        X = np.array(X)
        X = X.reshape(X.shape + (1,))
        Y = np.array(Y)

        # run encoder model
        vgg_features = self._run_vgg_encoder(X, Y)

        # evaluate decoder model
        loss = self._decoder.evaluate(vgg_features, Y, verbose=0)
        return loss

    @property
    def encoder(self) -> keras.Model:
        return self._encoder

    @property
    def decoder(self) -> keras.Model:
        return self._decoder

    def set_decoder_from_path(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'File "{path}" could not be found')

        decoder = keras.models.load_model(path)
        if decoder is None:
            raise ValueError(f'Could not load decoder model from "{path}"')

        self._decoder = decoder

    def set_encoder_from_path(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'File "{path}" could not be found')

        encoder = keras.models.load_model(path)
        if encoder is None:
            raise ValueError(f'Could not load encoder model from "{path}"')

        self._encoder = encoder
