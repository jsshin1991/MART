import numpy as np
import PIL.Image


def ConvertToGrayscale(attributions):
    return np.average(attributions, axis=2)


def Polarity(attributions, threshold, scaling):
    attributions = attributions / np.max(attributions)
    attributions = np.where(attributions < threshold, attributions, 1)
    return np.clip(scaling * attributions, 0, 1)


def Overlay(attributions, image):
    return np.clip(0.5 * image + 0.5 * attributions, 0, 255)


def pil_image(x):
    x = np.uint8(x)
    return PIL.Image.fromarray(x)


def Visualize(attributions,
              image,
              highlight_channel=[0, 255, 0],
              scaling=2,
              upper_threshold=0.9,
              lower_threshold=0.2):
    attributions = Polarity(attributions, threshold=upper_threshold, scaling=scaling)
    attributions_2 = ConvertToGrayscale(attributions)

    # Set threshold
    attributions_2 = np.where(attributions_2 > lower_threshold, attributions_2, 0.0)
    attributions_2 = np.expand_dims(attributions_2, 2) * highlight_channel

    attributions = Overlay(attributions_2, image)
    return attributions, attributions_2
