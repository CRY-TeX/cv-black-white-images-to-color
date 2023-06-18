import tensorflow as tf


def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index (SSIM) Loss: For tasks like image colorization, it might be beneficial to consider structural information in the loss function. The SSIM loss function compares local patterns of pixel intensities that have been normalized for brightness and contrast. By considering changes in structural information, contrast, and luminance, the SSIM loss can lead to better perceptual quality in the colorized images.
    """
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def dice_loss(y_true, y_pred):
    """
    Dice Loss: The Dice coefficient (also known as the Sørensen–Dice index) is a statistical metric used to compare the similarity of two samples. For image segmentation tasks, the Dice loss (1 - Dice coefficient) is often used. It works well when the classes are imbalanced (which is often the case in segmentation problems, where many pixels belong to the background class).
    """
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    dice_coef = numerator / (denominator + tf.keras.backend.epsilon())
    return 1 - dice_coef
