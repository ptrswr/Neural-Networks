
import matplotlib.pyplot as plt

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def draw_plots(training_hist):
    """
    :param training_hist: training history of best model
    :return:
    """
    training_acc = training_hist.history['accuracy']
    val_acc = training_hist.history['val_accuracy']

    training_loss = training_hist.history['loss']
    val_loss = training_hist.history['val_loss']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))

    axes[0].plot(training_acc, 'b')
    axes[0].plot(val_acc, 'g')
    axes[0].legend(['Training accuracy', 'Validation accuracy'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')

    axes[1].plot(training_loss, 'r')
    axes[1].plot(val_loss, 'c')
    axes[1].legend(['Training loss', 'Validation loss'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    plt.show()