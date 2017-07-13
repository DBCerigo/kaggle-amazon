import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def onehot_labels(strings):
    labels = ["agriculture",
        "artisinal_mine",
        "bare_ground",
        "blooming",
        "blow_down",
        "clear",
        "cloudy",
        "conventional_mine",
        "cultivation",
        "habitation",
        "haze",
        "partly_cloudy",
        "primary",
        "road",
        "selective_logging",
        "slash_burn",
        "water"
    ]
    strings = strings.split(' ')
    vector = np.zeros(len(labels))
    for s in strings:
        try:
            idx = labels.index(s)
            vector += to_categorical([idx], len(labels)).squeeze()
        except ValueError:
            raise Exception('Unrecognised label '+s)
    return vector

def get_batches(dirname, gen=ImageDataGenerator(), shuffle=False,
				batch_size=128, class_mode=None, target_size=(256,256)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
