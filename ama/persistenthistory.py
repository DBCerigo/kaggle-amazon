from keras.callbacks import Callback

class PersistentHistory(Callback):
    """Same as the default Keras History object, but doesn't
    delete data when new training begin."""

    def __init__(self):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

