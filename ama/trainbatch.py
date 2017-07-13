import os.path
import pandas as pd
import numpy as np
import preprocess

class TrainBatch():
    def __init__(self, path, label_path, img_size=(128,128), batch_size=128):
        self.path = path
        self.label_path = label_path
        self.batch_size = batch_size
        self.create(img_size)
    
    def next(self):
        return (self.image_gen.next(), self.label_gen.next())
        
    def create(self, img_size):
        img_batches = self.create_image_gen(img_size)
        filenames = [os.path.splitext(os.path.basename(fn))[0] for fn in self.image_gen.filenames]
        labels = self.get_labels(filenames)
        self.label_gen = self.create_label_gen(labels)
        self.labels = labels
        self.filenames = filenames
        
    def create_image_gen(self, img_size):
        batches = preprocess.get_batches(self.path, batch_size=self.batch_size,
            target_size=img_size)
        self.image_gen = batches
        self.nb_sample = batches.nb_sample
        
    def get_labels(self, filenames):
        tags = pd.read_csv(self.label_path)
        df = pd.DataFrame()
        df['image_name'] = filenames
        df = df.merge(tags, how='left', on='image_name')
        labels = np.stack([preprocess.onehot_labels(v) for v in df.tags], axis=0)
        return labels
        
    def create_label_gen(self, labels):
        batch_size = self.batch_size
        counter = 0
        while True:
			upper = min((counter+1)*batch_size, len(labels))
			yield labels[counter*batch_size:upper,:]
			if upper == len(labels):
				counter = 0
			else: counter+=1
