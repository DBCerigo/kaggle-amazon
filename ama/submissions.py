import numpy as np
import csv

def write_val_csv(preds, labels, filenames, labelorder, fp):
    with open(fp, 'w') as csvf:
        writer = csv.writer(csvf)
        pred_headers = [l+'_pred' for l in labelorder]
        true_headers = [l+'_true' for l in labelorder]
        writer.writerow(['image_name']+pred_headers+true_headers)
        for fn, pred, true in zip(filenames, preds, labels):
            writer.writerow([fn]+list(pred)+list(true))
            
def onehot_to_string(vector, thresholds, labelorder):
    s = ''
    for v, th, name in zip(vector, thresholds, labelorder):
        if v>th:
            s += ' '+name
    return s[1:]

def write_presubmission(preds, filenames, labelorder, fp):
    """Writes a file with a header for each label and the sigmoid return 
    that the model game. Not actually a submission file, as we need to 
    pick appropriate thresholds."""
    with open(fp, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['image_name']+labelorder)
        for fn, pred in zip(filenames, preds):
            writer.writerow([fn]+list(pred))
                                  
def write_submission(preds, filenames, labelorder, thresholds, fp):
    """Writes the actual submission file given an array of threholds,
    each corresponding to each class in labelorder"""
    with open(fp, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['image_name', 'tags'])
        for fn, pred in zip(filenames, preds):
            writer.writerow([fn, onehot_to_string(pred,thresholds,labelorder)])                            

def read_presubmission_for_f2(fp):
    with open(fp,'r') as csvf:
        reader = csv.reader(csvf)
        headers = reader.next()
        filenames = []
        guesses = []
        for row in reader:
            filenames.append(row[0])
            guesses.append(np.array(row[1:], dtype=float))
    return filenames, np.stack(guesses)

def read_presubmission_for_ensemble(fp):
    with open(fp,'r') as csvf:
        reader = csv.reader(csvf)
        headers = reader.next()
        filenames = []
        mapping = {}
        for row in reader:
            fn = row[0]
            filenames.append(fn)
            mapping[fn] = {}
            for i,v in enumerate(row[1:]):
                idx = i+1
                mapping[fn][headers[idx]] = v

        return filenames, mapping

def combine_presub_mappings(filenames, mappings):
    combined = {}
    for fn in filenames:
        combined[fn] = {}
        for h in mappings[0][fn].keys():
            combined[fn][h]=sum([float(m[fn][h])/len(mappings) for m in mappings])
    return combined

def write_presub_mapping_to_file(mapping, fn):
    headers = ['image_name',
        'agriculture',
        'artisinal_mine',
        'bare_ground',
        'blooming',
        'blow_down',
        'clear',
        'cloudy',
        'conventional_mine',
        'cultivation',
        'habitation',
        'haze',
        'partly_cloudy',
        'primary',
        'road',
        'selective_logging',
        'slash_burn',
        'water'
    ]
    with open(fn, 'w') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=headers)
        writer.writeheader()
        for fn,row in mapping.iteritems():
            row['image_name']=fn
            writer.writerow(row)
