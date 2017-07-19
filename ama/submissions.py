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
