# -----------------------------------------------------------------
# Code retrieve from: https://stackoverflow.com/questions/68364284/
# -----------------------------------------------------------------

from nltk import agreement
from nltk.metrics.distance import masi_distance
from nltk.metrics.distance import jaccard_distance

import pandas_ods_reader as pdreader

annotfile = "test-iaa-so.ods"

df = pdreader.read_ods(annotfile, "Sheet1")

df.columns = df.columns.str.strip()

annots = []

def create_annot(an):
    """
    Create frozensets with the unique label
    or with both labels splitting on pipe.
    Unique label has to go in a list so that
    frozenset does not split it into characters.
    """
    text = str(an)
    if "|" in text:
        labels = text.split("|")
    else:
        # single label has to go in a list
        # need to cast or not depends on your data
        try:
            labels = [str(int(text))]
        except ValueError:
            labels = [text]
    return frozenset(labels)


for idx, row in df.iterrows():
    annot_id = row['annotItem'] + str(idx).zfill(3)
    annot_coder1 = ['coder1', annot_id, create_annot(row.coder1)]
    annot_coder2 = ['coder2', annot_id, create_annot(row.coder2)]
    annot_coder3 = ['coder3', annot_id, create_annot(row.coder3)]
    annots.append(annot_coder1)
    annots.append(annot_coder2)
    annots.append(annot_coder3)

# based on https://stackoverflow.com/questions/45741934/
jaccard_task = agreement.AnnotationTask(distance=jaccard_distance)
masi_task = agreement.AnnotationTask(distance=masi_distance)
tasks = [jaccard_task, masi_task]
for task in tasks:
    task.load_array(annots)
    print("Statistics for dataset using {}".format(task.distance))
    print("C: {}\nI: {}\nK: {}".format(task.C, task.I, task.K))
    print("Pi: {}".format(task.pi()))
    print("Kappa: {}".format(task.kappa()))
    print("Multi-Kappa: {}".format(task.multi_kappa()))
    print("Alpha: {}".format(task.alpha()))
