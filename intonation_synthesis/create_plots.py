import h5py
import keras.backend as K
import numpy as np
from tsviz import VisualizationToolbox
from utils_tsviz import Modes


sess = K.get_session()
preds = h5py.File('/opt/ml/model/preds.hdf5')

vt = VisualizationToolbox()
vt.switchToTest()
for index in range(10):
    vt.setIterator(index)
    filename = vt.hts_validation_dataset.data[vt.getIterator()]
    X, y = vt.getExample()
    pred, saliency = sess.run(
        [vt.outputLayer, vt.layerGradientsWrtInput[-1]],
        feed_dict={vt.inputPlaceholder: np.expand_dims(X, axis=0)})

    preds.create_dataset("{}/preds".format(filename), data=pred)
    preds.create_dataset("{}/saliency".format(filename), data=saliency)

preds.close()

serviceOutput = vt.loadData(
    fastMode=Modes.FULL.value, visualizationFilters=True, verbose=True)