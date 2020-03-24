from tsviz import VisualizationToolbox
from utils_tsviz import Modes


vt = VisualizationToolbox()
vt.switchToTest()
vt.getExample()
serviceOutput = vt.loadData(
    fastMode=Modes.FULL.value, visualizationFilters=True, verbose=True)
data = serviceOutput['data']
data_keys = [key for key in data.keys()]

inputFeatureNames = vt.inputFeatureNames
