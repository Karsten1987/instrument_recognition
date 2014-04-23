import sys
sys.path.append("../svmlib")
sys.path.append("../svmlib/python")
#sys.path.append("../sandbox.svm/svmlib")
#sys.path.append("../sandbox.svm/svmlib/python")
from svmutil import *
from datetime import date
import scipy as sci
from scipy import *

class AudioSVM():
    def __init__(self, name, parameters=None, modellToLoad=None, verboseMode=False):
        if name == None:
            self.name = 'AudioSVMon'+str(date.toordinal(date.today()))
        else:
            self.name = name
        self.allFeatures = []
        self.allLabels = []
        
        self.means = []
        self.stds = []
        self.modell = modellToLoad
        if parameters != None:
            self.parameters = svm_parameter(parameters)
        else:
            self.parameters = None
            
        self.verbose = verboseMode
        
    def addFeature(self, label, feature):
        self.allLabels.append(label)
        #self.allFeatures.append(self.getSVMDictFromFeature(feature))
        self.allFeatures.append(feature)
        
    def getSVMDictFromFeature(self, feature):
        featureDict = {}
        for i in range(len(feature)):
            featureDict[i+1] = feature[i]
        return featureDict
        
    
    def printFeatures(self):
        for i in range(len(self.allFeatures)):
            print self.allLabels[i]
            print self.allFeatures[i]
            print '**************************'
        
    def normalizeFeatureMatrix(self):
        featureMatrix = sci.array(self.allFeatures)
        newMatrix = zeros(featureMatrix.shape)
        for i in range(featureMatrix[0,:].size):
            column = featureMatrix[:,i]
            #std = column.std()
            std = column.max() - column.min()
            self.stds.append(std)
            mean = column.mean()
            self.means.append(mean)
            for j in range(column.size):
                # KAROL IST DER MEISTER VON NUMPY ARRAYS
                newMatrix[j,i]= (column[j]-mean)/std
        featureMatrix = newMatrix
        featureList = featureMatrix.tolist()
        self.allFeatures = []
        for feature in featureList:
            self.allFeatures.append(self.getSVMDictFromFeature(feature))
        
    def trainModell(self):
        prob = svm_problem(self.allLabels, self.allFeatures)

        if self.modell != None:
            print 'no model to load'
            self.modell = svm_load_model(str(self.modell)+'.model')
        elif self.parameters != None:
            print 'training model with following parameters: '+str(self.parameters)
            self.modell = svm_train(prob, self.parameters)
        else:
            print 'train model with default parameters'
            self.modell = svm_train(prob)
            
        print 'cross-validation accuracy:'
        self.crossValid =  svm_train(prob, '-v 3')
        print self.crossValid
        modelName = (self.name)+'.model'
        svm_save_model(modelName,self.modell)
        return modelName
    
    def dumpDataToFile(self, filePath):
        f = open(filePath, 'r+')
        for i in range(len(self.allFeatures)):
            tmpString = str(self.allLabels[i])+' '
            for key in self.allFeatures[i].keys():
                tmpString = tmpString+str(key)+':'+str(self.allFeatures[i][key])+' '
            tmpString = tmpString+'\n'
            if (self.verbose):
                print 'string dumped:'
                print tmpString
            f.write(tmpString)
        f.close()
    
    def dumpTestDataToFile(self, filePath, label, feature):
        f = open(filePath, 'a')
        featureDict = self.getSVMDictFromFeature(feature)
        tmpString = str(label)+' '
        for key in featureDict.keys():
            tmpString = tmpString+str(key)+':'+str(featureDict[key])+' '
        tmpString = tmpString+'\n'
        if (self.verbose):
            print 'testdata string dumped:'
            print tmpString
        f.write(tmpString)
        f.close()
    
    def predict(self, label, feature):
        if self.modell == None:
            raise Exception('no modell trained yet. Please call trainModell first')
        else:
            tmpFeature = []
            for i in range(len(feature)):
                tmpFeature.append( (feature[i]-self.means[i])/self.stds[i])
            feature = tmpFeature
            featureDict = self.getSVMDictFromFeature(feature)
            if (self.verbose):
                print featureDict
            y = []
            y.append(label)
            x = []
            x.append(featureDict)
            p_label, p_acc, p_val = svm_predict(y, x, self.modell)
            return p_label, p_acc, p_val
            
if __name__ == '__main__':
    prob = (y, x) = svm_read_problem('../svmlib/foo_train')
    for i in range(len(y)):
        print 'dataset #' + str(i)
        print 'class label: ' + str(y[i])
        for key in x[i].keys():
            print 'feature #' + str(key) + ' ' + str(x[i][key])
            
    modell = svm_train(y, x)
    p_label, p_acc, p_val = svm_predict(y[4:], x[4:], modell)
    
