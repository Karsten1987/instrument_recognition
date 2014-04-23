from audiolib import SoundPlot
from svmFoo import AudioSVM
from threeDplots import Plot3d
import os
from os import path as pathHelper
from sklearn.decomposition import PCA

class Instrumentrecognition(object):
    def __init__(self, svmModelName, svmParameters, verboseMode):
        self.labelDict = {'guitar':3, 'cello':2, 'klavier':1}  
        self.pathDict = {'cello':'../../sandbox/sandbox.samples.audio/cello_mono/', 'guitar':'../../sandbox/sandbox.samples.audio/guitar_new_mono/', 'klavier':'../../sandbox/sandbox.samples.audio/klavier_mono/'}
        self.testPathDict = {'cello':'../../sandbox/sandbox.samples.audio/cello_test/', 'guitar':'../../sandbox/sandbox.samples.audio/guitar_test_new/', 'klavier':'../../sandbox/sandbox.samples.audio/klavier_test/'}
        self.Tplot = Plot3d()
        self.Tplot2 = Plot3d()
        self.Tplot3 = Plot3d()
        self.verbose = verboseMode
        if (self.verbose):
            print 'initialized 3D plot'
        
        self.statistic = []
        for key in self.labelDict.keys():
            tmpList = []
            tmpList.append(self.labelDict[key]) #label
            tmpList.append(0) # success
            tmpList.append(0) # errors
            self.statistic.append(tmpList)
        
        #params = '-s 2 -t 2  -h 0 -b 0 -v 3'
        self.SVM = AudioSVM(svmModelName, svmParameters, None, verboseMode)
        if (self.verbose):
            print 'initialized SVM'
       
    def loadAllFiles(self, fileName=None):
        fileReadable = False
        if fileName != None:
            if pathHelper.isfile(fileName):
                try:
                    self.loadFeaturesFromFile(fileName)
                    fileReadable = True
                except IOError as e:
                    print 'file not readable, continue with reading the model new from samples'
        if fileReadable == False:
            self.loadCelloFiles()
            self.loadGuitarFiles()
            self.loadPianoFiles()
            self.SVM.normalizeFeatureMatrix()
        
    def loadFeaturesFromFile(self, fileName):
        file = open(fileName, 'r')
        for line in file:
            tmpString = line.split(' ') #+1 1:0.3003 2:0.39393
            label = float(tmpString[0])
            feature = []
            print type(tmpString)
            for i in xrange(1, len(tmpString) - 1):
                featureSplit = tmpString[i].split(':')
                print featureSplit
                feature.append(float(featureSplit[1]))
            self.SVM.addFeature(label, feature)
	    if label==self.labelDict['guitar']:	
			self.Tplot.addFeature(feature)
	    if label==self.labelDict['cello']:
			self.Tplot2.addFeature(feature)
	    if label==self.labelDict['klavier']:
			self.Tplot3.addFeature(feature)
	self.PCAMatrix=PCA(n_components=2).fit(self.Tplot.allFeatures)	
	self.Tplot.allFeatures=self.PCAMatrix.transform(self.Tplot.allFeatures)
	self.PCAMatrix2=PCA(n_components=2).fit(self.Tplot2.allFeatures)	
	self.Tplot2.allFeatures=self.PCAMatrix2.transform(self.Tplot2.allFeatures)
	self.PCAMatrix3=PCA(n_components=2).fit(self.Tplot3.allFeatures)	
	self.Tplot3.allFeatures=self.PCAMatrix3.transform(self.Tplot3.allFeatures)
        file.close()
        
    def loadCelloFiles(self):
        print 'start processing cello files'
        count = 0
        for fileName in os.listdir(self.pathDict['cello']):
            fileName = str(self.pathDict['cello']) + str(fileName)
            if fileName.endswith('.wav'):
                if (self.verbose):
                    print 'processing ' + str(fileName)
                sPlot = SoundPlot(fileName)
                self.Tplot.addFeature(sPlot.features)
                self.SVM.addFeature(self.labelDict['cello'], sPlot.features)
                if (self.verbose):
                    print 'extracted unnormalized features: ' + str(sPlot.features)
            count += 1
        print 'finished processing guitar files - ' +str(count)+' in total'
        
    def loadGuitarFiles(self):
        print 'start processing guitar files'
        count = 0
        for fileName in os.listdir(self.pathDict['guitar']):
            fileName = self.pathDict['guitar'] + str(fileName)
            if fileName.endswith('.wav'):
                if (self.verbose):
                    print 'processing ' + str(fileName)
                sPlot = SoundPlot(fileName)
                self.Tplot2.addFeature(sPlot.features)
                self.SVM.addFeature(self.labelDict['guitar'], sPlot.features)
                if (self.verbose):
                    print 'extracted unnormalized features: ' + str(sPlot.features)
            count += 1
        print 'finished processing guitar files - ' +str(count)+' in total'
        
    def loadPianoFiles(self):
        print 'start processing piano files'
        count = 0
        for fileName in os.listdir(self.pathDict['klavier']):
            fileName = str(self.pathDict['klavier']) + str(fileName)
            if fileName.endswith('.wav'):
                if (self.verbose):
                    print 'processing ' + str(fileName)
                sPlot = SoundPlot(fileName)
                self.Tplot3.addFeature(sPlot.features)
                self.SVM.addFeature(self.labelDict['klavier'], sPlot.features)
                if (self.verbose):
                    print 'extracted unnormalized features: ' + str(sPlot.features)
            count += 1
        print 'finished processing guitar files - ' +str(count)+' in total'
        
    def printSVMFeatures(self):
        print 'printing SVM features'    
        self.SVM.printFeatures()
        
    def trainSVM(self):
        print 'starting SVM training'
        self.svmModelName = self.SVM.trainModell()
        print 'model saved unter following name: ' + str(self.svmModelName)
        
    def predictAll(self, dumpToFile=False, filePath=None):
        self.predictCello(dumpToFile, filePath)
        self.predictGuitar(dumpToFile, filePath)
        self.predictPiano(dumpToFile, filePath)
        
    def predictCello(self, dumpToFile=None, filePath=None):
        print 'try to predict a cello'
        for fileName in os.listdir(self.testPathDict['cello']):
            fileName = str(self.testPathDict['cello']) + str(fileName)
            if fileName.endswith('.wav'):
                if (self.verbose):
                    print 'processing ' + str(fileName)
                sPlot = SoundPlot(fileName)
                if dumpToFile:
                    self.SVM.dumpTestDataToFile(filePath, self.labelDict['cello'], sPlot.features)
                p_label, p_acc, p_val = self.SVM.predict(self.labelDict['cello'], sPlot.features)
                print 'expected\t' + str(self.labelDict['cello'])
                print 'actual\t' + str(p_label)
                if self.labelDict['cello'] == p_label[0]:
                    self.statistic[self.labelDict['cello'] - 1][1] = self.statistic[self.labelDict['cello'] - 1][1] + 1
                else:
                     self.statistic[self.labelDict['cello'] - 1][2] = self.statistic[self.labelDict['cello'] - 1][2] + 1
                print 'accuracy\t' + str(p_acc)
        
    def predictGuitar(self, dumpToFile=None, filePath=None):
        print 'try to predict a guitar'
        for fileName in os.listdir(self.testPathDict['guitar']):
            fileName = str(self.testPathDict['guitar']) + str(fileName)
            if fileName.endswith('.wav'):
                if (self.verbose):
                    print 'processing ' + str(fileName)
                sPlot = SoundPlot(fileName)
                if dumpToFile:
                    self.SVM.dumpTestDataToFile(filePath, self.labelDict['guitar'], sPlot.features)
                p_label, p_acc, p_val = self.SVM.predict(self.labelDict['guitar'], sPlot.features)
                print 'expected\t' + str(self.labelDict['guitar'])
                print 'actual\t' + str(p_label)
                print 'accuracy\t' + str(p_acc)
                if self.labelDict['guitar'] == p_label[0]:
                    self.statistic[self.labelDict['guitar'] - 1][1] = self.statistic[self.labelDict['guitar'] - 1][1] + 1
                else:
                     self.statistic[self.labelDict['guitar'] - 1][2] = self.statistic[self.labelDict['guitar'] - 1][2] + 1  
                       
    def predictPiano(self, dumpToFile=None, filePath=None):
        print 'try to predict a piano'
        for fileName in os.listdir(self.testPathDict['klavier']):
            fileName = str(self.testPathDict['klavier']) + str(fileName)
            if fileName.endswith('.wav'):
                if (self.verbose):
                    print 'processing ' + str(fileName)
                sPlot = SoundPlot(fileName)
                if dumpToFile:
                    self.SVM.dumpTestDataToFile(filePath, self.labelDict['klavier'], sPlot.features)
                p_label, p_acc, p_val = self.SVM.predict(self.labelDict['klavier'], sPlot.features)
                print 'expected\t' + str(self.labelDict['klavier'])
                print 'actual\t' + str(p_label)
                print 'accuracy\t' + str(p_acc)
                if self.labelDict['klavier'] == p_label[0]:
                    self.statistic[self.labelDict['klavier'] - 1][1] = self.statistic[self.labelDict['klavier'] - 1][1] + 1
                else:
                     self.statistic[self.labelDict['klavier'] - 1][2] = self.statistic[self.labelDict['klavier'] - 1][2] + 1
            
    def print3DPlot(self):
        print 'starting 3D plot'
        Tplots = [self.Tplot, self.Tplot2, self.Tplot3]
        self.Tplot.multiplePlot(Tplots)		
        
    def printStatistics(self):
        print self.labelDict
        for cl in self.statistic:
            print 'values for class:\t' + str(cl[0])
            print 'success:\t\t' + str(float(cl[1])) + '\t' + str(float((cl[1] * 100)) / (cl[1] + cl[2])) + '%'
            print 'error:\t\t' + str(float(cl[2])) + '\t' + str(float((cl[2] * 100)) / (cl[1] + cl[2])) + '%'
        totalError = 0.0
        totalSuccess = 0.0
        for label in self.statistic:
            totalError = totalError + label[2]
            totalSuccess = totalSuccess + label[1]
        print 'totalError:\t\t' + str(totalError) + '\t' + str((totalError * 100 / (totalError + totalSuccess))) + '%'
        print 'totalSuccess:\t' + str(totalSuccess) + '\t' + str((totalSuccess * 100 / (totalError + totalSuccess))) + '%'
        print 'crossvalidation accurancy:\t' + str(self.SVM.crossValid)
        

if __name__ == "__main__":
    
#    klavierFile = '../sandbox.samples.audio/klavier_mono/Klavier_complete-2.wav'
#    sPlot = SoundPlot(klavierFile)
#    sPlot.createTimePlot()
#    sPlot.createFFTPlot()
#    sPlot.showPlots()
    
    kernelParameters = {}
    kernelParameters['gausGridSearch'] = '-s 0 -t 2 -c 8192.0 -g 8.0 -b 0'
    kernelParameters['gausOriginal'] = '-s 0 -t 2 -c 10000 -b 0'
    kernelParameters['poly'] = '-s 0 -t 1 -b 0 -d 9 -r 2000'
    kernelParameters['tanh'] = '-s 0 -t 3 -b 0 -r 10000'
    #poly mit r=1000 bisher am besten
    
    instrReg = Instrumentrecognition(None, kernelParameters['gausOriginal'], True)
    instrReg.loadAllFiles('trainingdataNEWguitar')
    #instrReg.loadAllFiles()
    #instrReg.loadGuitarFiles()
    #instrReg.SVM.dumpDataToFile('trainingdataNEWguitar')
    #instrReg.trainSVM()
    #instrReg.SVM.printFeatures()
    #instrReg.predictAll(True, 'testdataNEWguitar')
    #instrReg.printStatistics()
    instrReg.print3DPlot()
       
        
