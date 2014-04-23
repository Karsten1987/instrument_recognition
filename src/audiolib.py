from pylab import *
import scikits.audiolab as audio
import numpy as np
from scipy import fft
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from os import path as pathHelper
import matplotlib.pyplot as plt
import array

class SoundPlot(object):
    def __init__(self, fileName):
        self.fileName = fileName
        (self.data, sampFreq, self.encoding) = audio.wavread(fileName)
        self.sampFreq = float(sampFreq)
        self.dataLength = float(len(self.data))
        self.duration = self.dataLength / self.sampFreq

        self.setTimeData()
        self.setFFTData()
        self.harmonicsCount = 5
        self.harmonics = self.getHarmonics(self.frqArrayX, self.frqArrayY)
        #self.printSoundData()
        self.setFeatureData()
        
    def printSoundData(self):
        print("wave file:\t\t" + pathHelper.abspath(self.fileName))
        print ("data length:\t\t" + str(self.dataLength))
        print("sampling frequency:\t" + str(self.sampFreq))
        print("soundfile duration:\t" + str(self.duration))
        print("soundfile encoding:\t" + str(self.encoding))
        
    ''' Subplot 1 - showing x=time, y=Amplitude'''
    def createSubPlot(self, plotposition, x, y, color, xName='Time (ms)', yName='Amplitude'):
        sub = subplot(plotposition)
        plot(x, y, color=color)
        xlabel(xName)
        ylabel(yName)
        return sub
    
    def setFeatureData(self):
        self.feature1 = self.frqArrayY[self.harmonics[0]] / self.frqArrayY[self.harmonics[1]]
        self.feature2 = self.frqArrayY[self.harmonics[0]] / self.frqArrayY[self.harmonics[2]]
        self.feature3 = self.frqArrayY[self.harmonics[0]] / self.frqArrayY[self.harmonics[3]]
        self.features = (self.feature1, self.feature2, self.feature3)
#        features = (self.feature1, self.feature2, self.feature3)
#        self.features = self.normalizeFeatureVector(features)
        #print("features: ", self.features)

    def normalizeFeatureVector(self, features):
        normVec = []
        sumOfSquares = 0.0
        for x in features:
            sumOfSquares = sumOfSquares + x**2
        sumOfSquares = math.sqrt(sumOfSquares)
        for x in features:
            normVec.append(x/sumOfSquares)
        return normVec

    def setFFTData(self):
        ''' modulate X-Array in Hz'''
        dataArrayRow = arange(self.dataLength)        
        ''' frq = 1/sec --> Hz'''
        frqArray = dataArrayRow / self.duration 
        self.frqArrayX = frqArray[range(int(self.dataLength) / 2)]
        Y = rfft(self.data) / self.dataLength
        Y = Y[range(len(self.data) / 2)]
        #Y = Y[:,0]
        self.frqArrayY = abs(Y)

    def setTimeData(self):
        timeArray = arange(0, len(self.data), 1)
        self.timeArray = timeArray / self.sampFreq
        #timeArray = timeArray * 1000  #scale to milliseconds

    def createFFTPlot(self):
        self.createSubPlot(212, self.frqArrayX, self.frqArrayY, 'r', 'Freq (Hz)', '|Y(freq)|')
        #plt.plot(maxXPeak, maxYPeak, 'bo')
        plt.plot(self.frqArrayX[self.harmonics[0]], self.frqArrayY[self.harmonics[0]], 'bo')
        plt.plot(self.frqArrayX[self.harmonics[1]], self.frqArrayY[self.harmonics[1]], 'bo')
        #annotate('2nd harmonics', xy=(frqArray[mylist[1]], Y[mylist[1]]), xytext=(700, 0.005), arrowprops=dict(facecolor='black', shrink=0.2))
        plt.plot(self.frqArrayX[self.harmonics[2]], self.frqArrayY[self.harmonics[2]], 'bo')
        plt.plot(self.frqArrayX[self.harmonics[3]], self.frqArrayY[self.harmonics[3]], 'bo')
        #annotate('3rd harmonics', xy=(frqArray[mylist[2]], Y[mylist[2]]), xytext=(1100, 0.005), arrowprops=dict(facecolor='black', shrink=0.2))
        #annotate('Base tone', xy=(maxXPeak, maxYPeak), xytext=(600, 0.01), arrowprops=dict(facecolor='black', shrink=0.2))
        plt.xlim(0, 2000)
        
    def createTimePlot(self):
        self.createSubPlot(211, self.timeArray, self.data, 'k')
        
    def showPlots(self):
        show()

    def getFeatures(self):
        return self.features    

    def lookForMax(self, X, Y, where):
        difference = 80
        start = where - difference
        #ending = where + difference
        maxval = Y[where]
        maxpos = where
        for k in xrange (difference * 2):
            if Y[start + k] > maxval:
                maxval = Y[start + k]
                maxpos = start + k
        return maxval, maxpos

    def findValue(self, tone, frqArray):
        minimum = abs(frqArray[0] - tone);
        for cell in range(len(frqArray)):
            if abs(frqArray[cell] - tone) < minimum:
                minimum = abs(frqArray[cell] - tone)
                min_index = cell
        return min_index        

    def getHarmonics(self, X, Y):
        #maxXPeak, maxYPeak, maxYIndex = self.getMaxPoint(X, Y)
        mylist = []
        harmonics_basic = self.findValue(440, X)
        for x in xrange(1, self.harmonicsCount):
            harmonics = harmonics_basic * x
            maxval, maxpos = self.lookForMax(X, Y, harmonics)
            mylist.append(maxpos)
            #print("mylist: "+str(mylist))
        return mylist

    def getMaxPoint(self, X, Y):
        maxYPeak = Y.max()
        maxYIndex = Y.argmax()
        maxXPeak = X[maxYIndex]
        return maxXPeak, maxYPeak, maxYIndex
    