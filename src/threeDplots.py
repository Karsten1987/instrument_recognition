from pylab import *
import scikits.audiolab as audio
import numpy as np
from scipy import fft
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from os import path as pathHelper
import matplotlib.pyplot as plt
import array


class Plot3d(object):
    def __init__(self):
        self.feature1=[]
        self.feature2=[]
        self.feature3=[]
        self.allFeatures=[]

    def addFeature(self,features):
        self.allFeatures.append(features)
        self.feature1.append(features[0])
        self.feature2.append(features[1])
        self.feature3.append(features[2])

    def plot(self):
        fig = plt.figure()
    	ax = Axes3D(fig)
        ax.scatter(self.feature1,self.feature2,self.feature3,c='r')
        plt.show()
  
    def getFeature1(self):
        return self.feature1
    def getFeature2(self):
        return self.feature2
    def getFeature3(self):
        return self.feature3

    def multiplePlot(self,*cos):
    	fig = plt.figure()
        ax = Axes3D(fig)
    	cos=cos[0]
    	c=('r','g','b')
    	i=0
    	for index in cos:
    		#ax.scatter(index.getFeature1(),index.getFeature2(),index.getFeature3(),c=c[i])
		ax.scatter(index.allFeatures[:,0],index.allFeatures[:,1],c=c[i])
    		#ax.scatter(index.allFeatures[:,0],zeros(len(index.allFeatures[:,0])),c=c[i])
    		i=i+1
    	plt.show()	

