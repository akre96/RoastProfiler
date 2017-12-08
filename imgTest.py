import numpy as np
import scipy as sp
import pandas as pd
import PIL as pil
import cv2
from skimage import io, filters
import matplotlib.pyplot as plt

def roastA(bean,ctrl):
    roast=cv2.imread(bean)
    control=cv2.imread(ctrl)

    roastG=cv2.cvtColor(roast,cv2.COLOR_BGR2GRAY)
    controlG=cv2.cvtColor(control,cv2.COLOR_BGR2GRAY)

    graynorm=roastG.mean()/controlG.mean()


    green=roast[:,:,1]
    greenC=control[:,:,1]
    gnorm=green.mean()/greenC.mean()

    red=roast[:,:,0]
    redC=control[:,:,0]
    rnorm=red.mean()/redC.mean()

    blue=roast[:,:,2]
    blueC=control[:,:,2]
    bnorm=blue.mean()/blueC.mean()

    return np.array([graynorm,rnorm,gnorm,bnorm])



drd=roastA("drd.JPG","crd.JPG")
lrd=roastA("lrd.JPG","crd.JPG")


dr=roastA("dr.JPG","cr.JPG")
lr=roastA("lr.JPG","cr.JPG")
print 'Whole Bean Dark Roast'
print dr
print drd

print 'Whole Bean Light Roast'
print lr
print lrd

print 'Whole Bean lighting difference'
print drd-dr
print lrd-lr

gdrd=roastA("gdrd.JPG","crd.JPG")
glrd=roastA("glrd.JPG","crd.JPG")


gdr=roastA("gdr.JPG","cr.JPG")
glr=roastA("glr.JPG","cr.JPG")

print 'Ground Bean lighting difference'
print gdrd-gdr
print glrd-glr

data={
    'wlr':lr,
    'wdr':dr,
    'wlrd':lrd,
    'wdrd':drd,
    'glr':glr,
    'gdr':gdr,
    'glrd':glrd,
    'gdrd':gdrd

}
plt.figure()
dataDF=pd.DataFrame(data)
dataDF=dataDF.transpose()
dataDF.columns=["GrayScale","Red","Green","Blue"]

ind=np.arange(len(dataDF))

plt.plot(ind,dataDF['Red'],'ro-')
plt.plot(ind,dataDF['Green'],'go-')
plt.plot(ind,dataDF['Blue'],'bo-')
plt.plot(ind,dataDF['GrayScale'],'ko-')
plt.xticks(ind,dataDF.index.values)
plt.ylabel('White Normalized Color Intensity')
plt.xlabel('Grind Roast Lighting')

plt.figure()

ind=np.arange(len(dataDF)/2)

plt.plot(ind,dataDF['Red'].iloc[0:4],'ro-')
plt.plot(ind,dataDF['Green'].iloc[0:4],'go-')
plt.plot(ind,dataDF['Blue'].iloc[0:4],'bo-')
plt.plot(ind,dataDF['GrayScale'].iloc[0:4],'ko-')
plt.xticks(ind,dataDF.index.values[0:4])
plt.ylabel('White Normalized Color Intensity')
plt.xlabel('Grind Roast Lighting')
plt.title('Ground Beans')

plt.figure()

ind=np.arange(len(dataDF)/2)

plt.plot(ind,dataDF['Red'].iloc[4:8],'ro-')
plt.plot(ind,dataDF['Green'].iloc[4:8],'go-')
plt.plot(ind,dataDF['Blue'].iloc[4:8],'bo-')
plt.plot(ind,dataDF['GrayScale'].iloc[4:8],'ko-')
plt.xticks(ind,dataDF.index.values[4:8])
plt.ylabel('White Normalized Color Intensity')
plt.xlabel('Grind Roast Lighting')
plt.title('Whole Beans')
plt.show()
