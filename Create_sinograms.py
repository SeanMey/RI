import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale,resize
from skimage.transform import iradon
import os
import numpy as np
import scipy.misc

def distortion(imagein,reng,width):
        # disturtion
    for nremg in range(2,reng):
        #print(range(0, width-60, nremg))
        imagein[:,range(0, width, nremg)]=1
    return imagein

def radtra(file,downR):
    image = imread( file, as_grey=True)
    width, height = np.shape(image)
    imageSqsize=max(width,height)
    image=resize(image,(imageSqsize,imageSqsize),mode='constant')
    ###image = rescale(image, scale=0.8, mode='reflect')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_title("Original")
    ax1.imshow(image, cmap=plt.cm.Greys_r)

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=False)
    
    
    sinogram = distortion(sinogram,downR,width)
    
    ax2.set_title("Radon transform\n(Sinogram)")
    ax2.set_xlabel("Projection angle (deg)")
    ax2.set_ylabel("Projection position (pixels)")
    ax2.imshow(sinogram, cmap=plt.cm.Greys_r,extent=(0, 360, 0, sinogram.shape[0]), aspect='auto')

    fig.tight_layout()
    plt.show()
    
    
    reconstruction_fbp = iradon(sinogram, theta=theta, circle=False)
    
    #scipy.misc.imsave(file.replace("images", "images_f_sinogram"), reconstruction_fbp)

    
    error = reconstruction_fbp - image
    error= 2
    print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

    imkwargs = dict(vmin=-0.2, vmax=0.2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
    ax1.set_title("Reconstruction\nFiltered back projection")
    ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    ax2.set_title("Reconstruction error\nFiltered back projection")
    ax2.imshow(image, cmap=plt.cm.Greys_r, **imkwargs)
    plt.show()
    
    
    
folder='images';
downR=10
#downR=8

filesList = os.listdir(folder)
for file in filesList[0:5]:
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".png"): 
        fileForSinogram = folder+"/"+filename
        #print(fileForSinogram)
        radtra(fileForSinogram,downR)
        continue
    else:
        continue