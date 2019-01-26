import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.colors as clr
import numpy as np
#import skimage.measure
import random
image=np.load("out.npz")
label=np.load("out_label.npz")
d1,d2,_,_=np.shape(image['arr_0'])

image_inn=np.reshape(image['arr_0'],[d1/10,10,64,64])
label_inn=np.reshape(label['arr_0'],[d1/10,10,64,64]) 
d1,d2,d3,d4=np.shape(image_inn)
print(np.shape(image_inn))

for ii in range(d1): #10000
    for i in range(d2): #10
        image=image_inn[ii,i,:,:] #[64,64]
        label=label_inn[ii,i,:,:] #[64,64]
        pyplot.figure(1,figsize=(10,2))
        min_val=0;max_val=1;mid_val=(min_val+max_val)/2.0;
        # make a color map of fixed colors
        cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162','#DCE6F1'], N=256)
        bounds=[min_val,mid_val,max_val]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        print("Filenumber",ii,"time steps ",i)
        #Image
        pyplot.subplot(2,10,i+1)
        pyplot.subplots_adjust(hspace = .001)
        if i==0: pyplot.title("Image_input")
        img = pyplot.imshow(image,interpolation='nearest',cmap = cmap)
        pyplot.axis('off')
        #Label
        pyplot.subplot(2,10,d2+i+1)
        if i==0: pyplot.title('Label')
        pyplot.subplots_adjust(hspace = .001)
        img = pyplot.imshow(label,interpolation='nearest',cmap = cmap)
        pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.show()
    pyplot.savefig("all_"+str(ii)+"_"+str(i)+".png")
    pyplot.close()
