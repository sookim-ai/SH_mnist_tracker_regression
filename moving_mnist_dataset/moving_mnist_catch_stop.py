from PIL import Image
import sys
import os
import math
import numpy as np
import random
from sklearn.preprocessing import normalize

###########################################################################################
# script to generate moving mnist video dataset (frame by frame) as described in
# [1] arXiv:1502.04681 - Unsupervised Learning of Video Representations Using LSTMs
#     Srivastava et al
# by Tencia Lee
# saves in hdf5, npz, or jpg (individual frames) format
###########################################################################################

# helper functions
def arr_from_img(im,shift=0):
    w,h=im.size
    arr=im.getdata()
    c = np.product(arr.size) / (w*h)
    return np.asarray(arr, dtype=np.float32).reshape((h,w,c)).transpose(2,1,0) / 255. - shift

def get_picture_array(X, index, shift=0):
    ch, w, h = X.shape[1], X.shape[2], X.shape[3]
    ret = ((X[index]+shift)*255.).reshape(ch,w,h).transpose(2,1,0).clip(0,255).astype(np.uint8)
    if ch == 1:
        ret=ret.reshape(h,w)
    return ret

# loads mnist from web on demand
def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
        return data / np.float32(255)
    return load_mnist_images('train-images-idx3-ubyte.gz')

# generates and returns video frames in uint8 array
def generate_moving_mnist(shape=(64,64), seq_len=10, seqs=100000, num_sz=28, nums_per_image=2):
    mnist = load_dataset()
    width, height = shape
    lims = (x_lim, y_lim) = width-num_sz, height-num_sz
    dataset = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
    #label
    dataset_label = np.empty((seq_len*seqs, 1, width, height), dtype=np.uint8)
    for seq_idx in xrange(seqs):
        # randomly generate direc/speed/position, calculate velocity vector
        direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
        speeds = np.random.randint(5, size=nums_per_image)+2
        veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in zip(direcs, speeds)]
        num_list=[0,1,2,3,4,5,7] #only choose in num_list because those 5 digits looks distinctive
        num_1=random.choice(num_list)
        num_list.remove(num_1)
        num_2=random.choice(num_list)
        mnist_images = [Image.fromarray(get_picture_array(mnist,r,shift=0)).resize((num_sz,num_sz), Image.ANTIALIAS) \
               for r in [num_1,num_2] ]
        #prevent occulsion by constraining moving area for object A and B
        # decide bounding box of object A and B and prevent object move outside of bounding box
        xs1,ys1,xe1,ye1,xs2,ys2,xe2,ye2=random.choice([(0,0.5,1,1,0,0,1,0.5),(0,0,1,0.5,0,0.5,1,1),(0,0,0.5,1,0.5,0,1,1),(0.5,0,1,1,0,0,0.5,1),(0,0.5,0.5,1,0.5,0,1,0.5),(0.5,0,1,0.5,0,0.5,0.5,1),(0,0,0.5,0.5,0.5,0.5,1,1),(0.5,0.5,1,1,0,0,0.5,0.5)]) 
        #decide initial position
        positions = [((np.random.rand()*(x_lim*(xe1-xs1))+(x_lim*(xs1))), (np.random.rand()*(y_lim*(ye1-ys1))+(y_lim*(ys1)))), ((np.random.rand()*(x_lim*(xe2-xs2))+(x_lim*(xs2))), (np.random.rand()*(y_lim*(ye2-ys2))+(y_lim*(ys2)))) ]
        for frame_idx in xrange(seq_len):
            canvases = [Image.new('L', (width,height)) for _ in xrange(nums_per_image)] #2
            canvas = np.zeros((1,width,height), dtype=np.float32)
            #Pick one number as label
            canvas_label = np.zeros((1,width,height), dtype=np.float32)
            #Pick stopping time
            #After stop_time target object will be gone
            #SH : Below is the part to write the input and label image, we should modify code in a way to write position label
            ###################################################################################################################
            ####################################################################################################################
            stop_time=np.random.randint(5,9)
            for i,canv in enumerate(canvases):
                #image
                if i == 0:
                    if frame_idx < stop_time:
                        canv.paste(mnist_images[i], tuple(map(lambda p: int(round(p)), positions[i])))
                        canvas += arr_from_img(canv, shift=0)
                    else:
                        canvas += np.zeros((1,width,height), dtype=np.float32)
                else: 
                    canv.paste(mnist_images[i], tuple(map(lambda p: int(round(p)), positions[i])))
                    canvas += arr_from_img(canv, shift=0)
                print("CHECK : ",seq_idx,frame_idx,i,canv)
                #label
                if i == 0:
                    #The track object still exist in frame
                    if frame_idx < stop_time:
                       canv.paste(mnist_images[i], tuple(map(lambda p: int(round(p)), positions[i])))
                       #SH: Write position label here like this
                       print("track position: ", positions[i], "Confidence socore", "1")
                       #SOO: Write position here
                       canvas_label += arr_from_img(canv, shift=0)        
                    #After STopping time,the track object disappears. 
                    else:
                       #writing label image
                       canvas_label += np.zeros((1,width,height), dtype=np.float32)
                       #SH: Write postion label here like this
                       print("track position: ", "[0,0]", "Confidence socore", "0") 
            ##########################################################################################################################
            ##########################################################################################################################

            # update positions based on velocity
            next_pos = [map(sum, zip(p,v)) for p,v in zip(positions, veloc)]
            # prevent occulusion
            next_1_x,next_1_y=next_pos[0]
            next_2_x,next_2_y=next_pos[1]
            next_1_x,next_1_y,next_2_x,next_2_y=next_1_x/x_lim,next_1_y/y_lim,next_2_x/x_lim,next_2_y/y_lim
            # constrain moving object A and B in bounding box
            next_pos = [[(np.random.rand()*(x_lim*(xe1-xs1))+(x_lim*(xs1))), (np.random.rand()*(y_lim*(ye1-ys1))+(y_lim*(ys1)))], [(np.random.rand()*(x_lim*(xe2-xs2))+(x_lim*(xs2))), (np.random.rand()*(y_lim*(ye2-ys2))+(y_lim*(ys2)))] ]
            # bounce off wall if a we hit one
            for i, pos in enumerate(next_pos):
                for j, coord in enumerate(pos):
                    if coord < -2 or coord > lims[j]+2:
                        veloc[i] = tuple(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j+1:]))
            positions = [map(sum, zip(p,v)) for p,v in zip(positions, veloc)]
            # copy additive canvas to data array
            dataset[seq_idx*seq_len+frame_idx] = (canvas * 255).astype(np.uint8).clip(0,255)
            #label
            dataset_label[seq_idx*seq_len+frame_idx] = (canvas_label * 255).astype(np.uint8).clip(0,255)
    return dataset,dataset_label

def main(dest, filetype='npz', frame_size=64, seq_len=30, seqs=100, num_sz=28, nums_per_image=2):
    dat,dat_label = generate_moving_mnist(shape=(frame_size,frame_size), seq_len=seq_len, seqs=seqs, \
                                num_sz=num_sz, nums_per_image=nums_per_image)
    n = seqs * seq_len
    if filetype == 'hdf5':
        import h5py
        from fuel.datasets.hdf5 import H5PYDataset
        def save_hd5py(dataset, destfile, indices_dict):
            f = h5py.File(destfile, mode='w')
            images = f.create_dataset('images', dataset.shape, dtype='uint8')
            images[...] = dataset
            split_dict = dict((k, {'images':v}) for k,v in indices_dict.iteritems())
            f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
            f.flush()
            f.close()
        indices_dict = {'train': (0, n*9/10), 'test': (n*9/10, n)}
        save_hd5py(dat, dest, indices_dict)
    elif filetype == 'npz':
        n,d1,d2,d3=np.shape(dat)
        dat_norm=np.reshape(normalize(np.reshape(dat,[n,d1*d2*d3]),axis=1),[n,d1,d2,d3])
        dat_label_norm=np.reshape(normalize(np.reshape(dat_label,[n,d1*d2*d3]),axis=1),[n,d1,d2,d3])
        np.savez(dest, dat)
        np.savez(dest+"_label",dat_label)
        #np.save("image.npy", dat)
        #np.save("label.npy",dat_label)
    elif filetype == 'jpg':
        for i in xrange(dat.shape[0]):
            Image.fromarray(get_picture_array(dat, i, shift=0)).save(os.path.join(dest, '{}.jpg'.format(i)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument('--dest', type=str, dest='dest')
    parser.add_argument('--filetype', type=str, dest='filetype')
    parser.add_argument('--frame_size', type=int, dest='frame_size')
    parser.add_argument('--seq_len', type=int, dest='seq_len') # length of each sequence
    parser.add_argument('--seqs', type=int, dest='seqs') # number of sequences to generate
    parser.add_argument('--num_sz', type=int, dest='num_sz') # size of mnist digit within frame
    parser.add_argument('--nums_per_image', type=int, dest='nums_per_image') # number of digits in each frame
    args = parser.parse_args(sys.argv[1:])
    print(vars(args).items())
    main(**{k:v for (k,v) in vars(args).items() if v is not None})
