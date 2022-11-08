import astropy.convolution as asc
import fabio
from matplotlib import pyplot as plt
import numpy as np

# main directory for the frames, leave empty if working in the local directory
main_dir=''
#main_dir='/asap3/petra3/gpfs/p21.1/2022/data/11016530/processed/mask_perkin/'
# saving directory for mask(s), leave empty if saving localy
save_dir=''
#save_dir='/asap3/petra3/gpfs/p21.1/2022/data/11016530/processed/mask_perkin/'
# frame(s) to be analysed for a mask
frame_flatf=['summed_4pf.tif']
#frame_flatf=['summed_4pf.tif','summed_2pf.tif','summed_1pf.tif','summed_05pf.tif']

# number of pixels to be masked on the edges, use a minimum of 10 px
border_lim=15
# standard-deviaton scaling(s) to be used for the mask
std_scale=[2]#,2.5,3]
# standard-deviation scaling (>1) for the convolution filter
# when too large, the computation is slow
std_kernel=3

# Plot Flag: True --> plotting will be made; False --> no plotting
plot_flag=True

# Save Flag
save_flag=False

###############################################################################
################    Evaluating, Plotting and Saving Mask(s)    ################
###############################################################################
try:
    # testing if first frame exists and initializing the mask_prov array
    im=fabio.open(main_dir+frame_flatf[0])
    img=im.data.astype("float32")
    sz=im.shape
    mask_prov=np.zeros([len(frame_flatf),len(std_scale),sz[0],sz[1]])
except:
    raise Exception("Can't open (first) frame. Chack the directory and frame name.")

# analysing each frame individually
for frame_idx in range(0,len(frame_flatf)):
    # loading frame
    im=fabio.open(main_dir+frame_flatf[frame_idx])
    img=im.data.astype("float32")

    # dimension of the frame
    sz=im.shape
    # mean intensity and standatd deviation of the frame
    avg0=img.mean()
    std0=img.std()
    
    # defining basic masks usefull for the convolution filter
    # finding negative and zero values (if any)
    msk0=np.where(img<=0)
    # finding low values
    msk1=np.where(img<avg0/2)
    # masking high values
    msk2=np.where(img>avg0*2)
    
    # convolution filter
    kernel = asc.Gaussian2DKernel(x_stddev=std_kernel)
    # copying the data to an independent variable
    img_prov=np.copy(img, order='K')
    # applying basic masks on the copy
    # border is left unchanged, and will be masked at the very end
    # to exclude masked values from the filter, astropy supports nan values
    img_prov[msk0]=np.nan
    img_prov[msk1]=np.nan
    img_prov[msk2]=np.nan
    # creating convoluted data, which will be smoothed and thus replicating the background
    img_conv=asc.convolve(img_prov,kernel)
    
    # creating a flat data: raw-background
    img_diff=img-img_conv
    # mean intensity and standatd deviation of the flat data
    avgf=img_diff.mean()
    stdf=img_diff.std()
    
    # plotting frame and background
    if plot_flag:
        # defining figure handle
        plt.figure("Data: "+frame_flatf[frame_idx])
        # original frame plot
        ax1=plt.subplot(121)
        ax1.imshow(img,vmin=avg0-2*std0, vmax=avg0+2*std0, origin='lower',
                   interpolation='none', cmap='viridis')
        plt.title('raw flat-field frame')
        #plt.show()
        # filtered frame: background
        ax2=plt.subplot(122)
        ax2.imshow(img_conv,vmin=avg0-2*std0, vmax=avg0+2*std0, origin='lower',
                  interpolation='none', cmap='viridis')
        plt.title('filtered flat-field frame')
        plt.show()
    
    # creating (provisionary) masks and ploting
    # provisionary in case there is more than one input frame
    for mask_idx in range(0,len(std_scale)):
        # finging the outlier pixels
        mskc1=np.where(img_diff<(avgf-(std_scale[mask_idx]*stdf)))
        mskc2=np.where(img_diff>(avgf+(std_scale[mask_idx]*stdf)))
        
        # initializing current mask
        #mask_prov[frame_idx,mask_idx]=np.zeros(sz)
        # merging masks
        mask_prov[frame_idx,mask_idx,msk0[0],msk0[1]]=1
        mask_prov[frame_idx,mask_idx,msk1[0],msk1[1]]=1
        mask_prov[frame_idx,mask_idx,msk2[0],msk2[1]]=1
        mask_prov[frame_idx,mask_idx,mskc1[0],mskc1[1]]=1
        mask_prov[frame_idx,mask_idx,mskc2[0],mskc2[1]]=1
        #masking the borders
        mask_prov[frame_idx,mask_idx,0:border_lim,:]=1
        mask_prov[frame_idx,mask_idx,:,0:border_lim]=1
        mask_prov[frame_idx,mask_idx,sz[0]-border_lim+1:sz[0],:]=1
        mask_prov[frame_idx,mask_idx,:,sz[1]-border_lim+1:sz[1]]=1
        
        # plotting differential frame and mask
        if plot_flag:
            plt.figure('Mask: '+frame_flatf[frame_idx]+' - %2.1f*std'%std_scale[mask_idx])
            ax1=plt.subplot(121)
            ax1.imshow(img_diff,vmin=avgf-(std_scale[mask_idx]*stdf), 
                       vmax=avgf+(std_scale[mask_idx]*stdf), origin='lower',
                      interpolation='none',cmap='gray')
            plt.title('differential frame')
            #plt.show()
            
            ax2=plt.subplot(122)
            ax2.imshow(mask_prov[frame_idx,mask_idx],vmin=0.,vmax=1.,origin='lower',
                       interpolation='none',cmap='gray')
            plt.title('mask')
            plt.show()
        
# merging masks obtained from different frames
mask=mask_prov.sum(0)
mask=mask.astype(dtype=bool)
# saving mask as numpy array
for mask_idx in range(0,len(std_scale)):
    plt.figure('Merged Mask %2.1f*std'%std_scale[mask_idx])
    plt.imshow(mask[mask_idx,:,:],vmin=0.,vmax=1.,origin='lower',
              interpolation='none',cmap='gray')
    plt.show()
    if save_flag:
        np.save(save_dir+'mask_%3.2fstd.npy'%std_scale[mask_idx],mask[mask_idx,:,:])
    


