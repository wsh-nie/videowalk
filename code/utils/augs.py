import torchvision
import skimage

import torch
from torchvision import transforms

import numpy as np
from PIL import Image

IMG_MEAN = (0.4914, 0.4822, 0.4465)
IMG_STD  = (0.2023, 0.1994, 0.2010)
NORM = [transforms.ToTensor(), 
        transforms.Normalize(IMG_MEAN, IMG_STD)]
"""
ToTensor():
    Convert a PIL Image or numpy.ndarray to tensor. 
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
Normalize:
    Normalize a tensor image with mean and standard deviation.
    Parameters:
        mean-
            Sequence of means for each channel.
        std-
            Sequence of standard deviations for each channel.
        inplace-
            Bool to make this operation in-place.
"""

class MapTransform(object):
    def __init__(self, transforms, pil_convert=True):
        self.transforms = transforms
        self.pil_convert = pil_convert

    def __call__(self, vid):
        if isinstance(vid, Image.Image):
            return np.stack([self.transforms(vid)])
        
        if isinstance(vid, torch.Tensor):
            vid = vid.numpy()

        if self.pil_convert:
            x = np.stack([np.asarray(self.transforms(Image.fromarray(v))) for v in vid])
            return x
        else:
            return np.stack([self.transforms(v) for v in vid])
    
def n_patches(x, n, transform, shape=(64, 64, 3), scale=[0.2, 0.8]):
    ''' unused '''
    if shape[-1] == 0:
        shape = np.random.uniform(64, 128)
        shape = (shape, shape, 3)

    crop = transforms.Compose([
        lambda x: Image.fromarray(x) if not 'PIL' in str(type(x)) else x,
        transforms.RandomResizedCrop(shape[0], scale=scale)
    ])    

    if torch.is_tensor(x):
        x = x.numpy().transpose(1,2, 0)
    
    P = []
    for _ in range(n):
        xx = transform(crop(x))
        P.append(xx)

    return torch.cat(P, dim=0)


def patch_grid(transform, shape=(64, 64, 3), stride=[0.5, 0.5]):
    stride = np.random.random() * (stride[1] - stride[0]) + stride[0] # the ratio of overlapping.
    stride = [int(shape[0]*stride), int(shape[1]*stride), shape[2]]
    
    spatial_jitter = transforms.Compose([
        lambda x: Image.fromarray(x), # convert numpy to PIL.image type
        transforms.RandomResizedCrop(shape[0], scale=(0.7, 0.9))
    ])

    def aug(x):
        if torch.is_tensor(x):
            x = x.numpy().transpose(1, 2, 0)
        elif 'PIL' in str(type(x)):
            x = np.array(x)#.transpose(2, 0, 1)
        
        winds = skimage.util.view_as_windows(x, shape, step=stride) # Split images_x into patches, the diamension is [x.shape[i]/step.shape[i]] + step.shape

        """
        Rolling window view of the input n-dimensional array.
        Parameters:
            arr_in : ndarray
                N-d input array.
            window_shape : integer or tuple of length arr_in.ndim
                Defines the shape of the elementary n-dimensional orthotope (better know as hyperrectangle) of the rolling window view. If an integer is given, the shape will be a hypercube of sidelength given by its value.
            step : integer or tuple of length arr_in.ndim
                Indicates step size at which extraction shall be performed. If integer is given, then the step is uniform in all dimensions.
        Return :
            arr_out : ndarray
                (rolling) window view of the input array. If arr_in is non-contiguous, a copy is made. 
        """
        winds = winds.reshape(-1, *winds.shape[-3:])

        P = [transform(spatial_jitter(w)) for w in winds] # transform for each patch
        return torch.cat(P, dim=0) # cat all of patch into the first dimension.

    return aug


def get_frame_aug(frame_aug, patch_size):
    train_transform = []

    if 'cj' in frame_aug:
        _cj = 0.1
        train_transform += [
            #transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(_cj, _cj, _cj, 0),
        ]
    if 'flip' in frame_aug:
        train_transform += [transforms.RandomHorizontalFlip()]

    train_transform += NORM
    train_transform = transforms.Compose(train_transform)
    """
    Composes several transforms together. This transform does not support torchscript. Please, see the note below.
    Parameters:
        transforms-
            list of transforms to compose
    """

    print('Frame augs:', train_transform, frame_aug)

    if 'grid' in frame_aug:
        aug = patch_grid(train_transform, shape=np.array(patch_size))
    else:
        aug = train_transform

    return aug


def get_frame_transform(frame_transform_str, img_size):
    tt = []
    fts = frame_transform_str
    norm_size = torchvision.transforms.Resize((img_size, img_size))

    if 'crop' in fts:
        tt.append(torchvision.transforms.RandomResizedCrop(
            img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2),)
        """
        Crop a random portion of image and resize it to a given size.
        Parameters: 
            size- 
                expected output size of the crop, for each edge. 
            scale=(0.08, 1.0)
                Specifies the lower and upper bounds for the random area of the crop, before resizing.
            ratio=(0.75, 1.3333333333333333)
                lower and upper bounds for the random aspect ratio of the crop, before resizing.
            interpolation==<InterpolationMode.BILINEAR: 'bilinear'>
                Desired interpolation enum defined by torchvision.transforms.InterpolationMode
        """
    else:
        tt.append(norm_size)

    if 'cj' in fts:
        _cj = 0.1
        # tt += [#transforms.RandomGrayscale(p=0.2),]
        tt += [transforms.ColorJitter(_cj, _cj, _cj, 0),]
        """
        Randomly change the brightness, contrast, saturation and hue of an image. 
        Parameters:
            brightness=0.
                How much to jitter brightness. brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
            contrast=0.
                How much to jitter contrast. contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]. 
            saturation=0.
                How much to jitter saturation. saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation] or the given [min, max]. 
            hue=0.
                How much to jitter hue. hue_factor is chosen uniformly from [-hue, hue] or the given [min, max]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        """

    if 'flip' in fts:
        tt.append(torchvision.transforms.RandomHorizontalFlip())
        """
        Horizontally flip the given image randomly with a given probability. 
        Parameters:
            p=0.5
                probability of the image being flipped.
        """

    print('Frame transforms:', tt, fts)

    return tt

def get_train_transforms(args):
    norm_size = torchvision.transforms.Resize((args.img_size, args.img_size)) # set a Resize transformers

    frame_transform = get_frame_transform(args.frame_transforms, args.img_size) # ('crop', 256)
    """
        RandomResizedCrop(size=(230, 230), scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=PIL.Image.BILINEAR)
    """
    frame_aug = get_frame_aug(args.frame_aug, args.patch_size) # ('grid', 64)
    """
        lambda x: Image.fromarray(x);
        RandomResizedCrop(shape[0], scale=(0.7, 0.9))
        NORM
    """
    frame_aug = [frame_aug] if args.frame_aug != '' else NORM
    
    transform = frame_transform + frame_aug

    train_transform = MapTransform(
            torchvision.transforms.Compose(transform)
        )

    plain = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        norm_size, 
        *NORM,
    ])

    def with_orig(x):
        x = train_transform(x), \
            plain(x[0]) if 'numpy' in str(type(x[0])) else plain(x[0].permute(2, 0, 1))

        return x

    return with_orig

