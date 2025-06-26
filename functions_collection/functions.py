import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import cv2
import random
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity as compare_ssim
# import CTProjector.src.ct_projector.projector.numpy as ct_projector

def pick_random_from_segments(X):
    # Generate the list from 0 to X
    full_list = list(range(X + 1))
    
    # Determine the segment size
    segment_size = len(full_list) // 4

    # Initialize selected numbers
    selected_numbers = []

    # Loop through each segment and randomly pick one number
    for i in range(4):
        start = i * segment_size
        end = (i + 1) * segment_size if i < 3 else len(full_list)  # Ensure last segment captures all remaining elements
        segment = full_list[start:end]
        selected_numbers.append(random.choice(segment))

    return selected_numbers


# function: set window level
def set_window(image,level,width):
    # if len(image.shape) == 3:
    #     image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    '''if no random pick, then random_pick = [False,0]; else, random_pick = [True, X]'''
    n = []
    for i in range(0, total_number, interval):
        n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)


# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)


# function: comparison error
def compare(a, b,  cutoff_low = 0 ,cutoff_high = 1000000, extreme = 5000):
    # compare a to b, b is ground truth
    # if a pixel is lower than cutoff (meaning it's background), then it's out of comparison
    c = np.copy(b)
    diff = abs(a-b)
   
    a = a[(c>cutoff_low)& (c < cutoff_high) & (diff<extreme)].reshape(-1)
    b = b[(c>cutoff_low)& (c < cutoff_high) & (diff<extreme)].reshape(-1)

    diff = abs(a-b)

    # mean absolute error
    mae = np.mean(abs(a - b)) 

    # mean squared error
    mse = np.mean((a-b)**2) 

    # root mean squared error
    rmse = math.sqrt(mse)

    # relative root mean squared error
    dominator = math.sqrt(np.mean(b ** 2))
    r_rmse = rmse / dominator * 100

    # structural similarity index metric
    cov = np.cov(a,b)[0,1]
    ssim = (2 * np.mean(a) * np.mean(b)) * (2 * cov) / (np.mean(a) ** 2 + np.mean(b) ** 2) / (np.std(a) ** 2 + np.std(b) ** 2)
    # ssim = compare_ssim(a,b)

    # # normalized mean squared error
    # nmse = np.mean((a-b)**2) / mean_square_value

    # # normalized root mean squared error
    # nrmse = rmse / mean_square_value

    # # peak signal-to-noise ratio
    # psnr = 10 * (math.log10((8191**2) / mse ))

    return mae, mse, rmse, r_rmse, ssim

# function: dice
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


# function: erode and dilate
def erode_and_dilate(img_binary, kernel_size, erode = None, dilate = None):
    img_binary = img_binary.astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)  

    if dilate is True:
        img_binary = cv2.dilate(img_binary, kernel, iterations = 1)

    if erode is True:
        img_binary = cv2.erode(img_binary, kernel, iterations = 1)
    return img_binary


def remove_scatter3D(img,target_label):
    new_img = np.copy(img)
    new_img[new_img == target_label] = 100
    need_to_remove = False
   
    a = np.copy(img)
    
    labeled_image = label(a == target_label)
    regions = regionprops(labeled_image)
    region_sizes = [region.area for region in regions]
      
    if len(region_sizes) > 1:
        need_to_remove = True
        
    # Step 3: Find Largest Region Label
    largest_region_label = np.argmax(region_sizes) + 1  # Adding 1 because labels start from 1

    # Step 4: Create Mask for Largest Region
    largest_region_mask = (labeled_image == largest_region_label)

    # Step 5: Apply Mask to Original Image
    result_image = a.copy()
    result_image[~largest_region_mask] = 0
    new_img[result_image==target_label] = target_label
    new_img[new_img == 100] = 0
    return new_img, need_to_remove