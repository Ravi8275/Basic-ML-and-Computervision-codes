#!/usr/bin/env python
# coding: utf-8

# # Importing all the necessary libraries

# In[1228]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
from sklearn.metrics import roc_curve,roc_auc_score,auc,confusion_matrix,accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image as im
import scipy.io as io
import time


# # 1.1

# # Function to convert mat files

# In[540]:


def CreateFolder(name):
    '''
    *This function is creating new folder*
    name - represents the folder name and must be str type
    '''
    os.makedirs(os.path.join(name), exist_ok = True)
    

def LoadImage(file_folder, n, FolderName = 'Images'):
    '''
    *This function is converting Images from .mat file to .jpeg file and saving them*
    FolderName - represents the creating folder for images
    file_folder - represents location of .mat files: '/home/N/jupyter_dir/IPPR/database_1/'
    n - represents number of our files
    '''
    mat = io.loadmat(file_folder + '/ground_truth_p44.mat') #converting from .mat format to readable for Python (here is dict)
    data = im.fromarray(mat['A']) #image data is inside of the mat(type is dict) and array name is 'img'
    CreateFolder(FolderName) #calling function CreateFolder to create folder
    data.save('Images' + '/ground_truth_p44.png')#saving images


# In[541]:


file_folder = r"/Users/raviirt/Downloads/gt"
n = 0
LoadImage(file_folder, n)


# __Reading the Gray Image__

# In[542]:


#Reading the citated image file into variable 'Image'
#Imread function from cv2 is used to read the image file by mentioning it's dedicated path
#It is stated in the questionarie to convert the image into grayscale.so writing '0' while reading the image,so it will bee saved in gray scale format
Gray_Image=cv2.imread('/Users/raviirt/Downloads/Images/p44.jpeg',0)


# In[543]:


#Having a glance at the Image
plt.imshow(Gray_Image,cmap='gray')
plt.title('Gray Scale image')
plt.savefig('Gray Scale Image')


# __Reading the Ground truth image__

# In[842]:


#reading the ground truth image
Ground_truth_image=cv2.imread('/Users/raviirt/Downloads/Images\ground_truth_p44.png',0)


# In[843]:


plt.imshow(Ground_truth_image,cmap='gray')
#multiplying with 255 as the image is binary image
plt.title('Ground Truth image')
plt.savefig('GRound Truth Image')


# __Gray Image Analysis through Histogram__

# In[546]:


Histogram_values = cv2.calcHist([Gray_Image], [0], None, [256], [0, 256])


# - Passing the Gray Image for which histogram has to be calculated
# - 0 represents the channel we want to plot about,being the gray channel which got only one channel so 0
# - None an input for masking parameter which we are not doing currently therefore None
# - 256 represents the hist value 
# - range is mentioned as 0-256 

# In[547]:


plt.plot(Histogram_values)
plt.title('Histogram for Gray scale Baloon image')
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.savefig('Histogram for a Gray scale image')


# - we can clearly see two peaks arrounf 100-150 intensity range (Because of the presence of two bright objects)

# __calculating the characteristics of the image__

# In[548]:


def Calculate_Metrics(Picture):
    Image_array=np.array(Picture)
    Average_Pixel_value=np.mean(Picture)
    Variance=np.var(Picture)
    Skewness=sp.skew(Picture,axis=None)
    print('Arithmetic mean of pixels:',Average_Pixel_value,
      '\nVariance:',Variance,'\nSkewness:',Skewness)


# In[549]:


Calculate_Metrics(Gray_Image)


# __Image Thresholding__/__Varying Threshold__

# In[550]:


Ground_truth_image.shape


# In[551]:


Gray_Image.shape


# __Checking the thresholder performance arround pixel mean value__

# In[889]:


initial_thresh_values=range(100,110)
plt.figure(figsize=(8,8))
for i,th_values in enumerate(initial_thresh_values):
    v,thresh=cv2.threshold(Gray_Image,th_values,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    plt.subplot(2,5,i+1)
    plt.imshow(thresh,cmap='gray')
plt.tight_layout()


# # Performing threshold over the complete range

# # Real performing over the entire range

# In[1097]:


Threshold_values=range(0,256)#setting the threshold value range as mentioned hovering over entire pixel range
False_alarm_rate_list=[]
Detection_rate_list=[]
start_time=time.time()
for threshold_value in Threshold_values:
    val,Threshold_image=cv2.threshold(Gray_Image,threshold_value,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    Threshold_image_binary=Threshold_image/255
    True_positives = np.sum(np.logical_and(Threshold_image_binary == 1, Ground_truth_image == 1))
    print('true positive',True_positives)
    True_Negatives = np.sum(np.logical_and(Threshold_image_binary == 0, Ground_truth_image == 0))
    print('True negative :',True_Negatives)
    False_Positives = np.sum(np.logical_and(Threshold_image_binary == 1, Ground_truth_image == 0))
    print('False positive:',False_Positives)
    False_Negatives = np.sum(np.logical_and(Threshold_image_binary == 0, Ground_truth_image == 1))
    print('false negative',False_Negatives)
    denominator_FAR = False_Positives + True_Negatives
    FAR = False_Positives / denominator_FAR if denominator_FAR > 0 else 0
    denominator_TPR = True_positives + False_Negatives
    TPR = True_positives / denominator_TPR
    False_alarm_rate_list.append(FAR)
    Detection_rate_list.append(TPR)
end_time=time.time()


# __Plotting ROC Curve__

# In[1098]:


elapsed_time=end_time-start_time
print(f'Total time taken for one threshold detector:{elapsed_time/60} minutes')


# In[1178]:


plt.figure(figsize=(8, 8))
plt.plot(False_alarm_rate_list,Detection_rate_list,color='orange',lw=5,label='One threshold curve')
plt.xlabel('False Alarm Rate (FAR)')
plt.ylabel('Detection Rate (DR)')
plt.title('ROC Curve for one threshold channel')
plt.legend(loc='lower right')
plt.savefig('ROC curve for one threshold detector')


# In[1179]:


one_Threshold_detector_result=Gray_Image>103


# In[1181]:


plt.imshow(one_det_result,cmap='gray')
plt.title('Perfect threshold value applied Gray Image')
plt.savefig('Perfect threshold value applied Gray Image')


# # Finding the best threshold value through Youden J statistic menthod

# In[892]:


#Converting both the detection rates false alarm rates to array to perform further operations
dr=np.array(Detection_rate_list)
fr=np.array(False_alarm_rate_list)
J_value=dr-fr
#We know that Youden value considers a better trade off between true positive rate and false alarm rate


# In[893]:


#Retrieving the index where we found the better combination of true postive and false alarm rate
best_index=np.argmax(J_value)


# In[894]:


#retrieving the best Detection rate 
dr[best_index]


# In[895]:


#retrieving the best false alarm rate
fr[best_index]


# In[896]:


#Finally getting the best threshold value with appropriate detection rate and false alarm rate
Threshold_values[best_index]


# # Outcome Discussion

# __Time Taken to perform thresholding__ : 0.01 Minutes

# __As we can see the total time taken to find out the best posiible threshold value is very less i.e 0.01 minutes.So instead of going through a rough value considering theoretical information and having inefficient threshold valuen is not advisable instead it is better to hover over the entire range of threshold values and find the best combination of evaluation metrics and the the threshold value at that point.So the best Detection rate we achieved is 0.9380 and best false alarm rate is 0.0798 which are considerably good__.

# # 1.2 Creating a detector for RGB image

# In[936]:


#Reading the image(RGB)
Colored_Image=cv2.imread('/Users/raviirt/Downloads/Images/p44.jpeg')#Image color channel will be BGR as we are reading using opencv
plt.imshow(Colored_Image)
plt.title('Coloured Image')
plt.savefig('Colored Image')


# __RGB Image Analysis Through Histogram__

# In[902]:


colors=['b','g','r']
channel_names=['Blue','Green','Red']
custom_colors = [(0, 0, 1), (0, 0.5, 0), (1, 0, 0)]
for i,color in enumerate(colors):
    Color_Histogram_Values=cv2.calcHist([Colored_Image],[i],None,[255],(0,256))
    plt.plot(Color_Histogram_Values,color=custom_colors[i],label=channel_names[i])
    plt.title('Histogram of Coloured image(RGB channel)')
    plt.xlabel('Pixel frequency')
    plt.ylabel('pixel intensity')
    plt.legend()
plt.savefig('Histogram of Colored image(RGB channel)')
    


# - Considering thresholds values by looking at the histogram.
# - Threshold value for Blue channel t1-95
# - Threshold value for green channel t2-117
# - Threshold value for red channel t3-131

# In[903]:


#Splitting the color channels 
B,G,R=cv2.split(Colored_Image)
#BGR and not RGB


# In[904]:


plt.figure(figsize=(15,10))
plt.subplot(1,3,1)
plt.imshow(B)
plt.title('Blue Channel Histogram')
plt.subplot(1,3,2)
plt.imshow(R)
plt.title('Red Channel Histogram')
plt.subplot(1,3,3)
plt.imshow(G)
plt.title('Green Channel Histogram')
plt.savefig('Individual Color channels')


# __Here on considering Blue and Green channels for the object segmentation__

# # Testing on individual channels

# # Blue channel

# In[1182]:


Threshold_values=range(0,256)#setting the threshold value range as mentioned hovering over entire pixel range
False_alarm_rate_list=[]
Detection_rate_list=[]
#Initialising empty lists to store the values of each and every False rate and Detection rate.
start_time=time.time()
for threshold_value in Threshold_values:
    val,Threshold_image=cv2.threshold(B,threshold_value,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    #After the thresholding(binary) image will be assigned either with 0 or 255
    Threshold_image_binary=Threshold_image/255
    #Converting the image into binary image for further analysis
    
    #Metrics calculation
    #Calculating True Positives
    True_positives=np.sum(np.logical_and(Threshold_image_binary==1,Ground_truth_image==1))
    #Calculating True Negatives
    True_Negatives=np.sum(np.logical_and(Threshold_image_binary==0,Ground_truth_image==0))
    #Calculating False Postives
    False_Positives=np.sum(np.logical_and(Threshold_image_binary==1,Ground_truth_image==0))
    #Calculating False Negatives
    False_Negatives=np.sum(np.logical_and(Threshold_image_binary==0,Ground_truth_image==1))
    
    #Calculating False Alarm Rate 

    FAR=False_Positives/(False_Positives+True_Negatives)
    
    #Calculating Detection rate 
    
    TPR=True_positives/(True_positives+False_Negatives)
    
    False_alarm_rate_list.append(FAR)
    Detection_rate_list.append(TPR)
end_time=time.time()


# In[1183]:


plt.figure(figsize=(8, 8))
plt.plot(False_alarm_rate_list,Detection_rate_list,color='blue',lw=5,label='One threshold curve')
plt.xlabel('False Alarm Rate (FAR)')
plt.ylabel('Detection Rate (DR)')
plt.title('ROC Curve for Blue channel')
plt.legend(loc='lower right')
plt.savefig('ROC curve for Blue channel threshold detector')


# # Green channel

# In[1184]:


Threshold_values=range(0,256)#setting the threshold value range as mentioned hovering over entire pixel range
False_alarm_rate_list=[]
Detection_rate_list=[]
#Initialising empty lists to store the values of each and every False rate and Detection rate.
start_time=time.time()
for threshold_value in Threshold_values:
    val,Threshold_image=cv2.threshold(G,threshold_value,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    #After the thresholding(binary) image will be assigned either with 0 or 255
    Threshold_image_binary=Threshold_image/255
    #Converting the image into binary image for further analysis
    
    #Metrics calculation
    #Calculating True Positives
    True_positives=np.sum(np.logical_and(Threshold_image_binary==1,Ground_truth_image==1))
    #Calculating True Negatives
    True_Negatives=np.sum(np.logical_and(Threshold_image_binary==0,Ground_truth_image==0))
    #Calculating False Postives
    False_Positives=np.sum(np.logical_and(Threshold_image_binary==1,Ground_truth_image==0))
    #Calculating False Negatives
    False_Negatives=np.sum(np.logical_and(Threshold_image_binary==0,Ground_truth_image==1))
    
    #Calculating False Alarm Rate 

    FAR=False_Positives/(False_Positives+True_Negatives)
    
    #Calculating Detection rate 
    
    TPR=True_positives/(True_positives+False_Negatives)
    
    False_alarm_rate_list.append(FAR)
    Detection_rate_list.append(TPR)
end_time=time.time()


# In[1185]:


plt.figure(figsize=(8, 8))
plt.plot(False_alarm_rate_list,Detection_rate_list,color='green',lw=5,label='One threshold curve')
plt.xlabel('False Alarm Rate (FAR)')
plt.ylabel('Detection Rate (DR)')
plt.title('ROC Curve for Green channel')
plt.legend(loc='lower right')
plt.savefig('ROC curve for Green channel threshold detector')


# # Red channel

# In[1186]:


Threshold_values=range(0,256)#setting the threshold value range as mentioned hovering over entire pixel range
False_alarm_rate_list=[]
Detection_rate_list=[]
#Initialising empty lists to store the values of each and every False rate and Detection rate.
start_time=time.time()
for threshold_value in Threshold_values:
    val,Threshold_image=cv2.threshold(R,threshold_value,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    #After the thresholding(binary) image will be assigned either with 0 or 255
    Threshold_image_binary=Threshold_image/255
    #Converting the image into binary image for further analysis
    
    #Metrics calculation
    #Calculating True Positives
    True_positives=np.sum(np.logical_and(Threshold_image_binary==1,Ground_truth_image==1))
    #Calculating True Negatives
    True_Negatives=np.sum(np.logical_and(Threshold_image_binary==0,Ground_truth_image==0))
    #Calculating False Postives
    False_Positives=np.sum(np.logical_and(Threshold_image_binary==1,Ground_truth_image==0))
    #Calculating False Negatives
    False_Negatives=np.sum(np.logical_and(Threshold_image_binary==0,Ground_truth_image==1))
    
    #Calculating False Alarm Rate 

    FAR=False_Positives/(False_Positives+True_Negatives)
    
    #Calculating Detection rate 
    
    TPR=True_positives/(True_positives+False_Negatives)
    
    False_alarm_rate_list.append(FAR)
    Detection_rate_list.append(TPR)
end_time=time.time()


# In[1187]:


plt.figure(figsize=(8, 8))
plt.plot(False_alarm_rate_list,Detection_rate_list,color='red',lw=5,label='One threshold curve')
plt.xlabel('False Alarm Rate (FAR)')
plt.ylabel('Detection Rate (DR)')
plt.title('ROC Curve for Red channel')
plt.legend(loc='lower right')
plt.savefig('ROC curve for RED channelthreshold detector')


# # Merging two channels and applying thresholding

# __Considering working on two channels.Here on considering the combination of RED and GREEN channels__

# __Here instead of hovering over the entire range which is a huge and time consuming process as the results would be of 256*256 considering a subset from the range i.e arround 70-160 considering each threshold values of individual channel which are estimated from histogram analysis.By doing so we are reducing the computational complexity by 50 percent__

# In[1527]:


#Setting the loop to iterate and get all the possible metrics values of all possible combinations
RGB_FAR_Values=[]
RGB_DETECTION_values=[]
i_lst=[]
RGB_threshold_values=range(70,160)#Setting the threshold value range
RGB_start_time=time.time()
for blue_threshold_values in RGB_threshold_values:#loop to iterate for Blue chanel image
    val1,Blue_Thresholded_Image=cv2.threshold(B,blue_threshold_values,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    #performing threshold on blue channel image
    for green_threshold_value in RGB_threshold_values:#loop to iterate for Green chanel image
        val2,Green_Thresholded_Image=cv2.threshold(G,green_threshold_value,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        
        #Combining both the thresholded images
        Final_Combination_Image=np.logical_and(Blue_Thresholded_Image,Green_Thresholded_Image)
    
        
        #Final_Combination_Binary_Image=Final_Combination_Image/255 
        Final_Combination_Image[Final_Combination_Image==255]=1

        FAR_RGB,TPR_RGB,th=roc_curve(Ground_truth_image.ravel(),Final_Combination_Image.ravel())
       

        RGB_FAR_Values.append(FAR_RGB[1])
        RGB_DETECTION_values.append(TPR_RGB[1])
RGB_end_time=time.time()      


# In[1528]:


RGB_DETECTION_values


# In[1529]:


RGB_FAR_Values


# In[1530]:


plt.figure(figsize=(8, 8))
#plt.plot(Detection_rate_list,False_alarm_rate_list,color='darkorange',lw=1,label='One threshold curve')
plt.plot(RGB_FAR_Values,RGB_DETECTION_values,color='Green',lw=1,label='two threshold curve',marker='o')
plt.xlabel('False Alarm Rate (FAR)')
plt.ylabel('Detection Rate (DR)')
plt.title('ROC Curve for thresholding over the range(70-160)')
plt.legend(loc='lower right')
plt.savefig('ROC Curve for thresholding over the range(70-160)')


# In[1512]:


RGB_elapsed_time=RGB_end_time-RGB_start_time
print(f'total time taken :{RGB_elapsed_time/60} minutes')


# __As we can see the time taken for computing the all possible combinations it is arround 1.34 minutes which can be considered as a bit longer one when compared to the firss single threshold detector process.But if the output threshold value of the combination channels gives a better results during object segmentation then this tradeoff can be ignored.So now proceeding to find the better combination of channel threshold values__

# In[1531]:


# Reshaping the lists to a 2D array for each threshold combination
RGB_FAR_Values = np.array(RGB_FAR_Values).reshape(len(RGB_threshold_values), len(RGB_threshold_values))
RGB_DETECTION_values = np.array(RGB_DETECTION_values).reshape(len(RGB_threshold_values), len(RGB_threshold_values))
#best TPR, FPR, and threshold values for each channel
best_tpr_red, best_fpr_red, best_threshold_red = 0, 0, 0
best_tpr_green, best_fpr_green, best_threshold_green = 0, 0, 0
# Iterating over all threshold combinations to find the best pair for each channel
for i in range(len(RGB_threshold_values)):
    # For Red Channel
    tpr_red = RGB_DETECTION_values[i, :]
    fpr_red = RGB_FAR_Values[i, :]
    best_idx_red = np.argmax(tpr_red - fpr_red)
    #Here nothing but we ar following the same Youden J method of calculating best combination of tpr and fpr
    best_tpr_red = max(best_tpr_red, tpr_red[best_idx_red])
    best_fpr_red = max(best_fpr_red, fpr_red[best_idx_red])
    best_threshold_red = RGB_threshold_values[best_idx_red]
    # For Green Channel
    tpr_green = RGB_DETECTION_values[:, i]
    fpr_green = RGB_FAR_Values[:, i]
    best_idx_green = np.argmax(tpr_green - fpr_green)
    best_tpr_green = max(best_tpr_green, tpr_green[best_idx_green])
    best_fpr_green = max(best_fpr_green, fpr_green[best_idx_green])
    best_threshold_green = RGB_threshold_values[best_idx_green]
print(f'Best TPR for Red Channel: {best_tpr_red}, Best FPR for Red Channel: {best_fpr_red}, Best Threshold for Red Channel: {best_threshold_red}')
print(f'Best TPR for Green Channel: {best_tpr_green}, Best FPR for Green Channel: {best_fpr_green}, Best Threshold for Green Channel: {best_threshold_green}')


# __The best tpr for red and green channel are 0.8371 abd 0.9520 which are considerable even though we could'nt achieve the ideal value of 1.The fpr both the channel is a bit high but the resulted combination is giving better results than the one threshold pixel detector__

# __So trying to form a new combination of image using two seperated threshold values of each individual channel__

# In[1538]:


Two_threshold_detector_result=np.logical_and(B>93,G>79)


# In[1542]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(Two_threshold_detector_result,cmap='gray')
plt.title('Two_threshold_detector_result')
plt.subplot(1,2,2)
plt.imshow(one_Threshold_detector_result,cmap='gray')
plt.title('one_Threshold_detector_result')
plt.savefig('comparison')


# __Outcome discussion__

# __By looking at the final output images it is very clear that two threshold detector is performing better in detecting the objects i.e baloons.Where the unwanted detection i.e background pixels as object pixels is nearly same among the two.So ignoring the time taken for compilation as we are having enough time and resources we can prefer two threshold detector as the best option for performing further operation on a test data__

# # Working with the best combination threshold values 

# # 1.3

# __Simple idea behind counting thenumber of baloons is counting the number of 1's in the image after applying thresholding and normalizing the image.More the number of one's vary the count.Initially setting the valuesfor the count of 1's  for different number of baloons and counting accordingly__

# In[1516]:


# Assuming you have the best_threshold_red and best_threshold_green values
#Reading the images from the database
database_folder = "/Users/raviirt/Downloads/Images"
output_folder="/Users/raviirt/Downloads/newimages"
listdir = os.listdir(database_folder)
listdir.sort()
final_label=[]
#Sorting it inorder to avoid any confusion
#fixing pixel count for different numbers of balloons
#creating a dictionary to check for conditions
thresholds = {
    "no_balloon": 1000,
    "one_balloon": 10000,
    "two_balloons": 11000
}
#Applying thresholds to all the images in the folder
for img_name in listdir:
    if img_name.endswith(".jpeg"):
        #Reading the image
        img_path = os.path.join(database_folder, img_name)
        img = cv2.imread(img_path)
        # Apply best thresholds to Red and Green channels
        vred,red_channel = cv2.threshold(img[:, :, 2], best_threshold_red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        vgreen,green_channel = cv2.threshold(img[:, :, 1], best_threshold_green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        #Combining thresholded images
        combined_threshold = np.logical_and(red_channel, green_channel)
        combined_threshold[combined_threshold == 255] = 1#Converting the thresholded image into binary image
        #Counting baloons based on the number of ones
        num_balloons = np.sum(combined_threshold)
        #Allocating the count a category with the condition block
        if num_balloons <= thresholds["no_balloon"]:
            num_detected_balloons = 0
        elif num_balloons <= thresholds["one_balloon"]:
            num_detected_balloons = 1
        elif num_balloons <= thresholds["two_balloons"]:
            num_detected_balloons = 2
        else:
            num_detected_balloons = 3
        final_label.append(num_detected_balloons)
        # Print or store the results
        print(f"For Image: {img_name}, Number of Balloons detected: {num_detected_balloons}")
        #Providing the folder path to save the new image information
        output_path=os.path.join(output_folder,f'result {img_name}')
        cv2.imwrite(output_path, combined_threshold * 255)


# # Displaying  the thresholded images

# In[1550]:


folder = '/Users/raviirt/Downloads/newimages'
listdir = os.listdir(folder)
listdir.sort()
K=0
for img_res in listdir:
    img_path = os.path.join(folder, img_res)
    img = cv2.imread(img_path)
    if img is not None:
        plt.figure(figsize=(12,5))
        plt.subplot(5,11,K+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(img_res)
        plt.show()
        plt.tight_layout()
    else:
        print(f"Failed to read image: {img_path}")


# In[1518]:


true_label = np.array([0,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,2,1,2,2,2,2,2,1,1,1,2,2,2,1,1,2,2,2,3,3,3,2,2])


# In[1519]:


Final_outcome=np.array(final_label)


# In[1520]:


true_label


# In[1521]:


Final_outcome


# In[1522]:


accuracy_score(true_label,Final_outcome)


# __Conclusion : On varying the pixel count of ones we can achieve better accuracy__
