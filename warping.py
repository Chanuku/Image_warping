import torch
import torchvision
import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import argparse
from skimage.transform import resize


# Load the trained model 
import model as myModel

ddir = '../Downloads/DeepLabv3-TS/test_files/sg_data/'
root = '../Downloads/DeepLabv3-TS/masked/'
exp_dir = '../Downloads/DeepLabv3-TS/exp_out/'
checkpoint = torch.load(exp_dir + 'trained_weights_all_ep30.pth.tar')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-c", "--coords",
    help = "comma seperated list of source points")
args = vars(ap.parse_args())

def convex(dir):
    img = cv2.imread(dir)
    img1 = img.copy()
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imgray,255,0,180)
    contours,_ = cv2.findContours(thr, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
    biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]


    #cnt = contours[0]
    cnt = biggest_contour

    x,y,w,h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img,[box],0,(0,255,0),1)


    return box


def tl(x,y,w,h):
    x = x**2
    y = y**2
    result = (x+y) **0.5
    return result

def bl(x,y,w,h):
    y = h-y
    x = x**2
    y = y**2
    result = (x+y) **0.5
    return result

def tr(x,y,w,h):
    x = w-x
    x = x**2
    y = y**2
    result = (x+y) **0.5
    return result

def br(x,y,w,h):
    x= w-x
    y= h-y
    x = x**2
    y = y**2
    result = (x+y) **0.5
    return result

def tr_re(x,y,bl_x,bl_y):
    x= bl_x-x
    y= bl_y-y
    x = x**2
    y = y**2
    result = (x+y) **0.5
    return result


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
    	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect


	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped


preprocess = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
model = myModel.createDeepLabv3(outputchannels=2) # give nclasses
#model.load_state_dict(torch.load(exp_dir + 'trained_weights.best_acc.pth.tar'))

#checkpoint = torch.load(exp_dir + 'trained_weights_burn_imgs_550size.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])

#model = torch.load(exp_dir + 'weights.pt')
# Set the model to evaluate mode
model.eval()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')



# Read  a sample image and mask from the data-set
per_point = [0 for j in range(8)]
flist = os.listdir(ddir)
index_count = [[0 for j in range(2)] for i in range(400000)]
length_cal = [0 for j in range(4)]

for file in flist:
    imfile = ddir + file
    print ('filename: ', file, ' in ', ddir)
    im = np.ascontiguousarray(cv2.imread(imfile)[:,:,::-1])
    print (im.shape)
    #resizing
    re_size = 400
    rows, cols, channels = im.shape
    scale = re_size / float(im.shape[1])
    im = cv2.resize(im,(re_size,int(rows*scale)))
    #end

    print(im.shape)
    intensor = preprocess (im)
    batch = intensor.reshape(1,*intensor.shape)
    print (batch.shape, batch.dtype)

    with torch.no_grad():
        a = model(batch.to(device))
        re = a['out'].argmax(dim=1).data.cpu().numpy()[0]
        #re = cv2.resize(re,(cols,rows))
        #plt.imsave(root+'low/'+file,re)

        #re = resize(re, (rows, cols))

        plt.imsave(root+file,re)

imgs = list(sorted(os.listdir(os.path.join(root))))
#print(imgs)

for filename in imgs :

    print(root + filename)
    img = Image.open(root + filename).convert("RGB")
    (width,height) = img.size
    outputImg = Image.new("RGB", (width,height),(0,0,0))
    output = outputImg.load()

    
    k=0

    for y in range(0, height): 
        for x in range(0, width): 
            r, g, b = img.getpixel((x, y))
            if r > 140 :
                output[x,y] = (255,0,0)
            elif b > 140 :
                output[x,y] = (0,0,255)
            elif g > 140 :
                output[x,y] = (0,255,0)
            else :
                output[x,y] = (0,0,0)    


    outputImg.save(root + filename)

    rectangle = convex(root+filename)
    # box[0][1~2] br ~ bl ~ tl ~tr

    
    #up
    for y in range(0,rectangle[2][1]):
        for x in range(0,width):
            output[x,y] = (0,0,0)
    #down
    for y in range(rectangle[0][1],height):
        for x in range(0,width):
            output[x,y] = (0,0,0)
    #left
    for y in range(0,height):
        for x in range(0,rectangle[1][0]):
            output[x,y] = (0,0,0)
    #right
    for y in range(0,height):
        for x in range(rectangle[3][0],width):
            output[x,y] = (0,0,0)
    outputImg = outputImg.convert("RGB")
    outputImg.save(root + filename)


    for y in range(0, height): 
        for x in range(0, width): 
            r, g, b = outputImg.getpixel((x, y))
            if(r>50):
                index_count[k][0] = x
                index_count[k][1] = y
                k+=1
                #print(k)
        
    for i in range(0,len(length_cal)):
        length_cal[i] = 3000
    for i in range(1,k):
        if(tl(index_count[i][0],index_count[i][1],width,height)<length_cal[0]):
            per_point[0] = index_count[i][0]
            per_point[1] = index_count[i][1]
            length_cal[0] = tl(index_count[i][0],index_count[i][1],width,height) #tl

        if(bl(index_count[i][0],index_count[i][1],width,height)<length_cal[1]):
            per_point[2] = index_count[i][0]
            per_point[3] = index_count[i][1]
            length_cal[1] = bl(index_count[i][0],index_count[i][1],width,height) #bl

        if(tr(index_count[i][0],index_count[i][1],width,height)<length_cal[2]):
            per_point[4] = index_count[i][0]
            per_point[5] = index_count[i][1]
            length_cal[2] = tr(index_count[i][0],index_count[i][1],width,height) #tr

        if(br(index_count[i][0],index_count[i][1],width,height)<length_cal[3]):
            per_point[6] = index_count[i][0]
            per_point[7] = index_count[i][1]
            length_cal[3] = br(index_count[i][0],index_count[i][1],width,height) #br

    #rotated 
    if(per_point[7]-per_point[3]>abs(per_point[3]-per_point[5])/10):
        for i in range(1,k):
            if(index_count[i][1]<index_count[i-1][1]):
                per_point[0] = index_count[i][0]
                per_point[1] = index_count[i][1]
                #print("distort changed!")
        #for i in range(1,k):      
            if(index_count[i][0]<index_count[i-1][0]):
            #if(tr_re(index_count[i][0],index_count[i][1],per_point[2],per_point[3])>tr_re(index_count[i-1][0],index_count[i-1][1],per_point[2],per_point[3])):
                per_point[4] = index_count[i][0]
                #per_point[5] = index_count[i][1]
            #length_cal[0] = tl(index_count[i][0],index_count[i][1],width,height) #tl
        #per_point[4] = per_point[6]
    print(per_point)

    expend_pixel = 1
    #tl
    per_point[0] -= expend_pixel #x
    per_point[1] -= expend_pixel #y

    #bl
    per_point[2] -= expend_pixel #x
    per_point[3] += expend_pixel #y

    #tr
    per_point[4] += expend_pixel #x
    per_point[5] -= expend_pixel #y

    #br
    per_point[6] += expend_pixel #x
    per_point[7] += expend_pixel #y

    args["image"] = ddir+filename

    args["coords"] =  "[("+str(per_point[0])+","+str(per_point[1])+"), ("+str(per_point[2])+","+str(per_point[3])+"), ("+str(per_point[4])+", "+str(per_point[5])+"), ("+str(per_point[6])+", "+str(per_point[7])+")]"
    #determine the coordinates without pre-supplying them tl, tr, br, bl
    image = cv2.imread(args["image"])
    
    rows, cols, channels = image.shape
    scale = re_size / float(image.shape[1])
    image = cv2.resize(image,(re_size,int(rows*scale)))
    
    pts = np.array(eval(args["coords"]), dtype = "float32")
        
    warped = four_point_transform(image, pts)
    #cv2.imshow("warped",warped)
    #cv2.waitKey(0)

    cv2.imwrite("../Downloads/DeepLabv3-TS/"+"warped"+filename,warped)
    #cv2.waitKey(0)
    print("warp_saved")

