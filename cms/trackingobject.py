import cv2 as cv
from django.conf import settings
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from django.conf import settings
# import simplejpeg
from numpy.__config__ import show
from numpy.core.defchararray import index
from numpy import *
# from imutils.video import VideoStream

def swimmingPool(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    blueLow = np.array([80,43,46])
    blueHigh = np.array([100,255,255])
    mask = cv.inRange(hsv, blueLow, blueHigh)
    mask=mask/255
    # maskColor = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # resultImg = cv2.bitwise_and(img, maskColor)
    return mask

colors = [[0,0,255],[255,0,0],[0,255,0],[255,255,0],[255,0,255],[0,255,255],[255,255,255],[0,0,122],[122,0,0],[0,122,0],[122,122,0],[122,0,122],[0,122,122],[122,122,122],[0,0,67],[67,0,0],[0,67,0],[67,67,0],[67,0,67],[0,67,67],[67,67,67]]

#LAB diff
def LABColourDistance(rgb_1, rgb_2):
     R_1,G_1,B_1 = rgb_1[:,:,0], rgb_1[:,:,1], rgb_1[:,:,2]
     R_2,G_2,B_2 = rgb_2[:,:,0], rgb_2[:,:,1], rgb_2[:,:,2]
     rmean = (R_1 +R_2 ) / 2
     R = R_1 - R_2
     G = G_1 -G_2
     B = B_1 - B_2
     return np.sqrt((2+rmean/256)*(R**2)+4*(G**2)+(2+(255-rmean)/256)*(B**2))
def distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

#去除面积小的区域
def Remove_small_block(label, label_pre, block_num):
    label = np.array(label).astype(np.uint8)
    
    # print(label.shape)
    # print(label_pre.shape)
    new_label = np.zeros_like(label_pre).astype(np.uint8)
    label = label[:, :, np.newaxis]
    h,w, _ = label.shape
    #print(label.shape)
    #ret, label = cv.threshold(label,127,255,cv.THRESH_BINARY)
    
    threshold = 500
    contours, _ = cv.findContours(label,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    n=len(contours)  #轮廓的个数
    visual_image = np.zeros([h,w,3],np.uint8)
    swimmer_list = []
    for i in range(len(contours)):
        #print(contours[i])
        area = cv.contourArea(contours[i]) #计算轮廓所占面积
        M = cv.moments(contours[i],False)
        if M["m00"]!=0:
            center = (int(M["m01"]/M["m00"]), int(M["m10"]/M["m00"])) #get the center of every segmentation block
        else:
            # continue
            center = (0,0)
        # print(center)
        #print(contours[i])
        min_rect = cv.minAreaRect(contours[i])
        rect_points = cv.boxPoints(min_rect)
        h1 = distance(rect_points[0],rect_points[1])
        h2 = distance(rect_points[1],rect_points[2])
        rate = max(h1,h2)*1.0/(min(h1,h2)+1e-6)
        if area < threshold or rate>5:                         #将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
            cv.drawContours(label,[contours[i]],-1, 0, thickness=-1)     #原始图片背景BGR值(84,1,68)
            continue
        mark = label_pre[center]
        
        if mark[0]==0 and mark[1]==0 and mark[2]==0:
            block_num+=1
            #print(colors[block_num%18])
            swimmer_list.append(contours[i])
            new_label = cv.drawContours(new_label,contours,i, colors[block_num%18], thickness=-1)
            visual_image = cv.drawContours(visual_image,contours,i, colors[block_num%18], thickness=-1)
            
        else:
            #print(label_pre[center])
            mark = ( int(mark[0]), int(mark[1]), int(mark[2])) 
            swimmer_list.append(contours[i])
            new_label = cv.drawContours(new_label,contours,i, tuple(mark), thickness=-1)
            visual_image = cv.drawContours(visual_image,contours,i, tuple(mark), thickness=-1)  
            
    return label, new_label, visual_image, block_num, swimmer_list #"new_label" means tracking segmentation ID

# print("file exists?", os.path.exists("E47_H2_BS_FINISH.mp4"))


'''  
def rgb2hsv(r, g, b,rgb):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = np.max(rgb, axis = 2)
    mn = np.min(rgb, axis = 2)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx
    return h, s, v
'''
def HSVDistance(rgb_1,rgb_2):
    HSV1 = cv.cvtColor(rgb_1, cv.COLOR_BGR2HSV)
    H_1 = HSV1[:,:,0]*2
    S_1 = HSV1[:,:,1]/255.0
    V_1 = HSV1[:,:,0]/255.0

    HSV2 = cv.cvtColor(rgb_2, cv.COLOR_BGR2HSV)
    H_2 = HSV2[:,:,0]*2 #0～360
    S_2 = HSV2[:,:,1]/255.0 #0～1
    V_2 = HSV2[:,:,0]/255.0 #0～1

    #H_2,S_2,V_2 = cv.cvtColor(rgb_2, cv.COLOR_BGR2HSV)
    R=100
    angle=30
    mark = math.pi/180.0
    h = R * math.cos(angle * mark)
    r = R * math.sin(angle * mark)
    
    x1 = r * V_1 * S_1 * np.cos(H_1 * mark)
    y1 = r * V_1 * S_1 * np.sin(H_1 * mark)
    z1 = h * (1 - V_1)
    x2 = r * V_2 * S_2 * np.cos(H_2 * mark)
    y2 = r * V_2 * S_2 * np.sin(H_2 * mark)
    z2 = h * (1 - V_2)
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return dx * dx + dy * dy + dz * dz

# 生成视频
def makevideo(image_input, image_output):
    videoinpath  = image_input #'E47_H2_BS.mp4'
    videooutpath = image_output #'E47_H2_BS_out.mp4'
    capture     = cv.VideoCapture(videoinpath  )
    fourcc      = cv.VideoWriter_fourcc(*'mp4v')
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    print((width,height))
    writer = cv.VideoWriter(videooutpath ,fourcc, 20.0, (width,height), True)
    if capture.isOpened():
        while True:
            ret,img_src=capture.read()
            if not ret:break
#             print(img_src.shape)
#             img_src = cv2.flip(img_src, 0) # 反转图像 1水平翻转 0垂直翻转 -1水平垂直翻转
            img_out = swimmingPool(img_src)
            
            writer.write(img_out)
    else:
        print('视频打开失败！')
    return writer
    writer.release()

def cannyCon(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    blueLow = np.array([80,43,46])
    blueHigh = np.array([100,255,255])
    mask = cv.inRange(hsv, blueLow, blueHigh)
#    plt.imshow(mask)
#    print(mask.shape)
    canny = cv.Canny(mask, 30, 220)
    canny = canny.astype(np.uint8)
#     print(canny.shape)
#     plt.imshow(canny)
    return canny
###################################################################
# make background
def makebackground():
    STORAGE_PATH = os.path.join(settings.BASE_DIR, 'media')
    cap = cv.VideoCapture(STORAGE_PATH+"/video_file.mp4")
    ret, frame_pre = cap.read()

    back_ground = np.zeros_like(frame_pre)
    back_ground = back_ground[np.newaxis, :]
    time_now = 0
    while (cap.isOpened()):
        time_now += 1
        ret, frame = cap.read()
        if time_now%5 == 0:
            back_ground = np.concatenate((back_ground, frame[np.newaxis,:]), axis = 0)
        k = cv.waitKey(20)
        if k & 0xff==ord('q'):
            break
        if frame is None:
            break

    back_ground = np.median(back_ground, axis=0).astype(np.uint8)
    # cv.imrite('back_media.jpg',back_ground)
    cap.release()
    cv.destroyAllWindows()
    return back_ground
#################################################################
def parseVideo():
    STORAGE_PATH = os.path.join(settings.BASE_DIR, 'media')
    cap = cv.VideoCapture(STORAGE_PATH+"/video_file.mp4")
    # fourcc = cv.VideoWriter_fourcc(*'MJPG')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # out = cv.VideoWriter('output.avi', fourcc, 20.0, (width, height))

    while (cap.isOpened()):
        ret, frame = cap.read()
    #   if ret == True:
    #     frame = cv2.flip(frame, 0)
        frame = cannyCon(frame)
        # out.write(frame)
    #     cv2.imshow('frame', frame)
        cv.imwrite(STORAGE_PATH+'/contour.jpg',frame)
        break

    cap.release()
    # out.release()
    cv.destroyAllWindows()

    img = cv.imread(STORAGE_PATH+'/contour.jpg')

    color = (255,255,255)
    indices = np.where(img == color)
    coordinates = zip(indices[0], indices[1])
    unique_coordinates = list(set(list(coordinates)))
    xy_coor = []
    for i in range(len(unique_coordinates)):
        temp0 = list(unique_coordinates[i])
        temp_y = temp0[0]
        temp_x = temp0[1]
        temp_list = [temp_x, temp_y]
        xy_coor.append(temp_list)
    ###终点线
    finish_line = []
    for i in range(1920):
        max_value = 0
        for j in range(len(xy_coor)):
            if xy_coor[j][0] == i:
                max_value = max(max_value, xy_coor[j][1])
        if max_value > 0:
            finish_line.append([i,max_value])

    #一共20个数，泳道1是lanes[0]-lanes[1]之间，以此类推
    lanes = []
    for i in range(1920):
        counter = 0
        for j in range(len(xy_coor)):
            if xy_coor[j][0] == i:
                counter = counter + 1
        lanes.append(counter)

    x_coor_lanes = []
    for i in range(len(lanes)):
        if lanes[i] > 10:
            x_coor_lanes.append(i)

    gaps = x_coor_lanes[len(x_coor_lanes)-1] - x_coor_lanes[0]
    interval = gaps / 10

    final_lanes = []
    left = x_coor_lanes[0]
    for i in range(10):
        right = left + interval
        final_lanes.append([int(left),int(right)])
        left = right


    validate = []
    for i in range(len(finish_line)):
        validate.append(finish_line[i][1])
    average = mean(validate)
    final_finish_line = []
    for i in range(len(finish_line)):
        if finish_line[i][1] > average:
            final_finish_line.append(finish_line[i])
    return {'final_lanes':final_lanes, 'final_finish_line': final_finish_line}
############################################################
class Tracker(object):
    STORAGE_PATH = os.path.join(settings.BASE_DIR, 'media')

    def __init__(self):
        self.video = cv.VideoCapture(self.STORAGE_PATH+'/video_file.mp4')
        f=open(self.STORAGE_PATH+"/response.txt", "w+")
        f.write('Processing ....\n\n\n<br><br>')
        f.close()

    def __del__(self):
        self.video.release()

    def get_frame(self):

        STORAGE_PATH = os.path.join(settings.BASE_DIR, 'media')

        # cap = self.video
        back_ground=makebackground()
        result = [0,0,0,0,0,0,0,0,0,0]
        result_index = 0
        ###frame rate could be changed
        # frame_rate = 25
        time_now = 0
        w = int(self.video.get(3))
        h = int(self.video.get(4))
        frame_rate= int(self.video.get(5))
        interval = 1 / frame_rate
        frame_pre0 = back_ground

        #generate video visual result
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_video = cv.VideoWriter(STORAGE_PATH+'/tracking.mp4', fourcc, frame_rate, (int(w/2), int(h/4)))

        ret, frame_pre = self.video.read()
        mask_img = back_ground.copy()
        # back_ground = np.zeros_like(frame_pre)
        #print (back_ground.shape)
        # back_ground = back_ground[np.newaxis, :]

        # mask_img = cv.imread('back_media.jpg')
        mask_pool= swimmingPool(mask_img)
        mask_pool = np.array(mask_pool)

        pre_label = np.zeros_like(frame_pre).astype(np.uint8)
        block_num = 0
        lanedata=parseVideo()
        final_lanes=lanedata['final_lanes']
        final_finish_line=lanedata['final_finish_line']

        while (self.video.isOpened()):
            time_now += 1
            ret, frame = self.video.read()
            print(self.video.get(cv.CAP_PROP_POS_FRAMES))
            print(self.video.get(cv.CAP_PROP_FPS))
            # if time_now%5 == 0:
            #     back_ground = np.concatenate((back_ground, frame[np.newaxis,:]), axis = 0)
            k = cv.waitKey(20)
            if k & 0xff==ord('q'):
                break
            if frame is None:
                break
            '''
            show_image = abs(frame-frame_pre0)
            show_image = show_image*show_image
            show_image = np.sqrt(np.sum(show_image,axis=2))
            show_image[show_image<15]=0
            show_image[show_image>=15]=255
            '''
            diff = HSVDistance(frame, frame_pre0)
            diff[diff<9]=0
            diff[diff>=9]=255
            diff = diff*mask_pool
            _, pre_label, tracking_visual_iamge,block_num, swimmers = Remove_small_block(diff, pre_label,block_num)

            '''
            #print(diff)
            diff = LABColourDistance(frame, frame_pre0)
            diff[diff<30]=0
            diff[diff>=30]=25
            '''
            #print(diff)
            #calculate color diff part1 LAB
            if len(swimmers) != 0:
                swimmer_nearest_x = []
                for i in range(len(swimmers)):
                    temp = []
                    for j in range(len(swimmers[i])):
                        temp.append(swimmers[i][j][0][1])
                    maxY_index = temp.index(max(temp))
                    swimmer_nearest_x.append(swimmers[i][maxY_index][0])

                for i in range(len(swimmer_nearest_x)):
                    ### 用于继续循环
                    flag = True
                
                    for g in range(len(final_lanes)):
                        if swimmer_nearest_x[i][0] > (final_lanes[g][0])  and swimmer_nearest_x[i][0] < (final_lanes[g][1])  and result[g] != 0:
                            flag = False
                        elif swimmer_nearest_x[i][0] > (final_lanes[g][0])  and swimmer_nearest_x[i][0] < (final_lanes[g][1]) and result[g] == 0:
                            #if g == 1:
                                #print(swimmer_nearest_x[i])
                            result_index = g
                            for j in range(len(final_finish_line)):      
                                # if result_index == 1:
                                #         print(swimmer_nearest_x[i][1], (final_finish_line[j][1] - 5))
                                if result_index != None:   
                                    if swimmer_nearest_x[i][0] == final_finish_line[j][0] and swimmer_nearest_x[i][1] >= (final_finish_line[j][1] - 6) and result[result_index] == 0:    
                                        result[result_index] = time_now
                                        f=open(STORAGE_PATH+"/response.txt", "a+")
                                        f.write("Lane " + str(result_index + 1) + " finish at frame: " + str(time_now) + ", which is " + str(round(time_now * interval, 2)) + " second in the video <br> ")                   
                                        f.close()
                    if not flag:
                        continue
                            
                


            #mask = show_image[:,:,0]+show_image[:,:,1]+show_image[:,:,2]
            #mask[mask<150]=0
            #mask[mask>150]=255
            video_result = np.concatenate((frame,tracking_visual_iamge),axis = 1).astype(np.uint8) #put original image and tracking image together.
            video_result = cv.resize(video_result, (int(w/2),int(h/4)))
            # cv.imshow('image', video_result)
            ret, jpeg = cv.imencode('.jpg', video_result)
            f=jpeg.tobytes()
            out_video.write(video_result)
            frame_pre = frame
            yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n'+ f  + b'\r\n\r\n')
        f=open(STORAGE_PATH+"/response.txt", "a+")
        f.write('<br>Final Results: <br>')
        f.write(str(result))
        f.close()
        print(result)
        out_video.release()
        cv.destroyAllWindows()
