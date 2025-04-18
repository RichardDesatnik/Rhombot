import cv2
import numpy as np
import robotpy_apriltag
import imutils
from pathlib import Path
import pandas as pd

#mass in kg
mass = 0.100
#time in min
time_test = 1.2
font = cv2.FONT_HERSHEY_DUPLEX
COT_final = 2000
# fontScale
fontScale = 5
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
# Using cv2.putText() method
def pix_dist(cx_s, cy_s, cx_f, cy_f):
    return (((cx_s-cx_f)**2) + ((cy_s-cy_f)**2))**0.5

cx5=0
cx7=0
cx10=0
cx11=0

cm_5_10 = 39.37
cm_10_11 = 80
cm_11_7 = 80
cm_5_7 = 39.37

cm_per_pixel = 80/3500

speed_thres = 1

def num_identify (image=None, image_c=None, y=0, h=0, x=0, w=0, grid=0):
    num = mirrored_image[y:(y+h), x:(x+w)]
    cont_num, hier_num = cv2.findContours(num,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    ig_num, size_num, array_num = hier_num.shape
    center_list = []
    center_x = []
    center_y = []
    for contour in range(size_num):
        moments =  cv2.moments(cont_num[contour])
        area_num = cv2.contourArea(cont_num[contour])
        #print(area_num)
        if area_num < 1000 and area_num > 10:
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            center_list.append([cx,cy])
            center_x.append(cx)
            center_y.append(cy)
            #print("Grid: " + str(grid) + " Center cx: " + str(cx) + " Center cy: " + str(cy) + " Area: " + str(area_num))
            #cv2.circle(mirrored_image_c, center=center,radius=1,color=(0,255,255),thickness=5)
        #print(size_num)
    if len(center_list) == 2:
        ans = 1
        #print("answer is 1")
    elif len(center_list) == 3:
        ans = 7
        #print("answer is 7")
    elif len(center_list) == 4:
        ans = 4
        #print("answer is 4")
    elif len(center_list) == 5:
        num_2 = []
        num_3 = []
        num_5 = []
        for middle in center_list:
            if middle[0] < 10 and middle[1] > 35:
                num_2.append(2)
            else:
                num_2.append(0)
            if middle[0] > 20 and middle[1] < 24:
                num_3.append(3)
            else:
                num_3.append(0)
        
        if 2 in num_2:
            #print("answer is 2")
            ans = 2
        else:
            if 3 in num_3:
                #print("answer is 3")
                ans = 3
            else:
                #print("answer is 5")
                ans = 5
        
    elif len(center_list) == 6:
        num_0 = []
        num_9 = []
        num_6 = []
        for middle in center_list:
            if middle[0] > 10 and middle[0] < 20 and middle[1] > 20 and middle[1] < 35:
                num_0.append(1)
            else:
                num_0.append(0)
            if middle[0] > 20 and middle[1] < 24:
                num_9.append(9)
                #print("CX: "+str(middle[0])+"CY: "+str(middle[1]))
            else:
                num_9.append(0)
                #print("CX: "+str(middle[0])+"CY: "+str(middle[1]))
            if middle[0] < 10 and middle[1] > 35:
                num_6.append(6)
                #print("CX: "+str(middle[0])+"CY: "+str(middle[1]))
            else:
                num_6.append(0)
                #print("CX: "+str(middle[0])+"CY: "+str(middle[1]))
        if 1 in num_0:
            if 9 in num_9:
                #print("value is 9")
                ans = 9
            else:
                if 6 in num_6:
                    #print("value is 6")
                    ans = 6
                else:
                    #print("error")
                    ans = "undefined"
        else:
            #print("value is 0")
            ans = 0

    elif len(center_list) == 7:
        ans = 8
        #print("answer is 8")
    else:
        ans = "not a number"
    return ans

#filename = Path("C:\\Users\\rdesa\\Desktop\\Terrain Paper\\Speed_COT\\Control\\Angle_0_R1\\Rhombot_0deg_Control_R1_Test1.mp4")
#video = cv2.VideoCapture("C:/Users/rdesa/Desktop/Terrain Paper/Speed_COT/Control/Angle_0_R1/Rhombot_0deg_Control_R1_Test3.mp4")
file = "Rhombot_30deg_R1_Control_Test7_short.mp4"
video = cv2.VideoCapture("C:/Users/rdesa/Desktop/Terrain Paper/Speed_COT/Control/Angle_30_R1/Rhombot_30deg_Control_R1_Short/Rhombot_30deg_R1_Control_Test7_short.mp4")





#C:\Users\rdesa\Desktop\Terrain Paper\Speed_COT\Control\Angle_30_R1\Rhombot_30deg_R1_Control_Test3
frame_rate = video.get(cv2.CAP_PROP_FPS)
print(frame_rate)


####################################################################################################
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec can be 'XVID', 'mp4v', etc.
out = cv2.VideoWriter('C:/Users/rdesa/Desktop/Terrain_Out.mp4', fourcc, frame_rate, (frame_width, frame_height))
####################################################################################################


#30 frames per second
cm_per_pixel = 1/20
seconds_per_frame = 1/30
#instant_speed = (cm_per_pixel*pixel_dist_per_frame)/(seconds_per_frame)
centerlist = []
i = 0
ii = 0
Rhom_dist = []
Rhom_dist_inst = []
Rhom_inst_speed = []
while True:
    
    ret, frame = video.read()
    #w,h,c = frame.shape
    #print("Width: " + str(w))
    #print("Heigh: " + str(h))
    #if w < h:
        #frame = np.transpose(frame, (1, 0, 2))

    #print(ret)
    if ret == True:


        frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #w,h = frame_g.shape
        #print("Width: " + str(w))
        #print("Heigh: " + str(h))
        frame_g = frame_g.astype('uint8')
        
        #Working on COT Videos
        #y = 1900
        #x = 800
        #h = 500
        #w = 500

        y = 2550
        x = 100

        h = 400
        w = 300
        
        
        frame_crop = frame_g[y:(y+h), x:(x+w)]
        frame_crop_c = frame[y:(y+h), x:(x+w)]
       
        detector = robotpy_apriltag.AprilTagDetector()
        detector.addFamily("tag36h11")
        #detector.addFamily("TagStandard41h12",2)
        detections = detector.detect(frame_g)
        
        for tag in detections:
            tagID = tag.getId()
            cx = int(tag.getCenter().x)
            cy = int(tag.getCenter().y)
            center = (cx,cy)
            #tagID -> 3 -> 45deg
            #tagID -> 2 -> 30deg
            #tagID -> 4 -> 15deg
            #tagID -> 0 -> 0deg
            if tagID == 2:
                Rhom_dist.append(center)
                Rhom_dist_inst.append((i,cx,cy))
                ii = ii + 1
                if ii > 2:
                    frame_s,cx_1,cy_1 = Rhom_dist_inst[ii-2]
                    #print("start frame")
                    #print(frame_s)
                    frame_f,cx_2,cy_2 = Rhom_dist_inst[ii-1]
                    #print("final frame")
                    #print(frame_f)
                    #print("what is ii")
                    #print(ii)
                    #print("length of Rhom_dist_inst")
                    #print(len(Rhom_dist_inst))
                    pixel_dist_per_frame = pix_dist(cx_s=cx_1, cy_s=cy_1, cx_f=cx_2, cy_f=cy_2)
                    time = (frame_f-frame_s)*(1/frame_rate)
                    instant_pix_speed = pixel_dist_per_frame/time
                    cm_per_s = instant_pix_speed*cm_per_pixel
                    cm_per_min = cm_per_s*60
                    #print("Time")
                    #print(time)
                    #print("Distance")
                    #print(pixel_dist_per_frame)
                    #print("Instant Speed")
                    #print(instant_speed)
                    if cm_per_s > speed_thres:
                        Rhom_inst_speed.append((cm_per_s,(cx_1,cy_1),(cx_2,cy_2))) 
                        #print(cm_per_s)
                    #print(pixel_dist_per_frame)
                    #instant_speed = (cm_per_pixel*pixel_dist_per_frame)/(seconds_per_frame)
                if len(centerlist) >= 1:
                    for path in centerlist:
                        cv2.circle(frame, path,radius=1,color=(0,255,0),thickness=5)    
                cv2.circle(frame, center,radius=1,color=(0,0,255),thickness=20)
                centerlist.append(center)
            elif tagID == 5:
                cv2.circle(frame, center,radius=1,color=(255,0,0),thickness=20)
                cx5,cy5 = center
            elif tagID == 10:
                cv2.circle(frame, center,radius=1,color=(255,0,0),thickness=20)
                cx10,cy10 = center
            elif tagID == 7:
                cv2.circle(frame, center,radius=1,color=(255,0,0),thickness=20)
                cx7,cy7 = center
            elif tagID == 11:
                cv2.circle(frame, center,radius=1,color=(255,0,0),thickness=20)
                cx11,cy11 = center
            if cx5 > 0 and cx10 > 0 and cx11 > 0 and cx7 > 0:

                dist1_5_10 = pix_dist(cx_s=cx5, cy_s=cy5, cx_f=cx10, cy_f=cy10)
                dist2_10_11 = pix_dist(cx_s=cx10, cy_s=cy10, cx_f=cx11, cy_f=cy11)
                dist3_11_7 = pix_dist(cx_s=cx11, cy_s=cy11, cx_f=cx7, cy_f=cy7)
                dist4_7_5 = pix_dist(cx_s=cx7, cy_s=cy7, cx_f=cx5, cy_f=cy5)

                #print(dist1_5_10)
                #print(dist2_10_11)
                #print(dist3_11_7)
                #print(dist4_7_5)

                cm_per_pixel = 40/(dist4_7_5)

                cv2.line(frame, (cx5, cy5), (cx10, cy10), (255, 0, 0), 5)
                cv2.line(frame, (cx10, cy10), (cx11, cy11), (255, 0, 0), 5)
                cv2.line(frame, (cx11, cy11), (cx7, cy7), (255, 0, 0), 5)
                cv2.line(frame, (cx7, cy7), (cx5, cy5), (255, 0, 0), 5)
            #print("TagID:" + str(tagID) + " Center X:" + str(cx) + " Center Y:" + str(cy))
        #print(Rhom_dist)
        """
        if i > 0:
            cx1,cy1 = Rhom_dist[i-1]
            cx2,cy2 = Rhom_dist[i]
            pixel_dist_per_frame = (((cx1-cx2)**2) + ((cy1-cy2)**2))**0.5
            print(pixel_dist_per_frame)
            instant_speed = (cm_per_pixel*pixel_dist_per_frame)/(seconds_per_frame)
        """
        #for detection in detections:

            # Access tag information (ID, position, orientation) from detection

            #tag_id = detection.id

        scale1 = 1
        
        """
        ret, frame_crop = cv2.threshold(frame_crop, 210, 255, cv2.THRESH_BINARY)

        k_e = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        frame_crop = cv2.dilate(frame_crop,k_e,iterations=2)
        frame_crop = cv2.erode(frame_crop,k_e,iterations=1)
        frame_crop = cv2.dilate(frame_crop,k_e,iterations=1)
        frame_crop = cv2.erode(frame_crop,k_e,iterations=1)
        
        
        find_frame = frame_crop
        cont, hier = cv2.findContours(frame_crop,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ig, size, array = hier.shape
        #print("Areas in frame")
        for contour in range(size):
            area = cv2.contourArea(cont[contour])
            if 400 > area and area > 50: 
                #print(area)
                #frame_crop_c = cv2.drawContours(frame_crop_c,[cont[contour]],-1,(0,0,255),1)
                pass
            elif area > 4000:
                #print("Big Area")
                #print(area)
                frame_crop_c = cv2.drawContours(frame_crop_c,[cont[contour]],-1,(0,255,0),3)
                moments =  cv2.moments(cont[contour])
                e = cv2.fitEllipse(cont[contour])
                angle = e[2] 
                x_screen, y_screen, w_screen, h_screen = cv2.boundingRect(cont[contour])
                frame_crop_screen = frame_crop[y_screen:(y_screen+h_screen), x_screen:(x_screen+w_screen)]
                frame_crop_screen_c = frame_crop_c[y_screen:(y_screen+h_screen), x_screen:(x_screen+w_screen)]
                delta_angle = (90 - angle)
                frame_level = imutils.rotate_bound(frame_crop_screen,delta_angle)
                frame_level_c = imutils.rotate_bound(frame_crop_screen_c,delta_angle)
                mirrored_image = cv2.flip(frame_level,0)
                mirrored_image_c = cv2.flip(frame_level_c,0)
                cont_m, hier_m = cv2.findContours(mirrored_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                ig_m, size_m, array_m = hier_m.shape
                for contour_m in range(size_m):
                    area = cv2.contourArea(cont_m[contour_m])
                    if area > 10 and area < 20000:
                        #print(area)
                        mirrored_image_c = cv2.drawContours(mirrored_image_c,[cont_m[contour_m]],-1,(0,0,255),1)
                        h_mirror,w_mirror = mirrored_image.shape
                        #print("size of small image")
                        #print("height")
                        #print(h_mirror)
                        #print("width")
                        #print(w_mirror)
                        #Size 298x139
                        grid_w = 32
                        grid_h = 60
                        x_off = 67#63
                        y_off = 8#10
                        off_1 = 3
                        off_2 = 0
                        w_m = grid_w
                        h_m = grid_h
                        
                        x_1 = x_off
                        y_1 = y_off

                        x_2 = grid_w + x_off + off_1
                        y_2 = y_off
                        x_3 = (grid_w*2) + x_off + (off_1*2)
                        y_3 = y_off 
                        
                        x_4 = x_off
                        y_4 = grid_h + y_off 
                        
                        x_5 = grid_w + x_off + off_1
                        y_5 = grid_h + y_off 
                        
                        x_6 = grid_w*2 + x_off + (off_1)*2
                        y_6 = grid_h + y_off
                        
                        num_x = -110
                        num_y = -30


                        mirrored_image_c = cv2.rectangle(mirrored_image_c, (x_1, y_1), (x_1+w_m, y_1+h_m), (255, 0, 255), 2)
                        
                        
                        
                        ans = num_identify (image=mirrored_image, image_c=mirrored_image_c, y=y_1, h=h_m, x=x_1, w=w_m, grid=1)
                        org = (x_1-num_x, y_1-num_y)
                        mirrored_image_c = cv2.putText(mirrored_image_c, str(ans), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("The number in grid 1 is")
                        #print(ans)
                        mirrored_image_c = cv2.rectangle(mirrored_image_c, (x_2, y_2), (x_2+w_m, y_2+h_m), (255, 0, 255), 2)
                        ans = num_identify (image=mirrored_image, image_c=mirrored_image_c, y=y_2, h=h_m, x=x_2, w=w_m, grid=2)
                        org = (x_2-num_x, y_2-num_y)
                        mirrored_image_c = cv2.putText(mirrored_image_c, str(ans), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("The number in grid 2 is")
                        #print(ans)
                        mirrored_image_c = cv2.rectangle(mirrored_image_c, (x_3, y_3), (x_3+w_m, y_3+h_m), (255, 0, 255), 2)
                        ans = num_identify (image=mirrored_image, image_c=mirrored_image_c, y=y_3, h=h_m, x=x_3, w=w_m, grid=3)
                        org = (x_3-num_x, y_3-num_y)
                        mirrored_image_c = cv2.putText(mirrored_image_c, str(ans), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("The number in grid 3 is")
                        #print(ans)
                        mirrored_image_c = cv2.rectangle(mirrored_image_c, (x_4, y_4), (x_4+w_m, y_4+h_m), (255, 0, 255), 2)
                        ans = num_identify (image=mirrored_image, image_c=mirrored_image_c, y=y_4, h=h_m, x=x_4, w=w_m, grid=4)
                        org = (x_4-num_x, y_4-num_y)
                        mirrored_image_c = cv2.putText(mirrored_image_c, str(ans), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("The number in grid 4 is")
                        #print(ans)
                        mirrored_image_c = cv2.rectangle(mirrored_image_c, (x_5, y_5), (x_5+w_m, y_5+h_m), (255, 0, 255), 2)
                        ans = num_identify (image=mirrored_image, image_c=mirrored_image_c, y=y_5, h=h_m, x=x_5, w=w_m, grid=5)
                        org = (x_5-num_x, y_5-num_y)
                        mirrored_image_c = cv2.putText(mirrored_image_c, str(ans), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("The number in grid 5 is")
                        #print(ans)
                        mirrored_image_c = cv2.rectangle(mirrored_image_c, (x_6, y_6), (x_6+w_m, y_6+h_m), (255, 0, 255), 2)
                        ans = num_identify (image=mirrored_image, image_c=mirrored_image_c, y=y_6, h=h_m, x=x_6, w=w_m, grid=6)
                        org = (x_6-num_x, y_6-num_y)
                        mirrored_image_c = cv2.putText(mirrored_image_c, str(ans), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        #print("The number in grid 6 is")
                        #print(ans)

                        
        """            
        #small_frame = cv2.resize(mirrored_image_c, (0,0), fx=scale1, fy=scale1)
        
        #cont, hier = cv2.findContours(frame_crop,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        scale2 = 0.2
        """
        x_org_speed = 2700
        y_org_speed = 250
        x_org_COT = 2700
        y_org_COT = 250 + 150
        x_org_peak = 2700
        y_org_peak = 250 + 300

        speed_org = (x_org_speed,y_org_speed)
        COT_org = (x_org_COT,y_org_COT)
        peak_org = (x_org_peak,y_org_peak)
        frame = cv2.putText(frame, "speed_str", speed_org, font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "COT_str", COT_org, font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "peak_str", peak_org, font, fontScale, color, thickness, cv2.LINE_AA)
        """
        frame_s = cv2.resize(frame, (0,0), fx=scale2, fy=scale2)
        
        
        #############################################################
        out.write(frame)
        #############################################################

        #cv2.imshow('Frame', small_frame)
        #cv2.imshow('Find', find_frame)
        
        cv2.imshow('Frame2',frame_s)

        if isinstance(frame,np.ndarray):
            frame_final = frame
        else:
            print("final frame")

        i = i + 1
        if cv2.waitKey(1) == ord('q'):
            break
    
    else:
        break

##############################

out.release()

##############################

cx_start,cy_start = Rhom_dist[0]
cx_final,cy_final = Rhom_dist[-1]
Total_dist = (((cx_start-cx_final)**2) + ((cy_start-cy_final)**2))**0.5

speed_ser = []
center1_ser = []
center2_ser = []
for data in Rhom_inst_speed:
    speed, center1, center2 = data
    speed_ser.append(speed)
    center1_ser.append(center1)
    center2_ser.append(center2)

speed_data = {'speed':speed_ser,'center1':center1_ser,'center2':center2_ser}
speed_df = pd.DataFrame(speed_data)
Peak_speed_idx =speed_df['speed'].idxmax()
Peak_speed = speed_df.loc[Peak_speed_idx,'speed']
Peak_speed_c1 = speed_df.loc[Peak_speed_idx,'center1']
Peak_speed_c2 = speed_df.loc[Peak_speed_idx,'center2']
#output critical information
print("total dist in cm")
dist_in_cm = Total_dist*cm_per_pixel
print(dist_in_cm)
print("Average Speed")
Avg_speed = dist_in_cm/time_test
print(Avg_speed)
print("peak speed in cm_s")
print(Peak_speed)

print("image size")
print(frame_final.shape)
#org = (x_4-num_x, y_4-num_y)

speed_str = "Speed: " + str(round(Avg_speed,2))
COT_str = "COT: " + str(round(COT_final,2))
peak_str = "Peak: " + str(round(Peak_speed,2))

x_org_speed = 2700
y_org_speed = 250
x_org_COT = 2700
y_org_COT = 250 + 150
x_org_peak = 2700
y_org_peak = 250 + 300

speed_org = (x_org_speed,y_org_speed)
COT_org = (x_org_COT,y_org_COT)
peak_org = (x_org_peak,y_org_peak)
frame_final = cv2.putText(frame_final, speed_str, speed_org, font, fontScale, color, thickness, cv2.LINE_AA)
frame_final = cv2.putText(frame_final, COT_str, COT_org, font, fontScale, color, thickness, cv2.LINE_AA)
frame_final = cv2.putText(frame_final, peak_str, peak_org, font, fontScale, color, thickness, cv2.LINE_AA)
#Draw on image
cv2.line(frame_final, (cx_start, cy_start), (cx_final, cy_final), (0, 0, 255), 5)
cv2.circle(frame_final,(cx_start,cy_start),1,(0,0,255),5)
cv2.circle(frame_final,(cx_final,cy_final),1,(0,0,255),5)

cv2.line(frame_final, (Peak_speed_c1),(Peak_speed_c2), (255,0,255),10)
#Write File
newname = file.split(".")
test_file = newname[0] + ".jpg"
print(test_file)
cv2.imwrite(test_file,frame_final)
#print(i)
#print(len(Rhom_dist))
video.release()
cv2.destroyAllWindows()