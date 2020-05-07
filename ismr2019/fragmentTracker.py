import numpy as np
import pandas as pd
import cv2, argparse, imutils, os, sys
from sys import argv
from sklearn import externals
from skimage.measure import compare_ssim
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from collections import deque
from dtw import dtw
from PIL import Image

class fragmentTracker:
    def __init__(self, imgSource, csvPath):
        _path = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.imgDirectory = imgSource
        
        self.imgSource = imgSource
        self.picklePath ="{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/pickles")
        self.orgFrames = []
        self.csvPath = csvPath
        
        cv2.useOptimized()


    def detectColor(self):
        """
        Thresholds a given image using HSV
        """
        
        vidFile = "{}{}".format(self.csvPath.replace("/csvs/", "/vids/"), self.imgSource.replace(".csv",".avi").replace("latest_run","videotraj"))
        print vidFile
        coloredFrames = []
        print "current video file {} ".format(vidFile)
        cap = cv2.VideoCapture(vidFile)
        success, frame = cap.read()
        i = 0
        if not os.path.isfile("{}/{}/coloredFrames.p".format(self.picklePath, self.imgSource.replace(".csv",""))):
            while(cap.isOpened()):
                frame_exists, curr_frame = cap.read()
                if frame_exists:
                    frame = curr_frame
                    self.orgFrames.append(frame[0:frame.shape[0]-100, 0:frame.shape[1]-400])
                    #frame = frame[380:frame.shape[0]-200, 210:frame.shape[1]-420]          
                    #frame = frame[200:frame.shape[0]-200, 210:frame.shape[1]-320]          
                    #frame = frame[x1:frame.shape[0]-x2, y1:frame.shape[1]-y2]          
                    #cv2.imshow("frame", frame)
                    #cv2.waitKey(1)
                    #hsvFrame = cv2.cvtColor(frame[100:frame.shape[0], 400:frame.shape[1]], cv2.COLOR_BGR2HSV)
                    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    kernel = np.ones((5,5), np.uint8)
                    mask = cv2.inRange(hsvFrame, (0,0,0), (180,35,123))#(2,150,100), (8,250,255))
                    
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.medianBlur(mask,5)
                    coloredFrames.append(mask)
                    
                else:
                    cap.release()
                    break         
            externals.joblib.dump(coloredFrames, "{}/{}/coloredFrames.p".format(self.picklePath, self.imgSource.replace(".csv","")))
        else:
            print "already processed"
            coloredFrames = externals.joblib.load("{}/{}/coloredFrames.p".format(self.picklePath, self.imgSource.replace(".csv","")))
        print "Done pickling"

        return coloredFrames

    def writeVideo(self):
        vidFile = self.imgSource
        print "writeVideo for file {}".format(vidFile)
        framePath = vidFile.replace(".avi","/")
        self.framePath =framePath
        try:
            if os.path.exists(framePath):
                "do nothing"
            else:
                os.makedirs(framePath)
        except :
            "do nothing"

        cap = cv2.VideoCapture(vidFile)
        success, frame = cap.read()
        i = 0
        while(cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                cv2.imwrite("{}/frame%d.png".format(framePath)%i, curr_frame)
                cv2.waitKey(1)
                #foo = Image.open("{}/frame%d.png".format(framePath)%i)
                #foo.save("{}/frame%d.png".format(framePath)%i,optimize = True, quality = 95)
                i+=1
            else:
                break
        cap.release()




    def compareImages(self):
        """
        Compare between adjacent images to find the highest dissimilarity
        """
        coloredFrames = self.detectColor()
        _min = 1
        _minFrame = 0
        diff_array = []
        #frame = frame[200:frame.shape[0]-200, 210:frame.shape[1]-320]          
        for i in range(100,len(coloredFrames)-1):
            #print i
            (score, diff) = compare_ssim(coloredFrames[i][200:coloredFrames[i].shape[0]-200, 210:coloredFrames[i].shape[1]-320], coloredFrames[i-1][200:coloredFrames[i+1].shape[0]-200, 210:coloredFrames[i+1].shape[1]-320], full=True)
            diff = (diff * 255).astype("uint8")
            
            
            diff_array.append(score)
            if (score<_min):
                _min = score
                _minFrame = i
        #print "_minFrame {}".format(_minFrame)
        plt.plot(diff_array, label= "SSIM across corresponding frames")
        plt.legend()
        #plt.show()
        
        if _minFrame > 0 and _minFrame<= 0.85*len(coloredFrames):
            return self.findFailure(int(_minFrame))
        
        return 0

    def findFailure(self, minFrame):
        metaFile = "{}timestamps{}".format(self.csvPath, self.imgSource.replace("latest_run",""))
        df = pd.read_csv(metaFile, delimiter =",")
        print df.at[minFrame, "field.stamp"]
        return df.at[minFrame, "field.stamp"]

    def centerFragment(self, load):
        """
        Tried to track the thresholded fragment
        """
        if load == 0:
            coloredFrames = self.detectColor()
            #externals.joblib.dump(coloredFrames, "{}/{}coloredFrames.p".format(self.picklePath, self.imgSource.replace(".csv","")))
        else:
            coloredFrames = externals.joblib.load("{}/coloredFrames.p".format(self.picklePath))
            self.orgFrames = externals.joblib.load("{}/{}.p".format(self.picklePath, self.imgDirectory))
        
        centers = []
        prev= [0,0]
        print "length {}".format(len(coloredFrames))
        #frame = frame[380:frame.shape[0]-200, 210:frame.shape[1]-420]          
        for i in range(1,len(coloredFrames)):
            cnts = cv2.findContours((coloredFrames[i][380:coloredFrames[i].shape[0]-200, 210:coloredFrames[i].shape[1]-420]).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            #print i
            #cv2.imshow("frame",coloredFrames[i])
            #cv2.waitKey(1)
            if len(cnts)>0:
                c = max(cnts, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    prev = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
                    centers.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
                else:
                    centers.append(prev)
            else:
                centers.append(prev)
        return np.array(centers), coloredFrames

    def trackFragment(self, load = 0):
        if not os.path.isfile("{}/{}/fragmentCenters.p".format(self.picklePath, self.imgDirectory.replace(".csv",""))):
            fragmentCenters, frames = self.centerFragment(load)
            print fragmentCenters.shape
            """
            for i in range(1,fragmentCenters.shape[0]):
                if fragmentCenters[i-1] is None or fragmentCenters[i] is None:
                    continue
                thickness = 10#int(np.sqrt(args["buffer"]/float(i+1))*2.5)
                #cv2.line(self.orgFrames[i], (fragmentCenters[i-1][0],fragmentCenters[i-1][1]), (fragmentCenters[i][0], fragmentCenters[i][1]), (255,255,255), thickness)
                #cv2.imwrite("tracked/frames%d.png"%i, self.orgFrames[i])
                #cv2.waitKey(1)
                #diff[i][0] = (fragmentCenters[i][0]**2 + fragmentCenters[i][1]**2)**0.5 - (fragmentCenters[i-1][0]**2 + fragmentCenters[i-1][1]**2)**0.
            cv2.destroyAllWindows()
            cv2.waitKey(1)        
            """
            externals.joblib.dump(fragmentCenters, "{}/{}/fragmentCenters.p".format(self.picklePath, self.imgDirectory.replace(".csv","")))
        else:
            print "fragment center already calculated for {}".format(self.imgSource.replace(".csv",""))


    def getDistance(self, centers):
        distance = np.zeros(len(centers))
        for i in range(len(centers)):
            distance[i] = (centers[i][0]**2 + centers[i][1]**2)**0.5
        return distance

    def playVideo(self):
        output = "movie.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        out = cv2.VideoWriter(output, fourcc, 30.0, (720, 1280))
        for i in range(1,2500):
            frame = cv2.imread("{}/left%06d.png".format(self.imgSource)%i)


    def compareTrack(self):
        self.trackFragment()
        
        reference = self.imgSource[0:12]+ "0"
        print reference
        ref = self.getDistance(externals.joblib.load("{}/{}/fragmentCenters.p".format(self.picklePath, reference)))
        err = self.getDistance(externals.joblib.load("{}/{}/fragmentCenters.p".format(self.picklePath, self.imgDirectory.replace(".csv",""))))
        """
        plt.ylabel('x-y position of the tissue (pixel)', fontsize = 35)
        plt.xlabel('Frame Number', fontsize = 35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.plot(ref, 'bo-' ,label='correct trajectory')
        plt.plot(err, 'g^-', label = 'faulty trajectory')
        plt.legend(fontsize = 35);
        plt.show()
        plt.close()
        """
        return self.computeDTW(ref,err)


    def computeDTW(self, x = None, y = None):
        distances = np.zeros((len(y), len(x)))
        for i in range(len(y)):
            for j in range(len(x)):
                distances[i,j] = (x[j]-y[i])**2
        accumulated_cost = np.zeros((len(y), len(x)))
        accumulated_cost[0,0] = distances[0,0]
        for i in range(1,len(x)):
            accumulated_cost[0,i] = distances[0,i]+accumulated_cost[0,i-1]
        for i in range(1,len(y)):
            accumulated_cost[i,0] = distances[i,0] + accumulated_cost[i-1,0]
        for i in range(1,len(y)):
            for j in range(1,len(x)):
                accumulated_cost[i,j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]
        path, cost = self.path_cost(x,y, accumulated_cost, distances)
        print "path.shape {} x {} y {}".format(len(path), len(x), len(y))
        x_label = "Frame Number"
        y_label = "x-y position of the tissue (pixels)"
        plt.xlabel(x_label, fontsize = 35)
        plt.ylabel(y_label, fontsize = 35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.plot(x, 'bo-' ,label='correct trajectory')
        plt.plot(y, 'g^-', label = 'faulty trajectory')
        plt.legend(fontsize = 35, loc = "lower left")

        paths = self.path_cost(x, y, accumulated_cost, distances)[0]
        dist_path = []

        for [map_x, map_y] in paths:
            dist_path.append([map_y, (x[map_x]- y[map_y])])
        
        dist_path = np.array(dist_path)
        dist_path = dist_path[dist_path[:,0].argsort()]
        
        dist_max = dist_path[0][1]
        dist_index = 0
        dist_indexvalue = 0
        for i in range(int(0.90*dist_path.shape[0])):
            if dist_max < abs(dist_path[i][1]):
                dist_max = (dist_path[i][1])
                dist_index = dist_path[i][0]
                dist_indexvalue = i
                print dist_index
                
            else:
                #print "{} {}".format(dist_max, dist_path[i][1])
                dist_cost = 0
        cost = np.array(cost)
        plt.axvline(dist_index, color = 'b', linestyle = ':')
        plt.annotate('Failure', xy=(dist_index-30, 580), xytext=(dist_index+180, 535),arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, facecolor='black'), fontsize = 45)

        for [map_x, map_y] in paths:
            plt.plot([map_x, map_y], [x[map_x], y[map_y]], 'r')

        #plt.show()
        if dist_index>0 and dist_index<= 0.95*len(y) and dist_max>5:
            return self.findFailure(int(dist_index))
        return 0
    def path_cost(self, x, y, accumulated_cost, distances):
        path = [[len(x)-1, len(y)-1]]
        cost = 0
        i = len(y)-1
        j = len(x)-1
        while i>0 and j>0:
            if i==0:
                j = j - 1
            elif j==0:
                i = i - 1
            else:
                if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                    i = i - 1
                elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                    j = j-1
                else:
                    i = i - 1
                    path.append([0,0])
                    j= j- 1
            path.append([j, i])
        for [y, x] in path:
            cost = cost +distances[x, y]
        return path, cost

    def reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def grabScreen(self):
        imgSource = "/home/uva-dsa1/Downloads/dsn2019/output.mp4"
        cap = cv2.VideoCapture(imgSource)
        fps = cap.get(cv2.CAP_PROP_FPS)

        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]

        while(cap.isOpened()):
            frame_exists, curr_frame = cap.read()
            if frame_exists:
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
            else:
                break
        cap.release()
        print timestamps


"""
csvPath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/csvs/")
fgt = fragmentTracker("/home/yq/hdd_files/samin/dsn2019/vids/homa_6.avi", csvPath)
fgt.writeVideo()

csvPath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/csvs/")
fgt = fragmentTracker("/home/yq/hdd_files/samin/dsn2019/vids/homa_2.avi", csvPath)
fgt.writeVideo()

try:
    script,mode = argv
except:
    print "Error: missing parameters"
    print usage
    sys.exit(0)

if mode == "0":
    fgt.compareImages()
elif mode == "1":
    fgt.trackFragment()
elif mode == "2":

    fgt.playVideo()
elif mode == "3":
    fgt.trackFragment(1)
elif mode == "4":
    fgt.compareTrack()
"""
