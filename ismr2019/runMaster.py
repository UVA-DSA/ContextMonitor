import sys, time, os, glob
import pandas as pd
import numpy as np
from sys import argv
from sklearn import externals
from getSegments import trajSegments
from injectFaults import injectFaults
from fragmentTracker import fragmentTracker
from detectFailure import detectFailure


import random, math


class runMaster:

    def runEverything(self,task, detect):
        self.task = task
        self.picklePath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/pickles")
        csvPath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/csvs/")
        injectionSummary = []
        filePath = "{}{}".format(csvPath, task)
        if detect == 0:
            """
            only learns constraints and reference transitions, no segmentation
            """
            seg = trajSegments(csvPath, task, "learn")

        elif detect == 1:
            """
            does segmentation using reference transitions and evaluates
            """
            seg = trajSegments(csvPath, task, "evaluate")

        elif detect == 2:
            """
            Injects faults
            """
            directoryPath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/segmented_trajectories/raw")
            """
            cartError = [3000, 65000, 6000, 9000]#, 0.01, 0.02, 0.03, 0.04, 0.05]
            graspError = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
            c_duration = [0.5, 0.6]
            g_duration = [0.55, 0.75]
            """
            cartError = [3000, 65000, 6000, 9000]#, 0.01, 0.02, 0.03, 0.04, 0.05]
            graspError = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            c_duration = [0.7, 0.9]
            g_duration = [0.65, 0.9]
            count = 0

            #cartErrorAdditions = [random.randint(3000,50000) for i in range(4)]
            #graspErrorAdditions = [random.uniform(0.8, math.pi/2) for i in range(4)]
            #cartError = cartError + cartErrorAdditions
            #graspError = graspError + graspErrorAdditions


            for c_err in cartError:
                for g_err in graspError:
                    injF = injectFaults(directoryPath, csvPath, self.picklePath, c_err, c_duration, g_err, g_duration, task, count)
                    inj_file = injF.getinjectedFile()
                    count +=1
                    injectionSummary.append([count, c_err, c_duration[0], c_duration[1], g_err, g_duration[0], g_duration[1], inj_file])
            print "--------------------------------"
            #_file = "{}{}".format(csvPath, task.replace(".csv","faultinjectionSummary.csv"))
            _file = "{}{}".format(csvPath, task.replace(".csv","faultinjectionSummaryNotDropped.csv"))
            isFile = self.checkcsv(_file)
            print isFile
            df = pd.DataFrame(data = (injectionSummary), columns = ["count", "c_err", "c_duration0","c_duration1", "g_err", "g_duration0","g_duration1", "filelink"])
            df.to_csv(_file, sep = ',')
        elif detect == 4:
                print "detecting mode for vision"
                _file = "{}{}".format(csvPath, task.replace(".csv","faultinjectionSummary.csv"))
                df = pd.read_csv(_file)
                summary = []

                count = 0
                for index, row in df.iterrows():
                    currTask = df.at[index, 'filelink']
                    currFile = currTask.split("/")
                    currFile = currFile[len(currFile)-1].replace("homa_","latest_run")
                    task = currFile
                    minFrame = self.extractFrames(task, csvPath)
                    #print "{} {} {}".format(packetNum, minFrame, minFrame - packetNum)

                    if (minFrame)>0:
                        failure = "TP"
                        #summary.append([task, failure, minFrame, packetNum])
                    else:
                        failure = "FP"
                        #summary.append([task, failure, "none", packetNum])
                    count = count + 1
        elif detect == 6:
                print "detecting mode for vision dtw"
                _file = "{}{}".format(csvPath, task.replace(".csv","faultinjectionSummaryNotDropped.csv"))
                df = pd.read_csv(_file)
                summary = []

                count = 0
                for index, row in df.iterrows():
                    currTask = df.at[index, 'filelink']
                    currFile = currTask.split("/")
                    currFile = currFile[len(currFile)-1].replace("homa_","latest_run")
                    task = currFile
                    minFrame = self.getDTW(task, csvPath.replace("csvs","csvs_nd"))
                    #print "{} {} {}".format(packetNum, minFrame, minFrame - packetNum)

                    if (minFrame)>0:
                        failure = "TP"
                        #summary.append([task, failure, minFrame, packetNum])
                    else:
                        failure = "FP"
                        #summary.append([task, failure, "none", packetNum])
                    count = count + 1
        elif detect == 7:
            print "detecting mode for not dropped"
            _file = "{}{}".format(csvPath, task.replace(".csv","faultinjectionSummaryNotDropped.csv"))
            df = pd.read_csv(_file)
            summary = []

            count = 0
            for index, row in df.iterrows():
                currTask = df.at[index, 'filelink']
                currFile = currTask.split("/")
                currFile = currFile[len(currFile)-1].replace("homa_","latest_run")
                task = currFile
                seg = trajSegments(csvPath.replace("csvs","csvs_nd"), task, "detect")
                print "detecting for file {}".format(task)
                dtf = detectFailure(task, csvPath.replace("csvs","csvs_nd"), self.picklePath.replace("pickles","pickles_nd"), 0)
                packetNum = dtf.detectFaults()
                self.kinPacketNum.append([currFile, packetNum])
                externals.joblib.dump(self.kinPacketNum, '{}/kinPacketNum.p'.format(self.picklePath))

        else:
            print "detecting mode"
            _file = "{}{}".format(csvPath, task.replace(".csv","faultinjectionSummary.csv"))
            df = pd.read_csv(_file)
            summary = []

            count = 0
            for index, row in df.iterrows():
                currTask = df.at[index, 'filelink']
                currFile = currTask.split("/")
                currFile = currFile[len(currFile)-1].replace("homa_","latest_run")
                task = currFile
                seg = trajSegments(csvPath, task, "detect")
                print "detecting for file {}".format(task)
                dtf = detectFailure(task, csvPath, self.picklePath, 0)
                packetNum = dtf.detectFaults()
                self.kinPacketNum.append([currFile, packetNum])
                externals.joblib.dump(self.kinPacketNum, '{}/kinPacketNum.p'.format(self.picklePath))
                """
                minFrame = self.extractFrames(task, csvPath)
                print "{} {} {}".format(packetNum, minFrame, minFrame - packetNum)

                if (minFrame)>0:
                    failure = "TP"
                    summary.append([task, failure, minFrame, packetNum])
                else:
                    failure = "FP"
                    summary.append([task, failure, "none", packetNum])
                count = count + 1
            self.summarizeResults(summary)
            """
    def checkcsv(self, _file):
        try:
            if os.path.isfile(_file):
                return 0
        except:
            return 1

    def extractFrames(self, vidFile, csvPath):
        fgt = fragmentTracker(vidFile, csvPath)
        #fgt.writeVideo()
        #fgt.compareTrack()
        minFrame = fgt.compareImages()
        if minFrame ==0:
            minFrame = fgt.compareTrack()
        return minFrame

    def summarizeResults(self, summary):
        writeFile = "Results.csv"
        df = pd.DataFrame(data=(summary), columns = ["task", "failure", "minFrame", "packetNum"])
        df.to_csv(writeFile)

    def getPercentages(self):
        self.picklePath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/pickles")
        task = self.task.split("_")[0]
        print task
        globPath = "{}/{}*".format(self.picklePath, task)
        allTransitions = []
        taskLength = 0
        allPercentage = []
        count = 0
        for name in glob.glob(globPath):
            if os.path.isfile("{}/manualLabels.p".format(name)):

                currTransitions = externals.joblib.load("{}/manualLabels.p".format(name))
                if count<7:
                    allPercentage.append(np.array(self.getDifference(currTransitions)))
                    count = count + 1
                percentageTime = []

        allPercentage = np.array(allPercentage).reshape(count,-1)
        for i in range(len(currTransitions)):
            print "median of  {} is {}".format(allPercentage[:,i], np.median(allPercentage[:,i]))
            percentageTime.append(np.median(allPercentage[:,i]))
        refPercentage = []
        for i in range(len(percentageTime)):
            if i!=2:
                refPercentage.append(percentageTime[i])
        print np.array(refPercentage)
        externals.joblib.dump(refPercentage,"{}/referenceTransitions.p".format(self.picklePath))

    def getDifference(self, transitions):
        taskLength = transitions[len(transitions)-1][1] - transitions[0][0]
        currPercentage = []
        print len(transitions)
        for i in range(len(transitions)):
            currPercentage.append(float(transitions[i][0])/taskLength)
        print currPercentage
        return currPercentage

rmt = runMaster()
try:
    script,mode = argv
except:
    print "Error: missing parameters"
    print usage
    sys.exit(0)

if mode == "0":
    "['0.0', '0.10517418785829781', '0.38677789210642366', '0.4873217698074379', '0.5531934440687932', '0.8042804820242315', '0.9004998012815513']"
    automove = "['0.0', '0.069841015118', '0.137835549008', '0.390237714406', '0.458762250517', '0.54939319747', '0.823513352564']"
    print "Learning Constraints and reference transition"

    for i in range(2,3):
        task = "new_test_%04d.csv"%i
        rmt.runEverything(task, int(mode))
        rmt.getPercentages()

elif mode == "1":
    print "Evaluating segments based on the learnt references"
    for i in range(160,161):
        task = "latest_run%d.csv"%i
        rmt.runEverything(task, int(mode))

elif mode == "2":
    print "Injecting Faults"
    for i in range(16,20):
        task = "homa_%d.csv"%i
        rmt.runEverything(task, int(mode))
elif mode == "3":
    print "Detecting Failures"
    for i in range(8,10):
        task = "homa_%d.csv"%i
        rmt.runEverything(task, int(mode))
