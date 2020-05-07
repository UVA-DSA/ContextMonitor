import os, sys, glob
import numpy as np
import pandas as pd
from sklearn import externals
import matplotlib.pyplot as plt
from pylab import savefig

class detectFailure:
    def __init__(self, task, csvPath, picklePath, minFrame):
        self.referenceFile = "new_test_21"
        self.task = task
        self.csvPath = csvPath
        self.picklePath = picklePath
        
        

    def findFailurePoint(self, packetNum):
        taskFile = "{}{}".format(self.csvPath, self.task)
        df = pd.read_csv(taskFile, delimiter = ',')
        
        print df.at[packetNum, "field.hdr.stamp"]
        return df.at[packetNum, "field.hdr.stamp"]

    def loadcurrFile(self):
        taskFile = "{}{}".format(self.csvPath, self.task)
        df = pd.read_csv(taskFile, delimiter = ',')
        traj = np.array(df[['field.pos_d0', 'field.pos_d1', 'field.pos_d2', 'field.pos_d3', 'field.pos_d4', 'field.pos_d5']])
        grasp =np.array(df[[ 'field.grasp_d0', 'field.grasp_d1']])
        self.resultDirectory = "{}/ismr_results/detection/{}/".format(os.path.abspath(os.path.dirname(sys.argv[0])), self.task)
        try:
            if not os.path.exists(self.resultDirectory):
                os.makedirs(self.resultDirectory)
        except:
            "do nothing"
        
        return traj, grasp

    def loadConstraints(self):
        cartConstraints = []
        graspConstraints = []
        summaryPath = self.csvPath.replace("csvs", "summary/csvs")
        globInput = "{}{}".format(summaryPath,"/*")
        for name in glob.glob(globInput):
            print "glob output :{} ".format(name)          
            df = pd.read_csv(name, delimiter = ',')
            cartConstraints.append(np.array(df[["x_max",	"y_max",	"z_max",	"x_min",	"y_min",	"z_min"]]))
            graspConstraints.append(np.array(df[["g_max",	"gmin"]]))
        cartConstraints = np.array(cartConstraints)
        graspConstraints = np.array(graspConstraints)
        #print "cartConstrants  {}".format(cartConstraints)
        globalCartConstraints = np.zeros((cartConstraints.shape[1],cartConstraints.shape[2]))
        globalGraspConstraints = np.zeros((graspConstraints.shape[1],graspConstraints.shape[2]))
        
        for i in range(cartConstraints.shape[0]):
            for j in range(cartConstraints.shape[1]):
                for k in range(cartConstraints.shape[2]):
                    if (k<cartConstraints.shape[2]/2):
                        globalCartConstraints[j][k] = np.amax(cartConstraints[:,j,k])
                    else:
                        globalCartConstraints[j][k] = np.amin(cartConstraints[:,j,k])     
        
        for i in range(graspConstraints.shape[0]):
            for j in range(graspConstraints.shape[1]):
                for k in range(graspConstraints.shape[2]):
                    if (k<graspConstraints.shape[2]/2):
                        globalGraspConstraints[j][k] = np.amax(graspConstraints[:,j,k])
                    else:
                        globalGraspConstraints[j][k] = np.amin(graspConstraints[:,j,k])
        #print globalGraspConstraints
        return globalCartConstraints, globalGraspConstraints

    def detectFaults(self):
        packetNum = 0
        transitions = self.loadTransitions()
        self.transitions = transitions
        cartConstraints, graspConstraints = self.loadConstraints()
        
        traj, grasp = self.loadcurrFile()
        for j in range(len(transitions)):
            segmentKinematics = traj[transitions[j][0]:transitions[j][1]]
            segmentGrasp = grasp[transitions[j][0]:transitions[j][1]]
            cartViolations, cartpacketNum, violated = self.checkCartViolations(segmentKinematics, cartConstraints[j])
            
            #print "Segment {} CartViolations {} packetNum {}".format(j, cartViolations, packetNum)
            graspViolations, grasppacketNum, violated = self.checkGraspViolations(segmentGrasp, graspConstraints[j])
            if violated ==1:
                packetNum = grasppacketNum+transitions[j][0]
                break
	if not os.environ.get('DISPLAY','') == '':        
	        self.cartPlots(traj, self.resultDirectory)
	        self.graspPlots(grasp, self.resultDirectory, packetNum)
        return self.findFailurePoint(packetNum)
            
    def checkCartViolations(self,segTraj, constraints):
        violations = [0,0,0]
        cartPacket = 0
        violated = 0
        for i in range(segTraj.shape[0]):
            for j in range(segTraj.shape[1]/2):
                if (segTraj[i][j]>1.01*constraints[j] or segTraj[i][j]<1.01*constraints[j+3]):
                    if np.array(violations).all() == 0:
                        cartPacket = i
                    violations[j] += 1
        violations = np.array(violations)
        if violations.any() >1000:
            violated = 1
        
        return violations, cartPacket, violated 

    def checkGraspViolations(self, segGrasp, constraints):
        violations = [0]
        graspPacket = 0
        violated = 0
        for i in range(segGrasp.shape[0]):
            if (segGrasp[i][0]>1.01*constraints[0] or segGrasp[i][0]<1.01*constraints[1]):
                if np.array(violations).all() == 0:
                    graspPacket = i
                violations[0] += 1
        if violations[0] > 1000:
            violated =1
        
        return violations, graspPacket, violated 

    def loadTransitions(self):
        #print "{}/{}/transition_points1.p".format(self.picklePath,self.task)
        transitions = externals.joblib.load("{}/{}/transition_points2.p".format(self.picklePath,self.task.replace(".csv","")))
        #print transitions
        return transitions
    
    def get3d(self, left, right):
        """
        Finds the distance from the origin of each x-y-z points
        """
        left_z = []
        right_z = []
        for i in range (left.shape[0]):
            left_z.append((left[i][0]**2 + left[i][1]**2 + left[i][2]**2)**0.5)
            right_z.append((right[i][0]**2 + right[i][1]**2 + right[i][2]**2)**0.5)
        return left_z, right_z

    def cartPlots(self, traj, resultDirectory):      
        figName = self.task.replace(".csv", "Cartesian.png")
        figName = "{}{}".format(resultDirectory, figName)
        transcripts = self.transitions
        left_z, right_z = self.get3d(traj[:,0:3], traj[:,3:6])
        y_label = "x-y-z value"
        x_label = "steps in trajectory"
        plt.plot(left_z, 'b', label='left_manip')
        plt.plot(right_z, 'r', label='right manip')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        for i in range(len(transcripts)):
            plt.axvline(transcripts[i][0],color='k', linestyle='--', label= 'segments')
            plt.text(transcripts[i][0],np.amax(right_z)+0.1, int(transcripts[i][2]), fontsize = 3)
            #i=i
        savefig(figName, dpi = 600, aspect = 'auto')
        #print figName
        plt.close()

    def graspPlots(self, grasp, resultDirectory, packetNum):      
        figName = self.task.replace(".csv", "Grasp.png")
        figName = "{}{}".format(resultDirectory, figName)
        transcripts = self.transitions
        y_label = "grasper angle"
        x_label = "steps in trajectory"
        plt.plot(grasp[:,0], 'b', label='left_manip')
        plt.plot(grasp[:,1], 'r', label='right manip')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.annotate('Fault detected', xy=(packetNum, np.max(grasp[:,0])), xytext=(15776, 0.7*np.max(grasp[:,0])),arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=10, facecolor='black'), fontsize = 40)
        for i in range(len(transcripts)):
            plt.axvline(transcripts[i][0],color='k', linestyle='--', label= 'segments')
            plt.text(transcripts[i][0],np.amax(grasp[:,0])+0.1, int(transcripts[i][2]), fontsize = 3)
            #i=i
        savefig(figName, dpi = 600, aspect = 'auto')
        #print figName
        plt.close()
