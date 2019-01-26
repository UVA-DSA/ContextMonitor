import pandas as pd
import numpy as np
from sklearn import mixture, preprocessing, metrics, decomposition, cluster, externals, preprocessing
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
from pylab import savefig
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
import sys, time, os
import os.path
from sys import argv
import time,math
import itertools
if os.environ.get('DISPLAY','') == '':
        mpl.use('Agg')

class trajSegments:

    def __init__(self, csvPath, task, mode):
        self.name = ""
        self.csvPath = csvPath
        self.task = task
        self.taskName = task.replace(".csv","")
        self.picklePath = "{}{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/pickles/",self.taskName)
        self.summary_traj = []
        self.seg_count = 0
        self.left_raw = []
        self.right_raw = []
        try:
            if not os.path.exists(self.picklePath):
                os.makedirs(self.picklePath)
        except:
            "nothing"
        self.mode  = mode
        self.generatePlots()
        if self.mode == "learn":
            self.doProcessing(csvPath, task)

    def readData(self):
        """
        Reads data from csv in the form of data frames and generates different matrices for different features (cartesian, joints, velocity)
        """
        traj = []
        joints = []
        grasp = []
        r3_index = []
        rob_states = []
        processedAnnotations = []
        dataFile = "{}{}".format(self.csvPath, self.task)
        print dataFile
        df = pd.read_csv(dataFile, delimiter=',')
        s = df[['field.runlevel', 'field.pos_d0', 'field.pos_d1', 'field.pos_d2', 'field.pos_d3', 'field.pos_d4', 'field.pos_d5']]
        v = df[['field.jpos0', 'field.jpos1', 'field.jpos2', 'field.jpos3', 'field.jpos4', 'field.jpos5', 'field.jpos6','field.jpos7', 'field.jpos8',
         'field.jpos9', 'field.jpos10', 'field.jpos11', 'field.jpos12', 'field.jpos13','field.jpos14', 'field.jpos15']]

        m = np.array(df[[ 'field.grasp_d0', 'field.grasp_d1']])
        t = np.array(df[["field.hdr.stamp"]])
        r_t = []
        s = np.array(s)
        v = np.array(v)

        for i in range(s.shape[0]):
            if s[i][0]==3 or s[i][0]==2:
                r3_index.append(i)
                traj.append(s[i,1:])
                joints.append(v[i,:])
                rob_states.append(s[i,:1])
                grasp.append(m[i])
                r_t.append(t[i])

        r_t = np.array(r_t)
        grasp = np.array(grasp)
        joints = np.array(joints)
        traj = np.array(traj)
        rob_states = np.array(rob_states)
        left_traj, right_traj, left_joints, right_joints = self.prepTraj(traj, joints)
        if self.mode == "evaluate" or self.mode == "learn" or self.mode == "detect":
            processedAnnotations = self.getAnnotation(df)

        if not self.mode == "learn":
            externals.joblib.dump(r_t, '{}/kinTime.p'.format(self.picklePath))
            externals.joblib.dump(grasp, '{}/grasp.p'.format(self.picklePath))
            externals.joblib.dump(traj, '{}/raw_traj.p'.format(self.picklePath))
            externals.joblib.dump(r3_index, '{}/r3_index.p'.format(self.picklePath))
            externals.joblib.dump(left_joints, '{}/procesed_joints.p'.format(self.picklePath))
        return dataFile, rob_states,left_traj, right_traj, left_joints, right_joints, grasp, processedAnnotations

    def removeDuplicates(traj, joints):
        new_traj = []
        new_joints = []
        ndp_index = []
        for i in range(1,len(traj)):
            if (traj[i] == traj[i-1]).all():
                pass
            else:
                ndp_index.append(i)
                new_traj.append(traj[i])
                new_joints.append(joints[i])
        if not self.mode == "detect":
            externals.joblib.dump(ndp_index, '{}/ndp_index.p'.format(self.picklePath))
        return np.array(new_traj), np.array(new_joints)

    def prepTraj(self, traj, joints):
        """
        MinMaxScaling for different features. The scaling transformation is dumped in the form of pickle file to be used later if needed
        """

        normalize = 1
        left_traj = traj[:,0:3]
        right_traj = traj[:,3:6]
        left_joints = joints[:,0:8]
        right_joints = joints[:,8:16]

        self.left_raw, self.right_raw = left_traj, right_traj

        if normalize == 1:
            scaler = preprocessing.MinMaxScaler().fit(left_traj)
            left_traj = scaler.transform(left_traj)
            scaler1 = preprocessing.MinMaxScaler().fit(right_traj)
            right_traj = scaler1.transform(right_traj)
            scaler2 = preprocessing.MinMaxScaler().fit(left_joints)
            left_joints = scaler2.transform(left_joints)
            scaler3 = preprocessing.MinMaxScaler().fit(right_joints)
            right_joints = scaler3.transform(right_joints)
            if not self.mode == "detect":
                externals.joblib.dump(scaler2, '{}/left_scaler.p'.format(self.picklePath))
                externals.joblib.dump(scaler3, '{}/right_scaler.p'.format(self.picklePath))
        if not self.mode == "detect":
            externals.joblib.dump(left_traj, '{}/left_traj.p'.format(self.picklePath))
            externals.joblib.dump(right_traj, '{}/right_traj.p'.format(self.picklePath))

        return left_traj, right_traj, left_joints,right_joints

    def getAnnotation(self, df):
        rawAnnotations = np.array(df[["field.seg_label"]])
        processedAnnotations = []
        

        prev = 0
        for i in range(rawAnnotations.shape[0]-1):
            if rawAnnotations[i]!=rawAnnotations[prev]:
                if i-prev>100:
                    processedAnnotations.append([prev, i, rawAnnotations[prev]])
                prev = i
        processedAnnotations.append([prev, len(rawAnnotations), rawAnnotations[prev]])
        if len(processedAnnotations) == 8:
            print "popping extra segment"
            processedAnnotations.pop(1)
            processedAnnotations[0][1] = processedAnnotations[1][0]
        processedAnnotations = np.array(processedAnnotations)
        print processedAnnotations
        if self.mode == "learn":
            externals.joblib.dump(processedAnnotations, "{}/manualLabels.p".format(self.picklePath))

        return processedAnnotations

    def makeSegments(self, left=None, right=None, annotations = [], t=0):
        """
        This function calls the GMM function to get segment boundaries, termed transcripts
        """
        transcripts = self.clustersNDP(left, annotations, t)
        return transcripts

    def makeSegmentations(self, kinematics=None, transcripts=None, grasp=None,task=None):
        """
        This function is used to group different segments according to the segment boundaries, which is then used for post-processing
        """

        global_left = []
        seg_kinematics = []
        seg_grasp = []
        traj_raw = np.concatenate((self.left_raw, self.right_raw), axis = 1)
        transcripts = np.array(transcripts)
        print "kinematics.shape: {}".format(transcripts)
        prev_seg = transcripts[0][transcripts.shape[1]-1]

        for i in range(0,transcripts.shape[0]):
            if transcripts[i][transcripts.shape[1]-1] != prev_seg:
                print "{}".format(prev_seg)
                self.processSegments(np.array(seg_kinematics),  np.array(seg_grasp), prev_seg, task)
                prev_seg = transcripts[i][transcripts.shape[1]-1]
                seg_kinematics = []
                global_left = []
                seg_grasp = []
            for n in range (int(transcripts[i][0]), int(transcripts[i][1])):
                seg_kinematics.append(traj_raw[n])
                global_left.append(traj_raw[n])

                seg_grasp.append(grasp[n])
        prev_seg = transcripts[i][transcripts.shape[1]-1]
        print "final_seg: {}".format(prev_seg)
        self.processSegments(np.array(seg_kinematics), np.array(seg_grasp), prev_seg, task)

    def processSegments(self, kinData, grasp, prev_seg,  task):
        """
        This function is used for generating local bounds in each segment and export them to a csv file
        """
        self.summary_traj


        gesture = "{}{}{}G{}.p".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/segmented_trajectories/", task,self.seg_count)
        gesture = '{}{}G{}.p'.format(self.picklePath, task,self.seg_count)
        print gesture
        self.seg_count+=1

        #coeffs = np.polynomial.chebyshev.Chebyshev.fit(kinData[:,0],kinData[:,1], 15)
        #print coeffs
        _mean, _std, _max, _min = [], [], [], []
        x_mean = np.mean(kinData[:,0])
        y_mean = np.mean(kinData[:,1])
        z_mean = np.mean(kinData[:,2])
        x_std = np.std(kinData[:,0])
        y_std = np.std(kinData[:,1])
        z_std = np.std(kinData[:,2])
        x_max = np.max(kinData[:,0])
        y_max = np.max(kinData[:,1])
        z_max = np.max(kinData[:,2])
        x_min = np.min(kinData[:,0])
        y_min = np.min(kinData[:,1])
        z_min = np.min(kinData[:,2])
        g_max = np.amax(grasp[:,0])
        g_min = np.amin(grasp[:,0])

        self.summary_traj.append([prev_seg, x_mean, y_mean, z_mean, x_std, y_std, z_std, x_max, y_max, z_max, x_min, y_min, z_min, g_max, g_min])

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

    def generate_transition_features(self, trajectory, temporal_window):
        """
        Generates transition features accoridng to a temporal window. Instead of having one data points per time, it concatenates a sequence of size temporal_window to be fed to the GMM
        """
        X_dimension = trajectory.shape[1]

        T = trajectory.shape[0]
        N = None

        for t in range(T - temporal_window):

        	n_t = self.make_transition_feature(trajectory, temporal_window, t)
        	N = self.safe_concatenate(N, n_t)

        return N

    def make_transition_feature(self, matrix, temporal_window, index):
        result = None
        for i in range(temporal_window + 1):
        	result = self.safe_concatenate(result, self.reshape(matrix[index + i]), axis = 1)
        return result

    def reshape(self, data):
        """
        Reshapes any 1-D np array with shape (N,) to (1,N).
        """
        return data.reshape(1, data.shape[0])

    def safe_concatenate(self, X, W, axis = 0):
        if X is None:
        	return W
        else:
        	return np.concatenate((X, W), axis = axis)

    def clustersNDP(self, demonstrations = None, annotations=None, t=3):
        """
        First layer of clustering based on joints
        """
        temporal_window = t
        traj = np.array(demonstrations)
        traj = self.generate_transition_features(traj, temporal_window)
        labels = np.zeros((len(traj),1))
        for i in range(len(annotations)):
            labels[annotations[i][0]:annotations[i][1]][0] = annotations[i][2]
        lowest_bic = np.infty
        bic = []
        gmm = mixture.BayesianGaussianMixture(n_components = 10, covariance_type='full', max_iter = 10000, tol = 1e-7, random_state = 00, weight_concentration_prior_type= 'dirichlet_process', weight_concentration_prior = 0.001)
        gmm.fit(traj)
        print "gmm converged {}".format(gmm.converged_)
        results = []
        for i in range(traj.shape[0]):
            results.append(gmm.predict(traj[i].reshape(1,-1)))
        value = gmm.score(traj)
        transition_points = []
        prev_index = 0

        for i in range(len(results)-1):
            if results[i]!=results[i+1] and i-prev_index>100:
                transition_points.append([prev_index,i,results[i]])
                prev_index = i
        transition_points.append([prev_index, traj.shape[0], results[i]])

        externals.joblib.dump(transition_points, '{}/transition_points1.p'.format(self.picklePath))
        if not self.mode == "detect":
            externals.joblib.dump(gmm, '{}/gmm.p'.format(self.picklePath))
            externals.joblib.dump(gmm.means_, '{}/gmm.latest_run.p'.format(self.picklePath))

        return transition_points

    def generatePlots(self):
        """
        Read data from csv, feeds them to the GMM and plots the segmented trajectory
        """
        if self.mode == "detect" and not os.path.isfile("{}{}".format(self.csvPath, self.task)):
            return

        picklePath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/pickles")
        dataFile, rob_states, left, right, left_joints, right_joints, grasp, annotations = self.readData()
        figName = dataFile.replace(".csv", ".png")

        traj = np.concatenate((left, right), axis = 1)
        task = dataFile.replace('.csv','').split('/')[4]

        if self.mode == "learn":
            transcripts = annotations
            self.makeSegmentations(traj,transcripts, grasp, task)
        else:
            transcripts = self.makeSegments(left_joints, right_joints, annotations, 3)

        if self.mode =="evaluate" or self.mode == "detect":
            refTranstionPath = self.picklePath.replace(self.task.replace(".csv",""), "")
            refSegments = externals.joblib.load("{}referenceTransitions.p".format(refTranstionPath, ))
            transcripts = self.omitSegments(refSegments, transcripts)
            
        if self.mode == "evaluate" or self.mode == "detect":
            self.evaluateSegments(annotations, transcripts)

        left_z, right_z = self.get3d(left, right)

        if os.environ.get('DISPLAY','') == '':
            print('no display found. Using non-interactive Agg backend')
            mpl.use('Agg')
            return

        y_label = "x-y-z value"
        x_label = "steps in trajectory"
        fig, ax1 = plt.subplots()
        ax1.plot(left_z, 'b', label='left_manip')
        #plt.plot(right_z, 'r', label='right manip')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        #plt.legend()

        for i in range(len(transcripts)):
            plt.axvline(transcripts[i][0],color='k', linestyle='--')
            plt.text(transcripts[i][0],np.amax(right_z)+0.1, int(transcripts[i][2]), fontsize = 3)
            #i=i
        ax2 = ax1.twinx()
        #savefig(figName, dpi = 600, aspect = 'auto')
    
        figName = figName.replace(".csv","grasp.csv")
        y_label = "grasper angle"
        x_label = "steps in trajectory"
        ax2.plot(grasp[:,0], 'r', label='grasp')
        #plt.plot(grasp[:,1], 'r', label='right manip')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        #plt.legend()
        savefig(figName, dpi = 600, aspect = 'auto')
        print figName

        plt.close()

    def doProcessing(self, filePath, task):
        """
        Writes the data of each segment to a csv file
        """
        task = task.replace(".csv","")
        summaryPath = filePath.replace("csvs", "summary/csvs")

        writeFile = '{}/{}summary.csv'.format(summaryPath, task)
        cp = pd.DataFrame(data = np.array(self.summary_traj), columns = ['seg','x_mean', 'y_mean', 'z_mean', 'x_std', 'y_std', 'z_std', 'x_max', 'y_max', 'z_max', 'x_min', 'y_min', 'z_min', 'g_max', 'gmin'])
        cp.to_csv(writeFile, sep = ',')

    def omitSegments(self, refSegments, segments):
        processedSegments = []
        check = 0
        j = 0
        print segments
        print refSegments
        _omitIndex = 0
        lenTraj = segments[len(segments)-1][1]
        #segments = np.array(segments)
        for i in range(0, len(refSegments)):
            _min = 1000000000000
            _minSegment = []
            #print "current reference {}".format(refSegments[i])
            for k in range(0, len(segments)):
                if abs(int(refSegments[i]*lenTraj) - segments[k][0]) <_min:
                    _min = abs(int(refSegments[i]*lenTraj) - segments[k][0])
                    _minSegment = segments[k]
                    _omitIndex = k
            print "Matched {} {}".format(int(refSegments[i]*lenTraj), _minSegment)
            segments.pop(_omitIndex)
            processedSegments.append(_minSegment)
            j = j+1
        processedSegments = np.array(processedSegments)
        processedSegments = processedSegments[processedSegments[:,0].argsort()]
        for i in range(len(processedSegments)-1):
            if processedSegments[i][1] != processedSegments[i+1][0]:
                processedSegments[i][1] = processedSegments[i+1][0]
        """
        #print np.array(processedSegments)
        time = externals.joblib.load("{}/kinTime.p".format(self.picklePath))
        segTime  = 0
        df = pd.read_csv("{}{}_header.csv".format(self.csvPath, self.task.replace(".csv","")))
        frameTime = np.array(df[["field.stamp"]])
        _iter = 0
        for i in range(len(segments)):
            segTime = time[segments[i][0]]
            diff  =np.infty
            for j in range(_iter,len(frameTime)):
                if abs(frameTime[j]-segTime)<diff:
                    diff = abs(frameTime[j]-segTime)
                    _iter = j
            print "closest matched frame {}".format(_iter)
        externals.joblib.dump(processedSegments, '{}/transition_points2.p'.format(self.picklePath))
        """
        externals.joblib.dump(processedSegments, '{}/transition_points2.p'.format(self.picklePath))
        return processedSegments
    def evaluateSegments(self, annotations, segments):
        """
        Jaccard similarity
        """
        overlap = 0.0
        jaccard_simmilarity = 0
        j = 0
        delta_t = []
        loopCount = len(annotations)
        if len(segments)<=len(annotations):
            loopCount = len(segments)
        print np.array(segments)
        for i in range(0, loopCount):
            _min = np.infty
            _minSegment = []
            for k in range(j, len(segments)):
                if abs(annotations[i][0] - segments[k][0]) <_min:
                    _min = abs(annotations[i][0] - segments[k][0])
                    _minSegment = segments[k]
                    j = k+1
            if _minSegment != []:
                overlap = self.calculateOverlaps(annotations[i], _minSegment)
                jaccard_simmilarity += overlap
                delta_t.append(annotations[i][1]-_minSegment[1])

        jaccard_simmilarity = jaccard_simmilarity/len(annotations)
        print "jaccard_simmilarity: {}".format(jaccard_simmilarity)
        print delta_t

    def calculateOverlaps(self, annotations, segments):
        _start = segments[0]
        _end = segments[1]
        if segments[0]<annotations[0]:
            _start = annotations[0]
        if segments[1]>annotations[1]:
            _end = annotations[1]
        diff = float(_end-_start)/float(annotations[1]-annotations[0])
        return diff

csvPath = "{}{}".format(os.path.abspath(os.path.dirname(sys.argv[0])), "/csvs/")
"""
for i in range(1,5):
    task = "automove_ravenstate_%d.csv"%i
    seg = trajSegments(csvPath, task, "evaluate")
"""


#def main():
