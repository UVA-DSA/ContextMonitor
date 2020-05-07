import numpy as np
from scipy.fftpack import fft
from sklearn import externals, mixture, neighbors, model_selection, preprocessing, pipeline, linear_model, metrics
import matplotlib
from pylab import savefig
import pandas as pd
import os, math, sys, calendar, time
from pandas.util.testing import assert_frame_equal
if os.environ.get('DISPLAY','') == '':
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Generates faulty trajectory to be tested on the robot
"""
class injectFaults:
    def __init__(self,directoryPath, csvPath, picklePath, cartError, cartDuration, graspError, graspDuration, task, injNum):
        self.directoryPath = directoryPath
        self.csvPath = csvPath
        self.task = task
        self.picklePath = "{}/{}".format(picklePath, task.replace(".csv",""))
        self.cartError = cartError
        self.graspError = graspError
        self.injNum = injNum
        self.cartDuration = cartDuration
        self.graspDuration = graspDuration
	if os.environ.get('DISPLAY','') == '':
    		print('no display found. Using non-interactive Agg backend')
   		matplotlib.use('Agg')
        self.generatePlots(self.task)

    def getinjectedFile(self):
        return self.writeFile
    def loadFile(self, expDir):
        """
        datafile is a dummy to overwrite the df, writefile is the new csv file with the fault injections
        """
        dataFile = "{}{}".format(self.csvPath, self.task)
        print dataFile
        self.writeFile = self.task.replace(".csv", "{}.csv".format(self.injNum))
        self.writeFile = "{}/{}".format(expDir,self.writeFile)
        print self.writeFile
        df = pd.read_csv(dataFile, delimiter=',')
        s = np.array(df[['field.pos_d0', 'field.pos_d1', 'field.pos_d2', 'field.pos_d3', 'field.pos_d4', 'field.pos_d5']])
        m = np.array(df[['field.grasp_d0', 'field.grasp_d1']])
        traj = np.array(s)
        grasp = np.array(m)
        e_traj = self.mfi_code(traj, int(self.cartDuration[0]*len(traj)), int(self.cartDuration[1]*len(traj)), self.cartError)#0,1 #60110, 65110)
        e_grasp, gscale_factor = self.grasp_errorModel(grasp, int(self.graspDuration[0]*len(grasp)), int(self.graspDuration[1]*len(grasp)), self.graspError)#0,1 #60110, 65110)
        print traj.shape
        count = 0
        for index, row in df.iterrows():
            df.at[index, 'field.pos_d0'] = e_traj[count][0]
            df.at[index, 'field.pos_d1'] = e_traj[count][1]
            df.at[index, 'field.pos_d2'] = e_traj[count][2]
            df.at[index, 'field.grasp_d0'] = e_grasp[count][0]
            count +=1
        print "grasp shape: {}".format(count)

        df.to_csv(self.writeFile, sep = ',')
        return e_traj, e_grasp, self.graspError,self.cartError

    def makeDirectory(self):
        count = self.injNum
        print count
        taskDir = self.task.replace(".csv","")
        print taskDir
        directory = "{}/ismr_results/{}/exp%d".format(os.path.abspath(os.path.dirname(sys.argv[0])), taskDir)%count
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except:
            "nothing"

        return directory

    def grasp_closeModel(self, c_data, _start,_end, target):
        e_data = np.zeros((c_data.shape[0], c_data.shape[1]))
        for i in range(e_data.shape[0]):
            e_data[i] = c_data[i]
        for i in range(_start, _end):
            e_data[i][0] = target
        return e_data, target

    """
    marginal addition over previous value in grasp
    """
    def grasp_errorModel(self, c_data, _start, _end, err):
        e_data = np.zeros((c_data.shape[0], c_data.shape[1]))
        target = err
        for i in range(e_data.shape[0]):
            e_data[i]=c_data[i]

        scale_factor = self.findSFactor(c_data,_start,_end, target)
        grasp_1 = np.amax(c_data[:,0])
        checkpoint = c_data[_start][0]
        next_checkpoint = c_data[_end-1]

        for i in range(_start, _end):
            if checkpoint < err:
                checkpoint +=  scale_factor
            e_data[i][0] = checkpoint


        return e_data, target

    def findSFactor(self, grasp, _start,_end, target):
        diff = target-grasp[_start][0]
        scale_factor = diff/int(0.2*(_end-_start))
        print "grasp0 {} sf: {}".format(grasp[_start][0], scale_factor)
        return scale_factor

    def mfi_code(self, traj, _start, _end, err):
        dist = err#, '700', '1500', '3000', '6000','9000')
        code = []
        param = []
        variable = ['field.pos_d0', 'field.pos_d1', 'field.pos_d2']
        for index in range(_start, _end):
            delta = ((dist/math.sqrt(3)))
            for col in range(traj.shape[1]/2):
                traj[index][col] = traj[index][col] + delta

        return traj

    def plotgrasp(self, e_grasp, grasp, gscale_factor, tscale_factor, _dir):
        transcripts = externals.joblib.load('{}/transition_points1.p'.format(self.picklePath))
        print transcripts
        left_z = e_grasp[:,0]
        right_z = grasp[:,0]
        y_label = "Grasper angle (rad)"
        x_label = "time (ms)"
        plt.plot(left_z, 'b', label='erroneous grasp')
        plt.plot(right_z, 'r', label = 'correct grasp')
        plt.xlabel(x_label, fontsize = 35)
        plt.ylabel(y_label, fontsize = 35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize = 35, loc = 'lower left')
        #51064-52063

        transcripts = self.reorderData(transcripts, 2)
        for i in range(len(transcripts)):
            plt.axvline(transcripts[i][0],color='k', linestyle='--', label= 'segments')
            plt.text(transcripts[i][0],np.amax(right_z)+0.13, int(transcripts[i][2]), fontsize = 25)

        figName = "{}/graspg_{}c{}.pdf".format(_dir, gscale_factor, tscale_factor)
        #plt.show()
        savefig(figName, dpi = 600, aspect = 'auto')
        print figName
        plt.close()

    def reorderData(self, data, column):
        data = np.array(data)
        for i in range(data.shape[0]):
            data[i][column] = i
        return data

    def get3d(self, left, right):
        left_z = []
        right_z = []
        for i in range (left.shape[0]):
            left_z.append((left[i][0]**2 + left[i][1]**2 + left[i][2]**2)**0.5)
            right_z.append((right[i][0]**2 + right[i][1]**2 + right[i][2]**2)**0.5)
        return np.array(left_z), np.array(right_z)

    def generatePlots(self, taskCsv=None):
        traj = []
        task = self.task.replace(".csv","")
        expDir = self.makeDirectory()
        segCount = 7
        c_traj = np.array(externals.joblib.load("{}/raw_traj.p".format(self.picklePath))) #replace the pickle file with the trajectory you want to load
        grasp = externals.joblib.load('{}/grasp.p'.format(self.picklePath))
        e_traj,e_grasp, gscale_factor, tscale_factor = np.array(self.loadFile(expDir))
        left_z, right_z = self.get3d(np.array(c_traj[:,0:3]),np.array(e_traj[:,0:3]))
	if not os.environ.get('DISPLAY','') == '':
        	self.plotCart(gscale_factor, tscale_factor, left_z, right_z, expDir)
        	self.plotgrasp(e_grasp, grasp, gscale_factor, tscale_factor, expDir)

    def plotCart(self, gscale_factor, tscale_factor, left_z, right_z, expDir):
        y_label = "x-y-z value (mm)"
        x_label = "Time (ms)"
        transcripts = externals.joblib.load('{}/transition_points1.p'.format(self.picklePath))
        plt.plot(left_z, 'b')
        plt.plot(right_z, 'r')

        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel(x_label, fontsize = 35)
        plt.ylabel(y_label, fontsize = 35)

        plt.legend(fontsize = 45, loc = "lower right")
        figName = "{}/cartg_{}c{}.pdf".format(expDir, gscale_factor, tscale_factor)
        transcripts = self.reorderData(transcripts,2)
        for i in range(len(transcripts)):
            plt.axvline(transcripts[i][0],color='k', linestyle='--', label= 'segments')
            plt.text(transcripts[i][0],1.035*np.amax(left_z), int(transcripts[i][2]), fontsize = 25)
        savefig(figName,aspect = 'auto')
        print figName

        plt.close()
        #cmd = " scp {}/new_test_16.csv  homa@172.28.39.158:/home/homa/raven_2/teleop_data/".format(expDir)
        #os.system(cmd)
