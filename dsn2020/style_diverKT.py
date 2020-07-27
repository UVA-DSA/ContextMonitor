import sys, os, glob, math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as ss



def loadOffsets():
        dfLength = 78
        kinOffset = {}
        kinSpan = {}
        kinOffset['cartesian'] = 0
        kinOffset['rotation'] = 3
        kinOffset['linearVelocity'] = 12
        kinOffset['angularVelocity'] = 15
        kinOffset['grasperAngle'] = 18
        kinSpan['cartesian'] = 3
        kinSpan['rotation'] = 9
        kinSpan['linearVelocity'] = 3
        kinSpan['angularVelocity'] = 3
        kinSpan['grasperAngle'] = 1
        return kinOffset, kinSpan

def loadDemonstrations(key):
    """
    This function globs over all the demonstrations for the given key (kinematics, transcriptions, video) and calls the plotgraph function after completing globbing
    """
    demonstrationsPath = dataPath + key
    globPath = demonstrationsPath + "/**/*"
    novices, intermediary, experts = loadMetaFile()

    cartesians = dict()
    for demonstration_name in glob.glob(globPath):
      cartesians[demonstration_name.split("/")[-1].replace(".txt","")] = readCartesians(demonstration_name)

    for novice_demonstrations in novices.keys():
      novices[novice_demonstrations] = cartesians[novice_demonstrations]

    for intermediary_demonstrations in intermediary.keys():
      intermediary[intermediary_demonstrations] = cartesians[intermediary_demonstrations]

    for experts_demonstrations in experts.keys():
      experts[experts_demonstrations] = cartesians[experts_demonstrations]
    return novices, intermediary, experts

def loadMetaFile():
    """
    This function loads the meta_file for needle_passing which gives the category-wise score for each demonstration along with the total score
    """
    novices = dict()
    intermediary = dict()
    experts = dict()
    metaFilePath = dataPath + "/meta_file_Suturing.txt"
    for name in glob.glob(metaFilePath):
      df = np.array(pd.read_csv(name, delimiter='\t', engine='python', header=None))
      for i in range(df.shape[0]):
        if df[i][2] == "N":
          novices[df[i][0]] = 1
        elif df[i][2] == "I":
          intermediary[df[i][0]] = 1
        else:
          experts[df[i][0]] = 1

    return novices, intermediary, experts

def readCartesians(demonstration):
    """
    This function reads the cartesian values from the kinematics file for each demonstration
    """
    df = np.array(pd.read_csv(demonstration, delimiter = '    ', engine='python', header = None))

    psm_offset = int(df.shape[1]/2)

    cartesians = np.asarray(df[:,psm_offset:])
    return cartesians

def splitKinematics():
    kinOffset, kinSpan = loadOffsets()
    novice_angularVelocity = dict()
    novice_grasperAngle = dict()
    novice_rotation = dict()
    novice_cartesian = dict()
    novice_linearVelocity = dict()
    novice_dict = [novice_angularVelocity, novice_grasperAngle, novice_rotation, novice_cartesian, novice_linearVelocity]

    intermediary_angularVelocity = dict()
    intermediary_grasperAngle = dict()
    intermediary_rotation = dict()
    intermediary_cartesian = dict()
    intermediary_linearVelocity = dict()
    intermediary_dict = [intermediary_angularVelocity, intermediary_grasperAngle, intermediary_rotation, intermediary_cartesian, intermediary_linearVelocity]

    expert_angularVelocity = dict()
    expert_grasperAngle = dict()
    expert_rotation = dict()
    expert_cartesian = dict()
    expert_linearVelocity = dict()
    expert_dict = [expert_angularVelocity, expert_grasperAngle, expert_rotation, expert_cartesian, expert_linearVelocity]

    novices, intermediary, experts = loadDemonstrations("/kinematics")
    count = 0
    for first_key in kinSpan.keys():
      offset, span = kinOffset[first_key], kinSpan[first_key]

      for expert_demonstration in experts.keys():
        current_demonstration = experts[expert_demonstration]
        manipulator_offset = int(current_demonstration.shape[1]/2)
        expert_dict[count][expert_demonstration] = np.concatenate((current_demonstration[:,offset:offset+span], current_demonstration[:,manipulator_offset + offset:manipulator_offset + offset+span]), axis=1)
        if first_key == 'rotation':
          expert_dict[count][expert_demonstration] = orientation(expert_dict[count][expert_demonstration])

      for novice_demonstration in novices.keys():
        current_demonstration = novices[novice_demonstration]
        manipulator_offset = int(current_demonstration.shape[1]/2)
        novice_dict[count][novice_demonstration] = np.concatenate((current_demonstration[:,offset:offset+span], current_demonstration[:,manipulator_offset + offset:manipulator_offset + offset+span]), axis=1)
        if first_key == 'rotation':
          novice_dict[count][novice_demonstration] = orientation(novice_dict[count][novice_demonstration])

      for intermediary_demonstration in intermediary.keys():
        current_demonstration = intermediary[intermediary_demonstration]
        manipulator_offset = int(current_demonstration.shape[1]/2)
        intermediary_dict[count][intermediary_demonstration] = np.concatenate((current_demonstration[:,offset:offset+span], current_demonstration[:,manipulator_offset + offset:manipulator_offset + offset+span]), axis=1)
        if first_key == 'rotation':
          intermediary_dict[count][intermediary_demonstration] = orientation(intermediary_dict[count][intermediary_demonstration])

      count +=1

    return novice_dict, intermediary_dict, expert_dict

def orientation(rotationMatrix):
    _ori = np.zeros((len(rotationMatrix),6))
    for i in range(_ori.shape[0]):
        for j in range(_ori.shape[1]):
            _ori[i][0] = math.atan2(rotationMatrix[i][7], rotationMatrix[i][8])
            _ori[i][1] = math.atan2(-rotationMatrix[i][6], (rotationMatrix[i][8]**2 +rotationMatrix[i][7]**2)**0.5)
            _ori[i][2] = math.atan2(rotationMatrix[i][3], rotationMatrix[i][0])
            _ori[i][3] = math.atan2(rotationMatrix[i][16], rotationMatrix[i][17])
            _ori[i][4] = math.atan2(-rotationMatrix[i][15],(rotationMatrix[i][17]**2 + rotationMatrix[i][16]**2)**0.5)
            _ori[i][5] = math.atan2(rotationMatrix[i][12], rotationMatrix[i][9])
    return _ori

def findrange(opt, sub):
  """
  This function finds the range for one histogram and applies that to another
  """
  common_range = np.zeros((opt.shape[1],2))

  for i in range(opt.shape[1]):
    common_range[i][0] = min(min(opt[:,i]), min(sub[:,i]))
    common_range[i][1] = max(max(opt[:,i]), max(sub[:,i]))

  grid = np.zeros((int(opt.shape[1]), 10))
  #print ("common range {}".format(common_range))
  for i in range(grid.shape[0]):
    grid[i] = np.linspace(common_range[i][0], common_range[i][1], num=10)
  #print ("grid {}".format(grid))

  return common_range, grid

def getHistogram(novice_dict, intermediary_dict, expert_dict):
    traversed_dict = dict()
    traversed = np.zeros((5,5))
    kinOffset, kinSpan = loadOffsets()
    for i in range(len(novice_dict)):
      for j in range(len(novice_dict)):
        if i!=j and traversed[i][j] == 0:

          novice_list = list()
          expert_list = list()
          intermediary_list = list()
          #print ("first Variable {} second Variable {}".format(kinSpan.keys()[i], kinSpan.keys()[j]))

          for key in novice_dict[i].keys():
            spani = int(novice_dict[i][key].shape[1]/2)
            spanj = int(novice_dict[j][key].shape[1]/2)
            left_kinematics = np.concatenate((novice_dict[i][key][:,0:spani],novice_dict[j][key][:,0:spanj]), axis=1)
            right_kinematics = np.concatenate((novice_dict[i][key][:,spani:],novice_dict[j][key][:,spanj:]), axis=1)
            novice_list.extend(np.concatenate((left_kinematics,right_kinematics), axis=1))

          novice_array = np.asarray(novice_list)
          #print (novice_array.shape)

          for key in intermediary_dict[i].keys():
            spani = int(intermediary_dict[i][key].shape[1]/2)
            spanj = int(intermediary_dict[j][key].shape[1]/2)
            left_kinematics = np.concatenate((intermediary_dict[i][key][:,0:spani],intermediary_dict[j][key][:,0:spanj]), axis=1)
            right_kinematics = np.concatenate((intermediary_dict[i][key][:,spani:],intermediary_dict[j][key][:,spanj:]), axis=1)
            intermediary_list.extend(np.concatenate((left_kinematics,right_kinematics), axis=1))

          intermediary_array = np.asarray(intermediary_list)
          #print (intermediary_array.shape)

          for key in expert_dict[i].keys():
            spani = int(expert_dict[i][key].shape[1]/2)
            spanj = int(expert_dict[j][key].shape[1]/2)
            left_kinematics = np.concatenate((expert_dict[i][key][:,0:spani],expert_dict[j][key][:,0:spanj]), axis=1)
            right_kinematics = np.concatenate((expert_dict[i][key][:,spani:],expert_dict[j][key][:,spanj:]), axis=1)
            expert_list.extend(np.concatenate((left_kinematics,right_kinematics), axis=1))

          expert_array = np.asarray(expert_list)
          #print (expert_array.shape)

          js_diff_ne, js_diff_ie, js_diff_ni = gethistograms(novice_array, intermediary_array, expert_array)
          traversed_dict["{}{}".format(kinSpan.keys()[i], kinSpan.keys()[j])] = [js_diff_ne, js_diff_ie, js_diff_ni]
          traversed[i][j] = 1
          traversed[j][i] = 1
    traversed_df = pd.DataFrame.from_dict(traversed_dict)
    traversed_df.to_csv("{}/traversedSuturingSkill.csv".format(path))
def gethistograms(novice_array, intermediary_array, expert_array):

    manip = 2
    span = int(novice_array.shape[1]/2)
    kldiff = list() ; jsdiff = list()
    #print (expert_array.shape)
    #print (intermediary_array.shape)
    #print (novice_array.shape)
    common_range, grid = findrange(expert_array, intermediary_array)

    intermediary_kernel = (ss.gaussian_kde(intermediary_array.T))#.evaluate([2, 2, 2, 2]))
    intermediary_values = intermediary_kernel.evaluate(grid)

    expert_kernel = (ss.gaussian_kde(expert_array.T))#.evaluate([2, 2, 2, 2]))
    expert_values = expert_kernel.evaluate(grid)

    novice_kernel = (ss.gaussian_kde(novice_array.T))#.evaluate([2, 2, 2, 2]))
    novice_values = novice_kernel.evaluate(grid)

    js_diff_ne = distance.jensenshannon(novice_values, expert_values)
    #print ("novice expert jsdiff {}".format(js_diff_ne))
    js_diff_ie = distance.jensenshannon(intermediary_values, expert_values)
    #print ("intermediary expert jsdiff {}".format(js_diff_ie))

    js_diff_ni = distance.jensenshannon(novice_values, intermediary_values)
    #print ("novice intermediary jsdiff {}".format(js_diff_ni))
    return js_diff_ne, js_diff_ie, js_diff_ni

def classifyDemonstrations():
    demonstrationsPath = dataPath + "/kinematics"
    globPath = demonstrationsPath + "/**/*"
    optimal_sequence = {'Needle_Passing_B004': 0, 'Needle_Passing_I002': 0, 'Needle_Passing_I004': 0, 'Needle_Passing_I005': 0, 'Needle_Passing_B003': 0, 'Needle_Passing_F001': 0, 'Needle_Passing_E001': 0, 'Needle_Passing_E003': 0}
    suboptimal_sequence = {'Needle_Passing_C002': ['G23F', 'G24F'], 'Needle_Passing_C003': ['G30F', 'G81F'], 'Needle_Passing_I003': ['G41F', 'G22F'], 'Needle_Passing_B001': ['G22F', 'G42F'], 'Needle_Passing_C004': ['G23F'], 'Needle_Passing_B002': ['G60FF'], 'Needle_Passing_H002': ['G21F', 'G61F'], 'Needle_Passing_C005': ['G60F', 'G40F'], 'Needle_Passing_H005': ['G31F', 'G61F'], 'Needle_Passing_H004': ['G21FF'], 'Needle_Passing_E004': ['G23FF'], 'Needle_Passing_E005': ['G21FF', 'G23FF'], 'Needle_Passing_F003': ['G20FF'], 'Needle_Passing_C001': ['G21FF', 'G22F', 'G23FF', 'G63F'], 'Needle_Passing_F004': ['G20F', 'G40F'], 'Needle_Passing_D005': ['G11F', 'G51F'], 'Needle_Passing_D004': ['G23F', 'G110F'], 'Needle_Passing_D003': ['G61F', 'G41F'], 'Needle_Passing_D002': ['G22FF', 'G81F', 'G24F'], 'Needle_Passing_D001': ['G11FF', 'G23F', 'G24F']}
    optimal_sequence = {'Suturing_G004': 0, 'Suturing_I004': 0, 'Suturing_I005': 0, 'Suturing_C004': 0, 'Suturing_C005': 0, 'Suturing_C003': 0, 'Suturing_I003': 0, 'Suturing_F005': 0}
    suboptimal_sequence = {'Suturing_D005': ['G30F', 'G60F', 'G62F', 'G42F'], 'Suturing_D004': ['G60F', 'G90F', 'G92F', 'G42F', 'G33FF'], 'Suturing_D003': ['G30F', 'G60F', 'G31FF'], 'Suturing_D002': ['G30FF', 'G60F', 'G61F'], 'Suturing_D001': ['G30F', 'G60F', 'G90FF', 'G31FF'], 'Suturing_E001': ['G30FF', 'G62FF'], 'Suturing_E002': ['G30FF', 'G31FF', 'G62F', 'G42F'], 'Suturing_E003': ['G62FF'], 'Suturing_E004': ['G30FF', 'G61FF'], 'Suturing_E005': ['G30FF', 'G31FF', 'G33FF'], 'Suturing_B005': ['G30FF', 'G31FF', 'G32FF'], 'Suturing_B004': ['G30FF'], 'Suturing_B001': ['G30FF', 'G31F', 'G22F', 'G32FF'], 'Suturing_B003': ['G30FF', 'G32FF', 'G62F'], 'Suturing_B002': ['G30FF', 'G31FF', 'G32FF'], 'Suturing_C002': ['G30FF'], 'Suturing_C001': ['G30FF'], 'Suturing_H003': ['G30FF', 'G32F', 'G62F', 'G33FF'], 'Suturing_H001': ['G31F', 'G61F', 'G32FF', 'G33FF'], 'Suturing_H005': ['G33F', 'G63F'], 'Suturing_H004': ['G31F', 'G61F', 'G32F', 'G62F', 'G33FF'], 'Suturing_F001': ['G62FF'], 'Suturing_F003': ['G61F', 'G41F'], 'Suturing_F002': ['G30FF', 'G61FF', 'G62FF'], 'Suturing_F004': ['G30FF', 'G61F', 'G41F', 'G32FF'], 'Suturing_G002': ['G33F', 'G81F'], 'Suturing_G003': ['G32FF'], 'Suturing_G001': ['G31FF', 'G32F', 'G82F', 'G84FF'], 'Suturing_G005': ['G32FF'], 'Suturing_I001': ['G30FF'], 'Suturing_I002': ['G32FF']}


    suboptimal_dict = dict()
    cartesians = dict()

    for key in suboptimal_sequence.keys():
        suboptimal_dict[key] = 0

    for demonstration_name in glob.glob(globPath):
      cartesians[demonstration_name.split("/")[-1].replace(".txt","")] = readCartesians(demonstration_name)

    for novice_demonstrations in optimal_sequence.keys():
      optimal_sequence[novice_demonstrations] = cartesians[novice_demonstrations]

    for intermediary_demonstrations in suboptimal_dict.keys():
      suboptimal_dict[intermediary_demonstrations] = cartesians[intermediary_demonstrations]


    return optimal_sequence, suboptimal_dict

def organizeDemonstrations():

    kinOffset, kinSpan = loadOffsets()
    optimal_angularVelocity = dict()
    optimal_grasperAngle = dict()
    optimal_rotation = dict()
    optimal_cartesian = dict()
    optimal_linearVelocity = dict()
    optimal_dict = [optimal_angularVelocity, optimal_grasperAngle, optimal_rotation, optimal_cartesian, optimal_linearVelocity]

    suboptimal_angularVelocity = dict()
    suboptimal_grasperAngle = dict()
    suboptimal_rotation = dict()
    suboptimal_cartesian = dict()
    suboptimal_linearVelocity = dict()
    suboptimal_dict = [suboptimal_angularVelocity, suboptimal_grasperAngle, suboptimal_rotation, suboptimal_cartesian, suboptimal_linearVelocity]

    optimal_demonstrations, suboptimal_demonstrations = classifyDemonstrations()

    count = 0
    for first_key in kinSpan.keys():
      offset, span = kinOffset[first_key], kinSpan[first_key]

      for optimal_demonstration in optimal_demonstrations.keys():
        current_demonstration = optimal_demonstrations[optimal_demonstration]
        manipulator_offset = int(current_demonstration.shape[1]/2)
        optimal_dict[count][optimal_demonstration] = np.concatenate((current_demonstration[:,offset:offset+span], current_demonstration[:,manipulator_offset + offset:manipulator_offset + offset+span]), axis=1)
        if first_key == 'rotation':
          optimal_dict[count][optimal_demonstration] = orientation(optimal_dict[count][optimal_demonstration])

      for suboptimal_demonstration in suboptimal_demonstrations.keys():
        current_demonstration = suboptimal_demonstrations[suboptimal_demonstration]
        manipulator_offset = int(current_demonstration.shape[1]/2)
        suboptimal_dict[count][suboptimal_demonstration] = np.concatenate((current_demonstration[:,offset:offset+span], current_demonstration[:,manipulator_offset + offset:manipulator_offset + offset+span]), axis=1)
        if first_key == 'rotation':
          suboptimal_dict[count][suboptimal_demonstration] = orientation(suboptimal_dict[count][suboptimal_demonstration])

      count +=1
    return optimal_dict, suboptimal_dict


def runEverything():
    #novice_dict, intermediary_dict, expert_dict = splitKinematics()
    novice_dict, intermediary_dict = organizeDemonstrations()
    #print (sorted(novice_dict[0].keys()))
    expert_dict= novice_dict
    getHistogram(novice_dict, intermediary_dict, expert_dict)


path = os.path.abspath(os.path.dirname(sys.argv[0])) + "/Suturing"
file_ = path+ "/meta_file_Suturing.txt"
dataPath = path
runEverything()
