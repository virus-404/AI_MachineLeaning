import collections
import itertools
import math
import sys
import time


"""
T3 Define a function to load the data into a bidimensional list
named data
"""
def read(file_name,spliter):
    data = []
    with open(file_name,'r') as file:
        for line in file:
            data.append(list(map(parser,line.strip('\n').split(spliter))))
    return data

def parser(element):
    try:
        return int(element)
    except ValueError:
        try:
            return float(element)
        except ValueError:
            return element

"""
T4 Create counts of possible results
"""
def unique_counts(part):
    results= {}
    for line in part:
        results[line[-1]] = results.get(line[-1],0) + 1
    return results

"""
T5 function that computes the Gini index of a node
"""
def gini_impurity(part):
    total = float(len(part))
    results = unique_counts(part)
    imp = 0
    for value in results.values():
        imp += (value/total)**2
    return 1 - imp

"""
T6 function that computes the entropy of a node
"""
def entropy(part):
    from math import log
    log2 = lambda x:log(x)/log(2)
    results = unique_counts(part)
    # Now calculate the entropy
    imp = 0.0
    for val in results.values():
        p = float(val)/len(part)
        imp -= p*log2(p)
    return imp

"""
T7 function that partitions a previous partition, taking
into account the values of a given attribute (column).
column is the index of the column and value is the value of
the partition criterion.
"""
def divideset(part, column, value):
    isplit_function = None
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda prot: prot[column]>=value
    else:
        split_function = lambda prot: prot[column]==value
    set1, set2 = [], []
    for line in part:
        if split_function(line): set1.append(line)
        else: set2.append(line)
    return set1, set2

'''
T8 class decisionnode, which represents a node
in the tree.
'''
class decisionnode:
    def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb

'''
T9 recursive function that builds a decision tree using any
of the impurity measures.
'''
def buildtree(part, scoref=entropy, beta=0):
    if len(part)==0: return decisionnode()
    best_gain, best_criteria, best_sets = calc_gainer(part,scoref)

    if (best_gain > beta):
        tb = buildtree(best_sets[0], scoref, beta)
        fb = buildtree(best_sets[1], scoref, beta)
        return decisionnode(best_criteria[0],best_criteria[1],None,tb,fb)
    else:
        return decisionnode(results=unique_counts(part))

"""
T10 iterative version of the previous function
"""
def buildtree_ite(part, scoref=entropy, beta=0):
    root = decisionnode()
    if len(part)==0: return root
    stack = [[root,part,'root']]

    while len(stack) != 0:
        actual = stack.pop()
        best_gain, best_criteria, best_sets = calc_gainer(actual[1],scoref)
        if (best_gain > beta):
            tmp = decisionnode(col=best_criteria[0], value=best_criteria[1],results = None)
            stack.append([tmp, best_sets[0],'true_branch'])
            stack.append([tmp, best_sets[1],'false_branch'])
        else:
            tmp = decisionnode(results=unique_counts(actual[1]))

        if(actual[2] == 'root'): root = tmp
        elif (actual[2] == 'true_branch'): actual[0].tb = tmp
        else: actual[0].fb = tmp

    return root

def calc_gainer(part,scoref):
    current_score = scoref(part)
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    for col in range(0,len(part[0]) - 1):
        elements = set([value[col] for value in part])
        for element in elements:
            set1, set2 = divideset(part, col, element)
            p = float(len(set1))/len(part)
            gain = current_score - p*scoref(set1) - (1-p)*scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, element)
                best_sets = (set1, set2)
    return best_gain, best_criteria, best_sets

"""
T11 fuction for printing the trees
"""
def printtree(tree,indent=''):
    # Is this a leaf node?
    if tree.results != None:
        return tree.results
    else:
    # Print the criteria
        print (indent,tree.col,' : ', tree.value,' ?')
        # Print the branches
        print (indent, 'T->', printtree(tree.tb,indent+' | '))
        print (indent, 'F->', printtree(tree.fb,indent+' | '))

"""
T12 function classify that allows to classify new objects.
It must return the dictionary that represents the partition of
the leave node where the object is classified.
"""
def classify(obj, tree):
    if tree.results == None:
        if isinstance(obj[tree.col], int) or isinstance(obj[tree.col], float):
            if obj[tree.col] >= tree.value: return classify(obj, tree.tb)
            else: return classify(obj, tree.fb)
        else:
            if obj[tree.col] == tree.value: return classify(obj, tree.tb)
            else: return classify(obj, tree.fb)
    else:
        return tree.results

"""
T13/14 function test that takes a test set and a training
set and computes the percentage of examples correctly
classified.poker-hand-testing-1000000.data
Show the quality of the classifier.
"""
def test_performance(testset, trainingset):
    tree = buildtree(trainingset)
    accuracy = 0.0
    for line in testset:
        obj = line.pop()
        result = classify(line,tree)
        if result.keys()[0] == obj:
            accuracy+=1
    return accuracy/len(testset)

"""
T15 other solutions
"""

"""
T16 a function that every pair of leaves with a common father check if their
union increases the entropy below a given threshold. If that
is the case, delete those leaves by joining their prototypes
in the father
"""

def prune(tree, threshold):
    if tree.tb.results is None: prune(tree.tb,threshold)
    if tree.fb.results is None: prune(tree.fb,threshold)

    if tree.tb.results is not None and tree.fb.results is not None:
        tb , fb = [], []
        for x, y in tree.tb.results.items():
            tb += [[x]] * y
        for x, y in tree.fb.results.items():
			 fb += [[x]] * y

        p = float(len(tb) / len(tb + fb))
        d = entropy(tb+fb) - p*entropy(tb) - (1-p)*entropy(fb)
        if d < threshold:
            tree.tb, tree.fb = None, None
            tree.results = unique_counts(tb + fb)

if __name__ == '__main__':
    tree_m = read('poker-hand-training-true-m.data',',')
    tree_xl = read('poker-hand-training-true-xl.data',',')
    print("-----------Tree build recursive-----------")
    printtree(buildtree(tree_m))
    print("-----------Tree build iterative-----------")
    printtree(buildtree_ite(tree_m))
    print("-------------Test Performance-------------")
    print (test_performance(read('poker-hand-testing-1000000.data',','), tree_xl))
    print ("------Test Performance [increased]-------")
    print (test_performance(read('poker-hand-testing-1000000.data',','), tree_m))
    tree = buildtree(tree_xl)
    print ("----Result of prunning with a 2.0 beta---")
    prune(tree, 1)
    printtree(tree)
