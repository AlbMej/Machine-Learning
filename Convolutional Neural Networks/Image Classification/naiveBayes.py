# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """

        # might be useful in the code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        bestConditionalProb = util.Counter()
        prior = util.Counter()
        featureFrequency = util.Counter()
        bestAccuracy = -1 

        n = len(trainingData)
        for i in range(n):
            image = trainingData[i]
            label = trainingLabels[i]
            prior[label] = prior.get(label, 0) + 1

            for feat, value in image.items():
                featureFrequency[(feat,label)] = featureFrequency.get((feat, label), 0) + 1
                if value > 0: 
                    bestConditionalProb[(feat, label)] = bestConditionalProb.get((feat, label), 0) + 1

        for k in kgrid: 
            #k_prior = k_conditionalProb = k_freq = util.Counter() 
            k_prior =  {}
            k_conditionalProb ={}
            k_freq ={}

            for key, val in bestConditionalProb.items():
                k_conditionalProb[key] = k_conditionalProb.get(key, 0) + val

            for key, val in prior.items():
                k_prior[key] = k_prior.get(key, 0) + val

            for key, val in featureFrequency.items():
                k_freq[key] = k_freq.get(key, 0) + val
            
            for label in self.legalLabels:
                for feat in self.features:
                    k_conditionalProb[(feat, label)] =  k_conditionalProb.get((feat, label), 0) + k
                    k_freq[(feat, label)] = k_freq.get((feat, label), 0) + (2 * k) 

            conditionals = k_conditionalProb.items()
            #k_prior.normalize()
            for x, count in conditionals:
                k_conditionalProb[x] = float(count) / k_freq[x]

            self.prior = k_prior
            self.conditionalProbs = k_conditionalProb

            predictions = self.classify(validationData)
            labels_len = len(validationLabels)
            accuracyCounts =  [predictions[i] == validationLabels[i] for i in range(labels_len)].count(True)

            percentCorrect = (float(accuracyCounts) / labels_len) * 100
            print("Performance on validation set for k=%f: (%.1f%%)" % (k, percentCorrect))
            if accuracyCounts > bestAccuracy:
                bestParamaters = (k_prior, k_conditionalProb, k)
                bestAccuracy = accuracyCounts

        self.prior, self.conditionalProbs, self.k = bestParamaters


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            #guesses.append(posterior.argMax())
            guesses.append(max(posterior, key = posterior.get))
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        #logJoint = util.Counter()

        logJoint = {}
        for label in self.legalLabels:
            logJoint[label] = math.log(self.prior[label])
            for feat, value in datum.items():
                logValue = self.conditionalProbs[feat,label]
                if value > 0: logJoint[label] += math.log(logValue)
                else: logJoint[label] += math.log(1 - logValue)
        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
                P(feature=1 | label1)/P(feature=1 | label2)

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []

        for feat in self.features:
            numerator = self.conditionalProbs[feat, label1] #.get((feat, label1), 0)
            denominator = self.conditionalProbs[feat, label2] #.get((feat, label2), 0)
            oddsRatio = numerator / denominator
            featuresOdds.append((oddsRatio, feat))
        
        featuresOdds.sort()
        truncated = featuresOdds[-100:]
        featuresOdds = [feat for (val, feat) in truncated]
        return featuresOdds