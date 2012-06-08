from matplotlib.pyplot import *
from numpy import *
from numpy.random import randint, normal
from scipy.stats.distributions import norm

def safelog10(x) :
    if x <= 0 :
        x = 1e-20
    return log10(x)

class NaiveBayes(object) :

    def __init__(self,X,C) :
        self.X = array(X)
        self.C = array(C)

        # determine distinct classes from C array
        self.classes = array(list(set(C)))
        self.class_inds = range(self.classes.size)
        self.class_labels = zeros(C.size)
        for i,c in zip(self.class_inds,self.classes) :
            self.class_labels[self.C==c] = i

        print self.class_labels
        print self.C.tolist()
        self.means = zeros((self.classes.size,X.shape[1]))
        self.stdvs = zeros((self.classes.size,X.shape[1]))
        self._estimate_feature_distributions()

    def _estimate_feature_distributions(self) :
        for c in self.class_inds :
            #print c, (self.class_labels==c).sum()
            self.means[c] = self.X[self.class_labels==c,].mean(axis=0)
            self.stdvs[c] = self.X[self.class_labels==c,].var(axis=0)

    def _get_class_posterior(self,x,c) :
        # calculate log probs for feature
        log_probs = zeros(x.size)
        for i,f in enumerate(x) :
            log_probs[i] = safelog10(norm.pdf(f,self.means[c,i],self.stdvs[c,i]))
        return log_probs

    def get_posteriors(self,x) :
        prior_likelihoods = zeros(self.classes.size)
        for c in self.class_inds :
            log_c_prob = safelog10(1.*(self.class_labels==c).sum()/self.C.size)
            log_probs = self._get_class_posterior(x,c)
            print log_c_prob, log_probs
            prior_likelihoods[c] = log_c_prob + log_probs.sum()
            print 'log p(C = %s|x = %s) == %.3f'%(self.classes[c],x,prior_likelihoods[c])

        # to avoid underflow
        #evidence = log10(pow(10,prior_likelihoods-prior_likelihoods.max()).sum())+prior_likelihoods.max()
        evidence = log10(pow(10,prior_likelihoods).sum())
        posteriors = prior_likelihoods/evidence
        return posteriors

    def get_predicted_class(self,x) :
        return self.classes[argmax(self.get_posteriors(x))]

def estimate_feature_distributions(X,C,classes) :
    means = zeros((classes.size,X.shape[1]))
    stdvs = zeros((classes.size,X.shape[1]))
    for c in classes :
        means[c] = X[C==c].mean(axis=0)
        stdvs[c] = X[C==c].var(axis=0)
    return means,stdvs

# toy example from wikipedia page
X = array([
    [6,180,12],
    [5.92,190,11],
    [5.58,170,12],
    [5.92,165,10],
    [5,100,8],
    [5.5,150,8],
    [5.42,130,7],
    [5.75,150,9]
   ])

"""
classes = array([0,1])
C = array([0,0,0,0,1,1,1,1])
nb = NaiveBayes(X,C)

x = array([6,130,8])
print nb.get_predicted_class(x)
"""

# randomly generated example
num_classes = 2
classes = range(num_classes)
class_means = randint(-10,10,(num_classes,2))
class_stdvs = abs(normal(size=(num_classes,2)))+1

# generate training samples
num_class_samples = randint(100,200,num_classes)
samples = []
sample_classes = []
colors = 'rgb'
print class_means, class_stdvs
print classes, num_class_samples
for c,num_c in zip(classes,num_class_samples) :
    print num_c, 'samples in class', c
    x,y = normal(class_means[c][0],class_stdvs[c][0],num_c), \
          normal(class_means[c][1],class_stdvs[c][1],num_c)
    scatter(x,y,marker='o',color=colors[c],alpha=0.75)
    samples.extend(zip(x,y))
    sample_classes.extend([colors[c]]*num_c)


# train the classifier
X = array(samples)
C = array(sample_classes)
nb = NaiveBayes(X,C)

test_samples = randint(-10,10,(10,2))
#test_samples = X[randint(0,len(sample_classes),10)]
for sample in test_samples :
    y_hat = nb.get_predicted_class(sample)
    scatter(sample[0],sample[1],60,marker='^',color=colors[y_hat],edgecolors='k')
show()

"""
# abalone dataset from UCI
abalone_labels = loadtxt('abalone.data',usecols=(0,),dtype='string',delimiter=',')
abalone_X = loadtxt('abalone.data',usecols=range(1,9),dtype='float',delimiter=',')

training_labels = abalone_labels[:-100]
training_X = abalone_X[:-100,]

nb = NaiveBayes(training_X,training_labels)

testing_labels = abalone_labels[-100:]
testing_X = abalone_X[-100:,]

correct = []
for l,x in zip(testing_labels,testing_X) :
    l_hat = nb.get_predicted_class(x)
    correct.append(l==l_hat)

print 'abalone sex prediction accuracy: %.2f'%(1.*sum(correct)/len(correct))
"""
