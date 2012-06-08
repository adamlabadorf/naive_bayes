from numpy import *
from scipy.stats.distributions import norm

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

        self.means = zeros((self.classes.size,X.shape[1]))
        self.stdvs = zeros((self.classes.size,X.shape[1]))
        self._estimate_feature_distributions()

    def _estimate_feature_distributions(self) :
        for c in self.class_inds :
            print c, (self.class_labels==c).sum()
            self.means[c] = self.X[self.class_labels==c,].mean(axis=0)
            self.stdvs[c] = self.X[self.class_labels==c,].var(axis=0)

    def _get_class_posterior(self,x,c) :
        # calculate log probs for feature
        log_probs = zeros(x.size)
        for i,f in enumerate(x) :
            log_probs[i] = log10(norm.pdf(f,self.means[c,i],self.stdvs[c,i]))
        return log_probs

    def get_posteriors(self,x) :
        prior_likelihoods = zeros(self.classes.size)
        for c in self.class_inds :
            log_c_prob = log10(1.*(self.class_labels==c).sum()/self.C.size)
            log_probs = self._get_class_posterior(x,c)
            prior_likelihoods[c] = log_c_prob + log_probs.sum()

        # to avoid underflow
        evidence = log10(pow(10,prior_likelihoods-prior_likelihoods.max()).sum())+prior_likelihoods.max()
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

classes = array([0,1])
C = array([0,0,0,0,1,1,1,1])
nb = NaiveBayes(X,C)

x = array([6,130,8])
print nb.get_predicted_class(x)

# abalone dataset from UCI
abalone_labels = loadtxt('abalone.data',usecols=(0,),dtype='string',delimiter=',')
abalone_X = loadtxt('abalone.data',usecols=range(1,9),dtype='float',delimiter=',')

training_labels = abalone_labels[:-100]
training_X = abalone_X[:-100,]

nb = NaiveBayes(training_X,training_labels)

testing_labels = abalone_labels[-100:]
testing_X = abalone_X[-100:,]

for l,x in zip(testing_labels,testing_X) :
    l_hat = nb.get_predicted_class(x)
    print l,l_hat
