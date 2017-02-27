from sklearn import tree

def f2Rule(f2):
    return (1 if f2 is 'smooth' else 0)

def labelRule(l):
    return (1 if l is 'apple' else 0)

def reverseLabelRule(l):
    return ('apple' if 1 else 'orange')

def featureProcessor (features):
    return [[f1,f2Rule(f2)]for f1,f2 in features]

def labelProcessor(labels,rule=labelRule):
    return [rule(l) for l in labels]

def classify(features,labels,target):
    assert(len(target) is 1)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(featureProcessor(features), labelProcessor(labels))
    return labelProcessor(clf.predict(featureProcessor(target)),rule=reverseLabelRule)

if __name__ == '__main__':
    features = [[140,'smooth'],[130,'smooth'],[150,'bumpy'],[170,'bumpy']]
    labels = ['apple','apple','orange','orange']

    print classify(features,labels,[[150,'smooth']])
