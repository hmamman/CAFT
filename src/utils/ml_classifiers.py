from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    # 'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=100),
    'svm': SVC(probability=True, kernel="rbf", gamma="scale"),
    'lr': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000),
    'nb': GaussianNB(var_smoothing=1e-9)
}

# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
nb_clf = GaussianNB()


# ensemble above classifiers for majority voting
voting_emseble_classifier = VotingClassifier(
    estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', nb_clf)],
    voting='soft'
)

# set a pipeline to handle the prediction process
ensemble_pipeline = Pipeline([('scaler', StandardScaler()),
                ('ensemble', voting_emseble_classifier)])


'''
Bank          0.8978
Census        0.8429
Credit        0.7833
Compas        0.8223
Meps          0.8708
'''
