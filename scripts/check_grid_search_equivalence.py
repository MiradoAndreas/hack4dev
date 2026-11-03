"""Check optimized ParameterGrid scores against sklearn GridSearchCV on one sample."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from scripts.grid_transfer import fit_grid, candidates
from scripts.transfer_adaptation import CLASSES, target_forest, probabilities, inner_splits

class Blend(ClassifierMixin, BaseEstimator):
    def __init__(self,max_depth=None,max_features='sqrt',min_samples_leaf=2,source_weight=.5):
        self.max_depth=max_depth
        self.max_features=max_features
        self.min_samples_leaf=min_samples_leaf
        self.source_weight=source_weight
    def fit(self,x,y):
        self.target_=target_forest(10042, {'n_estimators':150,'class_weight':'balanced_subsample','n_jobs':4,
            'max_depth':self.max_depth,'max_features':self.max_features,'min_samples_leaf':self.min_samples_leaf})
        self.target_.fit(x.iloc[:,:-4],y)
        self.classes_=CLASSES
        return self
    def predict(self,x):
        return CLASSES[(self.source_weight*x.iloc[:,-4:].to_numpy()+
            (1-self.source_weight)*probabilities(self.target_,x.iloc[:,:-4])).argmax(axis=1)]

config=json.loads(Path('data/baseline/experiment.json').read_text())
table=pd.read_parquet('data/preprocessed_age_labels/amsterdam_supervised_features.parquet').set_index('location_id',drop=False)
members=pd.read_csv('data/baseline/adaptation_membership.csv')
ids=members.loc[members.fold.eq(0)&members.repetition.eq(0)&members.samples_per_class.eq(25),'location_id']
sample=table.loc[ids].reset_index(drop=True)
source=joblib.load('data/baseline/madrid_source_model.joblib')
features=config['feature_names']
sp=probabilities(source,sample[features])
_,optimized,_=fit_grid(sample,features,sp,config['rf_parameters'],10042)
x=sample[features].copy()
for i in range(4):x[f'source_probability_{i}']=sp[:,i]
# Singleton lists preserve exactly the optimized grid's candidate ordering.
grid=[{k:[v] for k,v in p.items()} for p in candidates()]
search=GridSearchCV(Blend(),grid,cv=inner_splits(sample.age_class.to_numpy(),sample.spatial_group.to_numpy(),10042),
    scoring=make_scorer(f1_score,labels=CLASSES,average='macro',zero_division=0),refit=False,error_score='raise')
search.fit(x,sample.age_class)
for split in range(3):
    expected=optimized.loc[optimized.inner_fold.eq(split)].sort_values('candidate').macro_f1.to_numpy()
    np.testing.assert_allclose(search.cv_results_[f'split{split}_test_score'],expected,rtol=0,atol=1e-14)
print('PASS: all 96 optimized candidate/fold F1 scores match sklearn GridSearchCV on the 25/class sample.')
