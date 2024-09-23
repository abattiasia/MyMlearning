
#110_lazypredict_Compare_Classification_Models.py

#07 lazypredict to Compare Classification Models  lib -->  using lazypredict lib:
## [lazypredict](https://github.com/shankarpandala/lazypredict)
##https://github.com/shankarpandala/lazypredict
##!pip install lazypredict
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
  print(models);