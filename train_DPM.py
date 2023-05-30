import numpy as np
from custom_DPM import DPM
import json
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

def hyperparams_tuning(model,x_train,y_train):
    params  =  {'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_estimators': [25, 50, 75, 100],
                'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
                'reg_lambda':[0,1],
                'reg_alpha':[30,40,50,60,70],
                'gamma':[1,2,5,8],}

    clf = GridSearchCV(estimator=model, 
                   param_grid=params, 
                   cv = 3,
                   n_jobs = -1,
                   verbose=1)
    
    clf.fit(x_train, y_train)
    print(clf.best_params_)

def check_accuracy(model,x_test,y_test):
    y_pred = model.predict(x_test)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred))+'\n')
    print(classification_report(y_test.ravel(), y_pred))

#get train data
f = open(r'./shipsnet.json')
dataset = json.load(f)
f.close()

#Extract the labels from the dataset
labels = np.array(dataset['labels']).reshape(len(dataset['labels']),1)
data = np.array(dataset['data']).astype('uint8')
img_size = 80
# train data for root filter
data_root_filter = data.reshape(-1,3,img_size,img_size).transpose([0,2,3,1])[:,:,:,:1]

# train data for part filters
data_part_filters = [data_root_filter[:,:-1,25:55],
                     data_root_filter[:,25:55,:-1]]

# initialize DPM
dpm = DPM()
part_filters = []
# get and save part_filters
for data in data_part_filters:
    part_filter = dpm.compute_part_filters(data)
    part_filters.append(part_filter)

dpm.save_part_filters(part_filters)

# train root filter 
hog_features = []
for image in data_root_filter:
    fd = hog(image, orientations=dpm.orientations, pixels_per_cell=(dpm.pix_per_cell_root,dpm.pix_per_cell_root),
             cells_per_block=(dpm.cells_per_block_root, dpm.cells_per_block_root), block_norm = 'L2')
    hog_features.append(fd)

X_train, X_test, y_train, y_test = train_test_split(np.array(hog_features), labels, test_size=0.1, random_state=42)
#hyperparams_tuning(dpm.models["0"],X_train,y_train)

dpm.models["0"].fit(X_train,y_train.ravel())

# check the accuracy
check_accuracy(dpm.models["0"],X_test,y_test)

# train part filters
for i in range(dpm.parts_count):
    hog_features = []
    for image in data_part_filters[i]:
        fd = hog(image, orientations=dpm.orientations, pixels_per_cell=(dpm.pix_per_cell_part,dpm.pix_per_cell_part),
                cells_per_block=(dpm.cells_per_block_part, dpm.cells_per_block_part), block_norm='L2')
        hog_features.append(fd)

    X_train, X_test, y_train, y_test = train_test_split(np.array(hog_features), labels, test_size=0.1, random_state=42)
    dpm.models[str(i+1)].fit(X_train,y_train.ravel())

    # check the accuracy
    check_accuracy(dpm.models[str(i+1)],X_test,y_test)

dpm.save_clfs()
