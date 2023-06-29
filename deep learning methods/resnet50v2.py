# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 23:20:42 2023

@author: Tiago
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 22:00:43 2023

@author: Tiago
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
#from Dataframe_traditional_methods import New_DataFrame
import sys
import numpy as np
from matplotlib.image import imread
from PIL import Image, ImageOps
from skimage import feature
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold,GridSearchCV,RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import cross_validate
from keras.applications import ResNet50V2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,Input, Flatten
import tensorflow.keras as keras
import tensorflow_addons as tfa
from skimage.filters import threshold_otsu, gaussian
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Model, Sequential
from sklearn.metrics import classification_report, roc_curve, auc
import keras.backend as K
from sklearn.metrics import roc_auc_score
import cv2
from skimage.color import gray2rgb
import os
from tensorflow.keras import regularizers


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
    except RuntimeError as e:
        print(e)
        
sys.path.append('metrics')

data_dir =("insert_path")
train_path = ("insert_path")
test_path = ("insert_path")
main_dir = os.listdir(data_dir)
print(main_dir)
print()
categories = ["CNV", "DME", "DRUSEN", "NORMAL"]


# Set the image size and batch size
img_size = (224, 224)
BATCH_SIZES = [16,32,64]
num_classes=4

# Define the preprocessing function
 
def preprocessing(img):
    
        # Convert to PIL Image
        img = Image.fromarray(np.uint8(img))
        # Resize image
        img = img.resize(img_size)
        img_arr = np.array(img).astype('float64')  
        return img_arr
        
###########################Create Dataframe - train and test ####################################
    
X_train = []
y_train = []
patient_ids_train = []
img_arrays = []
class_train =[]

for label, class_dir in enumerate(categories):  
    for image_path in os.listdir(os.path.join(train_path, class_dir)):
       patient_id = image_path.split('-')[1]
       img_path = os.path.join(train_path, class_dir, image_path)
       img = imread(img_path)
       
       X_train.append(os.path.join(train_path, class_dir, image_path))
       y_train.append(label)
       class_train.append(class_dir)
       patient_ids_train.append(patient_id)

  
df_train = pd.DataFrame({'Image': X_train, 'Category': y_train, 'Patient ID': patient_ids_train,'Class_Name': class_train  })
                         #, 'Fotografia': img_arrays})
print("-", categories[0], " ", y_train.count(0))
print("-" ,categories[1], " ",y_train.count(1))
print("-", categories[2], " ",y_train.count(2))
print("-", categories[3], " ",y_train.count(3))
print()
print("")

#Test Data

X_test=[]
y_test=[]
patient_ids_test=[]
img_arrays = []
class_test =[]

for label, class_dir in enumerate(categories):

    for image_path in os.listdir(os.path.join(test_path, class_dir)): # os.listdir returns a list of all the files and directories in a specified path.     

        patient_id = image_path.split('-')[1] # Extract the patient ID from the image name
        img_path = os.path.join(test_path, class_dir, image_path)
        img = imread(img_path)
 
        X_test.append(os.path.join(test_path, class_dir, image_path))
        y_test.append(label)
        class_test.append(class_dir)
        patient_ids_test.append(patient_id)

        
df_test= pd.DataFrame({'Image':X_test, 'Category': y_test, 'Patient ID': patient_ids_test, 'Class_Name': class_test})
#, 'Fotografia': img_arrays})      
print("Test DataLoad: done")
print("-", categories[0], " ", y_test.count(0))
print("-" ,categories[1], " ",y_test.count(1))
print("-", categories[2], " ",y_test.count(2))
print("-", categories[3], " ",y_test.count(3))


feature_x= np.array(df_train["Image"])
target_y= np.array(df_train["Category"])
groups= np.array(df_train["Patient ID"])
target_y = to_categorical(target_y, num_classes=len(categories)) 

feature_x_test= np.array(df_test["Image"])
target_y_test= np.array(df_test["Category"])
target_y_test = to_categorical(target_y_test, num_classes=len(categories))


############################### 2: Data augmentation##################################

datagen_train = ImageDataGenerator(preprocessing_function=preprocessing, rescale=1./255)
                                   #rotation_range=15,
                                    #zoom_range=[0.9, 1.1],
                                    #height_shift_range=0.05,
                                    #width_shift_range=0.05,
                                   # brightness_range= (1.4, 2))


datagen_val= ImageDataGenerator(preprocessing_function=preprocessing,rescale=1./255) 

datagen_test= ImageDataGenerator(preprocessing_function=preprocessing,rescale=1./255) 

####################################### 3: Cross Validation  ###################################################
num_folds=5
target_y_decoded = np.argmax(target_y, axis=1)
stratified_group_kfold = StratifiedGroupKFold(n_splits=num_folds)
splits = stratified_group_kfold.split(feature_x, target_y.argmax(1), groups)
# Compute class weights
class_weights = compute_class_weight('balanced', 
                                                  classes= np.unique(target_y_decoded), 
                                                  y= target_y_decoded)
class_weights= dict(zip(np.unique(target_y_decoded), class_weights))
print(f"Class weights: {class_weights}")


# Split the data into training and validation sets using StratifiedGroupKFold
class_weights_list= list(class_weights.values())
all_metrics= [tf.keras.metrics.CategoricalAccuracy(name='accuracy'), keras.metrics.AUC(name='auc',multi_label=False) ,Precision(num_classes=4), Recall(num_classes=4), tfa.metrics.F1Score(name='f1-score',num_classes=4, average='macro'),WeightedAccuracy(num_classes=4, class_weights=class_weights_list),tfa.metrics.FBetaScore(name='fbeta',num_classes=4, beta=2.0, average='macro')]

##########################   4: MODEL   #########################################################
# Create Model - ResNet50
pretrained_model= ResNet50V2(input_shape= (224, 224, 3),
              weights='imagenet',
             include_top= False,
             ) 

for layer in pretrained_model.layers[:154]: #all block 5 unfrooze
    layer.trainable = False

for i, layer in enumerate(pretrained_model.layers):
    print(i,layer.name,"-", layer.trainable)

resnet = pretrained_model


def resnet_new_layers(pretrained_model,num_classes): #para adicionar camadas
    top_model=pretrained_model.output
    top_model= GlobalAveragePooling2D()(top_model)
    top_model = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.001))(top_model)
    top_model= BatchNormalization()(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001))(top_model)
    top_model= BatchNormalization()(top_model)
    top_model = Dense(100, activation='relu',kernel_regularizer=regularizers.l2(0.001))(top_model)
  
    top_model= Dense(num_classes, activation= 'softmax')(top_model) #MUDAR ISTO

    return top_model


best_val_fbeta=0
for fold, (train_idx, val_idx) in enumerate(splits):
    # Create the new training and Validation DataFrame
    fold_train_df = df_train.iloc[train_idx]
    fold_val_df = df_train.iloc[val_idx]
    model=[]
    FC_Head=[]
    checkpoint=[]
    earlystop=[]
    reduceLR=[]
    callbacks_list=[]

    for batch_size in BATCH_SIZES:
            print(f'Training for Fold {fold} ------ batch size: {batch_size}')
            checkpoint = ModelCheckpoint(filepath=f"/insert_path", monitor='val_fbeta', mode='max', verbose=1,save_best_only=True)   
            min_delta=0.000001
            earlystop =EarlyStopping(monitor='val_fbeta',min_delta=  min_delta,patience=7,mode='max')
            reduceLR= ReduceLROnPlateau(monitor='val_fbeta',factor=0.2,patience=3,mode='max',min_delta= min_delta, min_lr= 0.000001)
            callbacks_list = [earlystop, checkpoint,reduceLR]
            FC_Head= resnet_new_layers(pretrained_model,num_classes)
            model=Model(inputs=pretrained_model.input, outputs= FC_Head)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy', #- ver isto
                          metrics=all_metrics)


            train_generator = datagen_train.flow_from_dataframe(
            dataframe=fold_train_df,
            directory=train_path,
            x_col='Image',
            y_col='Class_Name',
            target_size=img_size,
            class_mode='categorical',
            batch_size=batch_size,
            color_mode='rgb',
            shuffle=True)
            

            val_generator = datagen_val.flow_from_dataframe(
            dataframe=fold_val_df,
            directory=train_path,
            x_col='Image',
            y_col='Class_Name',
            target_size=img_size,
            class_mode='categorical',
            color_mode='rgb',
            batch_size=batch_size,
            shuffle=False)
              
           
            # Fit the model - no epochs
            history = model.fit(
            train_generator,
            validation_data=val_generator,
            verbose=0,
            epochs=30,
            callbacks= callbacks_list,
            class_weight=class_weights,
            workers=8,
            use_multiprocessing=True,
        )
    
     
            train_loss, train_accuracy, train_auc, train_precision, train_recall, train_f1_score, train_balanced_accuracy,train_fbeta = model.evaluate(train_generator, batch_size=batch_size ,verbose=0, workers=4,use_multiprocessing=True)
            val_loss, val_accuracy, val_auc, val_precision, val_recall, val_f1_score, val_balanced_accuracy, val_fbeta = model.evaluate(val_generator,batch_size=batch_size,verbose=0)
    
        
            #best model saved during training
            
            if val_fbeta > best_val_fbeta:
                
                batch_metrics_list=[]
                results_metrics_list=[]
                batch_metrics = {}
                batch_metrics_test = {}
                
                best_val_fbeta= val_fbeta
                best_model=model
                
                batch_metrics['Batch_size'] = batch_size
                batch_metrics['Fold'] = fold
                batch_metrics['Train_Loss'] = train_loss
                batch_metrics['Train_Accuracy'] = train_accuracy
                batch_metrics['Train_AUC'] = train_auc
                batch_metrics['Train_Precision'] = train_precision
                batch_metrics['Train_Recall'] = train_recall
                batch_metrics['Train_F1_Score'] = train_f1_score
                batch_metrics['Train_Balanced_Accuracy'] = train_balanced_accuracy
                batch_metrics['Train_Fbeta'] = train_fbeta
               
                
                batch_metrics['Val_Loss'] = val_loss
                batch_metrics['Val_Accuracy'] = val_accuracy
                batch_metrics['Val_AUC'] = val_auc
                batch_metrics['Val_Precision'] = val_precision
                batch_metrics['Val_Recall'] = val_recall
                batch_metrics['Val_F1_Score'] = val_f1_score
                batch_metrics['Val_Balanced_Accuracy'] = val_balanced_accuracy
                batch_metrics['Val_Fbeta'] = val_fbeta
                            
                batch_metrics_list.append(batch_metrics)           
                plt.subplot()
                plt.rcParams['figure.figsize'] = (6.0, 4.0)
                plt.title('Baseline Model Fbeta-Score')
                plt.plot(history.history['fbeta'])
                plt.plot(history.history['val_fbeta'])
                plt.ylabel('Fbeta-Score')
                plt.xlabel('Epochs')
                plt.ylim(0.5, 1) # set y-limit between 0 and 1
                plt.legend(['Training Fbeta-Score','Validation Fbeta-Score'])
                plt.savefig("insert_path.png", transparent=False, bbox_inches='tight', dpi=400)
                plt.show()
                plt.clf()
                
                                    
                plt.subplot()
                plt.title('Baseline Model Loss')
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.ylabel('Loss')
                plt.xlabel('Epochs')
                plt.ylim(0, 1) # set y-limit between 0 and 1
                plt.legend(['Training Loss','Validation Loss'])
                plt.savefig("insert_path.png", transparent=False, bbox_inches='tight', dpi=400)
                plt.show()
                plt.clf()

               
                test_generator = datagen_test.flow_from_dataframe(
                dataframe=df_test,
                directory=test_path,
                x_col='Image',
                y_col='Class_Name',
                target_size=img_size,
                class_mode='categorical',
                color_mode='rgb',
                batch_size=batch_size,
                shuffle=False)
                # Get the true class labels for the test set
                test_classes = np.argmax(target_y_test, axis=1)
                
              
                # Evaluate the best model on the test set
                
                test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1_score, test_balanced_accuracy,test_fbeta = model.evaluate(test_generator, verbose=0)
                batch_metrics_test['Batch_size'] = batch_size
                batch_metrics_test['Fold'] = fold
                batch_metrics_test['Test_Loss'] = test_loss
                batch_metrics_test['Test_Accuracy'] = test_accuracy
                batch_metrics_test['Test_AUC'] = test_auc
                batch_metrics_test['Test_Precision'] = test_precision
                batch_metrics_test['Test_Recall'] = test_recall
                batch_metrics_test['Test_F1_Score'] = test_f1_score
                batch_metrics_test['Test_Balanced_Accuracy']=test_balanced_accuracy
                batch_metrics_test['Test_Fbeta']=test_fbeta
                
                test_predictions = model.predict(test_generator,1000 // batch_size+1) #1000 is the size of test dataset 
                y_pred = np.argmax(test_predictions, axis=1)
                cm = confusion_matrix(test_classes, y_pred)
                batch_metrics_test['confusion_matrix'] = [cm]
                report_dict = classification_report(test_classes, y_pred)
                batch_metrics_test['classification_report'] = [report_dict]
            
            
                #One vs Rest ROC Curves
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(len(categories)):
                    fpr[i], tpr[i], _ = roc_curve(target_y_test[:, i], test_predictions[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
             
                # Compute micro-average ROC curve and AUC
                fpr["micro"], tpr["micro"], _ = roc_curve(target_y_test.ravel(), test_predictions.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
             
                # Compute macro-average ROC curve and AUC
               
                for i in range(len(categories)):
                    fpr[i], tpr[i], _ = roc_curve(target_y_test[:, i], test_predictions[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                fpr_grid = np.linspace(0.0, 1.0, 2000)
                        # Interpolate all ROC curves at these points
                mean_tpr = np.zeros_like(fpr_grid)
                
                for i in range(len(categories)):
                    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    
                        # Average it and compute AUC
                mean_tpr /= len(categories)
                
                fpr["macro"] = fpr_grid
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
                #Plot all OvR ROC curves together
                from itertools import cycle
                
                fig, ax = plt.subplots(figsize=(6, 6))
                
                plt.plot(
                    fpr["micro"],
                    tpr["micro"],
                    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
                    color="deeppink",
                    linestyle=":",
                    linewidth=4,
                )
                
                plt.plot(
                    fpr["macro"],
                    tpr["macro"],
                    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
                    color="navy",
                    linestyle=":",
                    linewidth=4,
                )
                
                colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
                for class_id, color in zip(range(len(categories)), colors):
                    RocCurveDisplay.from_predictions(
                        target_y_test[:, class_id],
                        test_predictions[:, class_id],
                        name=f"ROC curve for {categories[class_id]}",
                        color=color,
                        ax=ax,
                    )
                
                plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
                plt.xlim([0, 1])
                plt.ylim([0.0, 1])
                plt.xlabel("False Positive Rate (1 - Specificity)")
                plt.ylabel("True Positive Rate (Sensitivity)")
                plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
                plt.legend()
                plt.show()
                fig.savefig("insert_path.png")
             
             
                batch_metrics_test['Plot'] = Image.open("insert_path.png")    
                results_metrics_list.append(batch_metrics_test)
                # clear the session
                K.clear_session()
                plt.clf()
                del test_generator
    
            del FC_Head
            del model
            del callbacks_list 
            model=[]
            FC_Head=[]
            checkpoint=[]
            earlystop=[]
            reduceLR=[]
            callbacks_list=[]
                
    # concatenate the batch_metrics dictionaries for all folds into a dataframe
df = pd.DataFrame.from_records(batch_metrics_list)
df_results = pd.DataFrame.from_records(results_metrics_list)

print("---------------------Train-----------------------------------")
print()
print("Results obtained:")
print()
print("Batch size: ", batch_metrics['Batch_size'])
print("Fold: ",batch_metrics['Fold'])
print("Loss: ",batch_metrics['Train_Loss'])
print("Accuracy: ",batch_metrics['Train_Accuracy'])
print("Precision: ",batch_metrics['Train_Precision'] )
print("Recall: ",batch_metrics['Train_Recall'] )
print("F1 Score: ",batch_metrics['Train_F1_Score'] )
print("Weighted Accuracy: ",batch_metrics['Train_Balanced_Accuracy'])
print("F beta: ",batch_metrics['Train_Fbeta'])

print("---------------------Validation-----------------------------------")
print()
print("Results obtained:")
print()
print("Batch size: ", batch_metrics['Batch_size'])
print("Fold: ",batch_metrics['Fold'])
print("Loss: " ,batch_metrics['Val_Loss'] )
print ("Accuracy: ",batch_metrics['Val_Accuracy'])
print("Precision: ", batch_metrics['Val_Precision'])
print("Recall: ",batch_metrics['Val_Recall'])
print("F1 Score: ",batch_metrics['Val_F1_Score'])
print("Weighted Accuracy: ",batch_metrics['Val_Balanced_Accuracy'])
print("F beta: ",batch_metrics['Val_Fbeta'])

print("---------------------Test-----------------------------------")
print()
print("Results obtained:")
print()
print("Batch size: ", batch_metrics_test['Batch_size'])
print("Fold: ",batch_metrics_test['Fold'])
print("Loss: ",batch_metrics_test['Test_Loss'])
print("Accuracy: ",batch_metrics_test['Test_Accuracy'])
print("AUC: ",batch_metrics_test['Test_AUC'] )
print("Precision: ",batch_metrics_test['Test_Precision'] )
print("Recall: ",batch_metrics_test['Test_Recall'] )
print("F1 Score: ",batch_metrics_test['Test_F1_Score'] )
print("Fbeta Score: ",batch_metrics_test['Test_Fbeta'] )
print()
print("Confusion Matrix")
print(batch_metrics_test['confusion_matrix'] )
print()
print("Classification report")
print("2")
print("Batch size: ",batch_metrics_test['classification_report'])
print()
print("Batch size: ",batch_metrics_test['Plot'])


