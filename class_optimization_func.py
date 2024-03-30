# Classification / Optimization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.integrate import trapz, simps
# from functions_jones_1_V2 import *
from importlib import reload
import seaborn as sns
import os
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import scipy.stats as stats
from itertools import product
warnings.simplefilter('ignore')
import transmission 

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
#from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
#from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from statsmodels.multivariate.manova import MANOVA
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec


import wandb
from wandb.keras import WandbCallback

import matplotlib.patches as patches
#from pandas.io.json import json_normalize
from matplotlib.path import Path
import matplotlib.patches as patches

import plotly.express as px



def plot_box_intensities(df_result_only_intensity, title):

    melted_df = pd.melt(df_result_only_intensity, id_vars = 'label', var_name='Sub Pixel', value_name='Intensitaet' )
    fig = plt.figure()
    sns.boxplot(data = melted_df, x = 'Sub Pixel',  y = 'Intensitaet',  hue = 'label',  linewidth=.75)
    plt.title(title)
    plt.show()

    
def upsample(X, y):
    # verwende den RandomOverSampler aus dem imblearn Paket, um den Datensatz zu balancieren
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def downsample(X, y):
    # verwende den RandomOverSampler aus dem imblearn Paket, um den Datensatz zu balancieren
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

def plot_correlation(df_result, encoded_Y): 
    # run correlation matrix and plot
    df_correlation = df_result[['intensity_1', 'intensity_2', 'intensity_3', 'intensity_4']]
    df_correlation['Label'] = encoded_Y
    f, ax = plt.subplots(figsize=(10, 8))
    print(df_correlation)
    corr = df_correlation.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)

def variance_analysis_manova(df_intensitys, df_result_intensity):
    #Anova
    manova = MANOVA.from_formula('intensity_1 + intensity_2 + intensity_3 + intensity_3 ~ label', data=df_result_intensity)
    result = manova.mv_test()
    x = pd.DataFrame((manova.mv_test().results['label']['stat']))
    print(x)
    #print(x)
    p_value = x.iloc[1, 4]
    #print(p_value)
    print('Manova results')
    print()
    print(p_value)
    if p_value < 0.05:
        print('p_value is smaller than 0.05 and therefore indicates, that there is a significant difference between the labels') 
    else: 
        print('p_value is bigger than 0.05 and therefore indicates, that there is no significant difference between the labels') 
    print()
    
    #print(result.summary())
    return p_value

def plot_manova_results(df_result_simulation): 
    x = df_result_simulation['thickness_waveplate']
    y = df_result_simulation['rotation_waveplate']
    z = df_result_simulation['manova_pscore']

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    #ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')

    # Set logarithmic scale for z-axis
    #ax.set_zscale('log')

    # Plot lines from points to z-axis
    #for i in range(len(x)):
        #ax.plot([x[i], x[i]], [y[i], y[i]], [1, z[i]], c='b')

    # Set labels and title
    ax.set_xlabel('Thickness')
    ax.set_ylabel('Rotation')
    ax.set_zlabel('Manova Result')
    ax.set_title('3D Scatter Plot with Logarithmic Z-axis')
    #ax.zaxis.set_tick_params(width=1, length=4, which='major', direction='out')

    # Set log scale for z-axis
    #ax.set_zscale('log')
    #bubble_size = x*y/10
    z = np.log(z)
    cbar = plt.colorbar(ax.scatter(x, y, z, c=z, cmap='viridis', s=30, alpha=1))
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], c='b')
    cbar.set_label('Y values')
    
    plt.show()
    
    
    # Create 3D plot
    plt.figure()
    
    # Set log scale for z-axis
    
    bubble_size = -z*10
    print(bubble_size)
    sc = plt.scatter(x, y,s=bubble_size, c=z, cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('-log(pscore)')
    plt.ylabel('Y values')
    
    plt.show()


def variance_analysis_oneway(df_intensity, df_result_intensity, dataset_object):
    #Anova
    labels_smaller_threshold = []
    labels_bigger_threshold = []
    p_values_array = []

    print('Oneway Anova results')
    print()
    for series_name, series in df_intensity.items():
        # print(series_name)
        # print(series)

        # Assuming 'data' is a DataFrame with 'label' and 'light_intensity' columns
        data_label1 = df_result_intensity[df_result_intensity['label'] == dataset_object.sample_name[0]][series_name]
        data_label2 = df_result_intensity[df_result_intensity['label'] == dataset_object.sample_name[1]][series_name]

        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(data_label1, data_label2)
        p_values_array.append(p_value)
        print('p_value for ' + series_name + ' ' + str(p_value))
        if p_value <= 0.05:
            labels_smaller_threshold.append(series_name)
        else:
            labels_bigger_threshold.append(series_name)
        
    mean_anova = sum(p_values_array)/len(p_values_array)
        # # Print the results
        # print('oneway Anova for ' + series_name)
        # print("P-Value:", p_value)
        # print("F-Statistic:", f_statistic)
    
    print('p_value is smaller than the threshhold 0.05 which indicates a significant difference between the labels for the features: ')
    print(labels_smaller_threshold)
    print('p_value is bigger than the threshhold 0.05 which indicates no significant difference between the labels for the features: ')
    print(labels_bigger_threshold)
    print()
    return mean_anova

def pca_function(X_values, df):
    
    # ??? nur kopiert 
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    pca.fit(X_values)

    # print the explained variances
    #print("Variances (Percentage):")
    #print(pca.explained_variance_ratio_ * 100)
    #print()
    #print("Cumulative Variances (Percentage):")
    #print(pca.explained_variance_ratio_.cumsum() * 100)
    print()

    # plot a scree plot
    components = len(pca.explained_variance_ratio_) #if components is None else components
    # plt.plot(range(1,components+1), np.cumsum(pca.explained_variance_ratio_ * 100))
    # plt.xlabel("Number of components")
    # plt.ylabel("Explained variance (%)")
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 0.85)
    pca.fit(X_values)
    #print("Cumulative Variances (Percentage):")
    #print(np.cumsum(pca.explained_variance_ratio_ * 100))
    components = len(pca.explained_variance_ratio_)
    #print(f'Number of components: {components}')

    # Make the scree plot
    # plt.plot(range(1, components + 1), np.cumsum(pca.explained_variance_ratio_ * 100))
    # plt.xlabel("Number of components")
    # plt.ylabel("Explained variance (%)")
    pca_components = abs(pca.components_)
    #print(pca_components)

    print('Top 4 most important features in each component')
    print('===============================================')
    for row in range(pca_components.shape[0]):
        # get the indices of the top 4 values in each row
        temp = np.argpartition(-(pca_components[row]), 3)
        # sort the indices in descending order
        indices = temp[np.argsort((-pca_components[row])[temp])][:4]
        # print the top 4 feature names
        print(f'Component {row}: {df.columns[indices].to_list()}')
    print()

    #Transforming all the 30 Columns to the 6 Principal Components
    X_pca = pca.transform(X_values)
    #print(X_pca.shape)
    #print(X_pca)
    
    
def svm_classifier(X_train, X_test, y_train, y_test, kernel, degree, C, dataset_object):

    model_svm = svm.SVC(kernel=kernel, degree = degree, C=C)
    # Train the classifier on the training data
    model_svm.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = model_svm.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(y_pred, y_test)
    f1 = f1_score(y_test, y_pred)
    #print(y_test)
    #print(y_pred)
    print()
    print(kernel + ' Support Vector Machine results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy, f1





def logistic_regression(X_train, X_test, y_train, y_test, dataset_object):

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print()
    print('Logistic Regression results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)

    return accuracy, f1


def lighbm_classifier(X_train, X_test, y_train, y_test, dataset_object):
    # surpess warnings 
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    # define the model
    model = LGBMClassifier(force_col_wise=True)
    # fit the model on the whole dataset
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print()
    print('Light Gradient Boosted Machine (LightGBM) Ensemble results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy, f1

def random_Forest(X_train, X_test, y_train, y_test, dataset_object):
    
    # define the model
    model_rf = RandomForestClassifier(max_depth=2, random_state=0)
    # fit the model /train it
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print()
    print('Random Forest results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy, f1
    
    
def xgboost(X_train, X_test, y_train, y_test, dataset_object):
    # define the model
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
        # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print()
    print('Extreme Gradient Boosting (XGBoost) results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy, f1
    
def multi_svc(X_train, X_test, y_train, y_test, dataset_object): 
    model = SVC()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred)
    
    print()
    print('Multi SVC Classificator results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy# , f1


def random_Forest_multi(X_train, X_test, y_train, y_test, dataset_object):
    
    # define the model
    model_rf = RandomForestClassifier(max_depth=2, random_state=0)
    # fit the model /train it
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred)

    print()
    print('Random Forest results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy

def lighbm_classifier_multi(X_train, X_test, y_train, y_test, dataset_object):
    # surpess warnings 
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    # define the model
    model = LGBMClassifier(force_col_wise=True)
    # fit the model on the whole dataset
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred)

    print()
    print('Light Gradient Boosted Machine (LightGBM) Ensemble results')
    print()
    print('Accuracy: %.2f' % (accuracy*100))
    print(confusion_matrix(y_test, y_pred))
    print('y_test')
    print(y_test)
    print('y_pred')
    print(y_pred)
    
    return accuracy


# def multi_lightgbm(X_train, X_test, y_train, y_test, dataset_object): 
    
#     warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

#     # define the model
#     model = LGBMClassifier(force_col_wise=True)
#     # fit the model on the whole dataset
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     # Evaluate the model
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     print()
#     print('Light Gradient Boosted Machine (LightGBM) Ensemble results')
#     print()
#     print('Accuracy: %.2f' % (accuracy*100))
#     print(confusion_matrix(y_test, y_pred))
#     print('y_test')
#     print(y_test)
#     print('y_pred')
#     print(y_pred)
    
#     return accuracy, f1







# def nn_model(X_train, X_test, y_train, y_test, epoch, dataset_object): 
#     model = Sequential()
#     if dataset_object.normalise_result_intensity == True:
#         model.add(Dense(12, input_shape=(8,), activation='relu'))
#     else:
#         model.add(Dense(12, input_shape=(4,), activation='relu'))
        
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     #model.add(Dense(120, activation='relu'))
#     model.add(Dense(24, activation='relu'))
#     model.add(Dense(12, activation='relu'))
#     model.add(Dense(units=1, activation='sigmoid'))
#     #opt = keras.optimizers.legacy.Adam()
    
#     opt ='Adam'
#     print(opt)
#     model.compile(loss='binary_crossentropy', optimizer=opt , metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=epochs, batch_size=5, verbose=0)
#     y_pred = model.predict(X_test)
#     test_loss, accuracy = model.evaluate(X_test, y_test)

#     print('run with epochs = ' + str(epochs))
#     print('Accuracy: %.2f' % (accuracy*100))
#     print()

#     return accuracy, y_pred[:, 0]


# def nn_classifier(X_train, X_test, y_train, y_test, dataset_object):
#     accuracy_safe = 0
#     epochs_pos = [100, 300, 600, 1000, 1500]
#     k = 0
    
#     print()
#     print('Neural Network Classification')
#     print()
    
#     for epochs in epochs_pos:
#         accuracy, y_pred = nn_model(X_train, X_test, y_train, y_test, epochs, dataset_info)
#         if accuracy > accuracy_safe:
#             accuracy_safe = accuracy
#             epochs_safe = epochs
#             #f1_safe = f1
#             y_pred_safe = y_pred
#             y_pred_probability = y_pred
#         else:
#             break

#     y_pred_safe[y_pred_safe<0.5] = 0
#     y_pred_safe[y_pred_safe>=0.5] = 1
#     f1 = f1_score(y_test, y_pred_safe)
#     print(confusion_matrix(y_test, y_pred_safe))
#     print(y_test, y_pred_safe, y_pred_probability)

#     return accuracy_safe, epochs_safe, f1


def nn_classifier_hyp_tun(X_train, X_test, y_train, y_test, dataset_object): 
        
    global X_train_hyp
    global y_train_hyp
    global X_test_hyp
    global y_test_hyp
    y_test_hyp = y_test
    X_test_hyp = X_test
    X_train_hyp = X_train
    y_train_hyp = y_train
    print(y_train)
    global score_acc
    score_acc = make_scorer(accuracy_score)
    
    params_nn2 ={
        'neurons': (10, 120),
        'activation':(0, 4),
        'optimizer':(0,7),
        'learning_rate':(0.001, 1),
        'batch_size':(10, 100),
        'epochs':(10, 500),
        'layers1':(2,3),
        'layers2':(2,3),
        'dropout':(0,1),
        'dropout_rate':(0,0.3),

    }
    #fixed_params = {'X_train': X_train, 'y_train': y_train}
    # Run Bayesian Optimization
    print("NaN values in y_train:", np.isnan(y_train).any())
    print("NaN values in y_test:", np.isnan(y_test).any())
    nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)
    nn_bo.maximize(init_points=15, n_iter=4)
    #x = np.linspace(0, 500, 100).reshape(-1, 1)
    #y = 1

    results = nn_bo.res
    params = nn_bo.space.params
    # Creating DataFrames for results and parameters
    df_results = pd.DataFrame(results)
    df_params = pd.json_normalize(params)

    # Combining results and parameters
    df_dict = pd.concat([ df_params, df_results], axis=1)

    df_params = pd.DataFrame(df_dict['params'].tolist())
    df_combined = pd.concat([df_dict.drop(columns='params'), df_params], axis=1)

    # https://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    fig = px.parallel_coordinates(df_combined, color="target", labels={
                "activation": "Aktivierung", "batch_size": "Batch Groesse",
                "dropout": "Dropout", "dropout_rate": "Dropout Rate", "layers1": "Anzahl Schichten 1", "layers2": "Anzahl Schichten 1", "learning_rate": "Lernrate",  "neurons": "Neuronen", "optimizer": "Optimierer", "target": "Genauigkeit"},
                             color_continuous_scale=px.colors.sequential.Viridis,
                             color_continuous_midpoint=0.6)
    fig.update_layout(coloraxis_colorbar=dict(
        title="Genauigkeit",
        yanchor="bottom",
        y=0,
        ticks="outside"
    ))
    fig.show()


    #plot_gp(nn_bo, x, y) # quelle: https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/visualization.ipynb
    
    params_nn_ = nn_bo.max['params']
    learning_rate = params_nn_['learning_rate']
    activationL = ['relu',  'softsign', 'tanh',
                   'elu', 'relu']
    params_nn_['activation'] = activationL[round(params_nn_['activation'])]
    params_nn_['batch_size'] = round(params_nn_['batch_size'])
    params_nn_['epochs'] = round(params_nn_['epochs'])
    params_nn_['layers1'] = round(params_nn_['layers1'])
    params_nn_['layers2'] = round(params_nn_['layers2'])
    params_nn_['neurons'] = round(params_nn_['neurons'])
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
    optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
                 'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
                 'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
                 'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
    params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
    print(params_nn_)
    
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    
    global params_nn_fin 
    params_nn_fin = params_nn_
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=params_nn_['epochs'], batch_size=params_nn_['batch_size'],
                             verbose=0)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print(nn.build_fn().summary())
    nn.fit(X_train, y_train,  verbose=1)
    
    y_pred = nn.predict(X_test)
    accuracy = nn.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)

    
    print('Accuracy: %.2f' % (accuracy*100))
    print()

    return accuracy, f1
    
    
    
# Create function
def nn_cl_bo2(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, dropout, dropout_rate):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
    optimizerD= {'Adam':Adam(learning_rate=learning_rate), 'SGD':SGD(learning_rate=learning_rate),
                 'RMSprop':RMSprop(learning_rate=learning_rate), 'Adadelta':Adadelta(learning_rate=learning_rate),
                 'Adagrad':Adagrad(learning_rate=learning_rate), 'Adamax':Adamax(learning_rate=learning_rate),
                 'Nadam':Nadam(learning_rate=learning_rate), 'Ftrl':Ftrl(learning_rate=learning_rate)}
    activationL = ['relu',  'softsign', 'tanh', 
                   'elu', 'relu']
    neurons = round(neurons)
    activation = activationL[round(activation)]
    optimizer = optimizerD[optimizerL[round(optimizer)]]
    batch_size = round(batch_size)
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)
    def nn_cl_fun():
        nn = Sequential()
        nn.add(Dense(12, input_dim=4, activation=activation))
        for i in range(layers1):
            nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate, seed=124))
        for i in range(layers2):
            nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return nn
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=124)
    
    score = cross_val_score(nn, X_train_hyp, y_train_hyp, scoring=score_acc, cv=kfold, error_score="raise", fit_params={'callbacks':[es]}).mean()
    
    return score

# Fitting Neural Network
def nn_cl_fun():
    nn = Sequential()
    print(params_nn_fin)
    nn.add(Dense(12, input_dim=4, activation=params_nn_fin['activation']))
    for i in range(params_nn_fin['layers1']):
        nn.add(Dense(params_nn_fin['neurons'], activation=params_nn_fin['activation']))
    if params_nn_fin['dropout'] > 0.5:
        nn.add(Dropout(params_nn_fin['dropout_rate'], seed=124))
    for i in range(params_nn_fin['layers2']):
        nn.add(Dense(params_nn_fin['neurons'], activation=params_nn_fin['activation']))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer=params_nn_fin['optimizer'], metrics=['accuracy'])
    return nn


    
# def posterior(optimizer, x_obs, y_obs, grid):
#     optimizer._gp.fit(x_obs, y_obs)

#     mu, sigma = optimizer._gp.predict(grid, return_std=True)
#     return mu, sigma

# def plot_gp(optimizer, x, y):
#     # hyperparameter sweep wandb damit sieht es deutlich besser asu aber etwas aufwendig ?
#     data = optimizer.res
#     iterations = list(range(1, len(data) + 1))
#     scores = [entry['target'] for entry in data]

#     # Plot Convergence
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, scores, marker='o', linestyle='-', color='b')
#     plt.xlabel('Iteration')
#     plt.ylabel('Target Score')
#     plt.title('Convergence of Bayesian Optimization')
#     plt.grid(True)
#     plt.show()

#     # Plot Hyperparameter Landscape
#     params = [entry['params'] for entry in data[1:]]  # Omit the first point for clearer visualization

#     fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
#     fig.suptitle('Hyperparameter Landscape')

#     for i, (param_name, param_values) in enumerate(params[0].items()):
#         row, col = divmod(i, 3)
#         axes[row, col].scatter(iterations[1:], [param['params'][param_name] for param in params], c=scores[1:], cmap='viridis', marker='o')
#         axes[row, col].set_xlabel('Iteration')
#         axes[row, col].set_ylabel(param_name)
#         axes[row, col].set_title(f'{param_name} Landscape')

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()

# def plot_manova_results(): 
#     fig = plt.figure(figsize=(16, 10))
#     steps = len(optimizer.space)
#     fig.suptitle(
#         'Gaussian Process and Utility Function After {} Steps'.format(steps),
#         fontdict={'size':30}
#     )

#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
#     axis = plt.subplot(gs[0])
#     acq = plt.subplot(gs[1])

#     print(optimizer.res)
#     x_obs = np.array([[res["params"]["epochs"]] for res in optimizer.res])
#     y_obs = np.array([res["target"] for res in optimizer.res])

#     mu, sigma = posterior(optimizer, x_obs, y_obs, x)
#     # axis.plot(x, y, linewidth=3, label='Target')
#     axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
#     axis.plot(x, mu, '--', color='k', label='Prediction')

#     axis.fill(np.concatenate([x, x[::-1]]), 
#               np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
#         alpha=.6, fc='c', ec='None', label='95% confidence interval')

#     axis.set_xlim((-2, 10))
#     axis.set_ylim((None, None))
#     axis.set_ylabel('f(x)', fontdict={'size':20})
#     axis.set_xlabel('x', fontdict={'size':20})

#     utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
#     utility = utility_function.utility(x, optimizer._gp, 0)
#     acq.plot(x, utility, label='Utility Function', color='purple')
#     acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
#              label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
#     acq.set_xlim((-2, 10))
#     acq.set_ylim((0, np.max(utility) + 0.5))
#     acq.set_ylabel('Utility', fontdict={'size':20})
#     acq.set_xlabel('x', fontdict={'size':20})

#     axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
#     acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
#     plt.show()

# def train():
#     # Default values for hyper-parameters we're going to sweep over
#     config_defaults = {
#         'epochs': 5,
#         'batch_size': 128,
#         'weight_decay': 0.0005,
#         'learning_rate': 1e-3,
#         'activation': 'relu',
#         'optimizer': 'nadam',
#         'hidden_layer_size': 128,
#         'conv_layer_size': 16,
#         'dropout': 0.5,
#         'momentum': 0.9,
#         'seed': 42
#     }

#     # Initialize a new wandb run
#     wandb.init(config=config_defaults)
    
#     # Config is a variable that holds and saves hyperparameters and inputs
#     config = wandb.config
    
#     # Define the model architecture - This is a simplified version of the VGG19 architecture
#     model = Sequential()
    
#     # Set of Conv2D, Conv2D, MaxPooling2D layers with 32 and 64 filters
#     model.add(Dense(12, input_dim = 4, 
#                      activation ='relu'))
#     model.add(Dense(units=config.neurons, activation='relu'))
#     model.add(Dense(units=config.neurons, activation='relu'))
#     model.add(Dense(units=config.neurons, activation='relu'))
#     model.add(Dropout(config.dropout))
#     model.add(Dense(units=config.neurons, activation='relu'))


    
#     model.add(Dense(1, activation='sigmoid'))
    
#     # Define the optimizer
#     if config.optimizer=='sgd':
#       optimizer = SGD(lr=config.learning_rate, decay=1e-5, momentum=config.momentum, nesterov=True)
#     elif config.optimizer=='rmsprop':
#       optimizer = RMSprop(lr=config.learning_rate, decay=1e-5)
#     elif config.optimizer=='adam':
#       optimizer = Adam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)
#     elif config.optimizer=='nadam':
#       optimizer = Nadam(lr=config.learning_rate, beta_1=0.9, beta_2=0.999, clipnorm=1.0)

#     model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics=['accuracy'])

#     model.fit(X_train_g, y_train_g, batch_size=config.batch_size,
#               epochs=config.epochs,
#               validation_data=(X_test_g, y_test_g),
#               callbacks=[WandbCallback(data_type="tabular", validation_data=(X_test_g, y_test_g), labels=labels),
#                           EarlyStopping(patience=10, restore_best_weights=True)])
    
# def sweep_hypertuning(X_train, X_test, y_train, y_test, dataset_object):

#     global X_train_g 
#     global X_test_g
#     global y_train_g
#     global y_test_g
#     global labels 
#     labels = dataset_object.sample_name
    
#     X_train_g = X_train
#     X_test_g = X_test
#     y_train_g = y_train
#     y_test_g = y_test 
#     #wandb.init()
    
    
#     sweep_config = {
#         'method': 'random', #grid, random
#         'metric': {
#           'name': 'accuracy',
#           'goal': 'maximize'   
#         },
#         'parameters': {
#             'epochs': {
#                 'values': [50, 100, 200]
#             },
#             'batch_size': {
#                 'values': [ 128, 64, 32, 10]
#             },
#             'dropout': {
#                 'values': [0.0, 0.1, 0.3]
#             },
#             'neurons': {
#                 'values': [ 32, 64, 124]
#             },
#             'weight_decay': {
#                 'values': [0.0005]
#             },
#             'learning_rate': {
#                 'values': [0.1, 0.01]
#             },
#             'optimizer': {
#                 'values': ['adam', 'nadam', 'sgd', 'rmsprop']
#             },
#             'activation': {
#                 'values': ['relu', 'elu', 'selu', 'softmax']
#             }
#         }
#     }
#     sweep_id = wandb.sweep(sweep_config, entity="loschnora", project="sweeps-try3")
#     wandb.agent(sweep_id, train, count = 5)
    