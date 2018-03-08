# coding: utf-8


import numpy as np
from numpy import genfromtxt
import random
import tensorflow as tf
from keras import backend as K
import pandas as pd

from keras.models import Sequential, load_model, Model
from keras.preprocessing import sequence
from keras.layers import Input, Dense, merge, Flatten, Convolution2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Embedding, LSTM, Bidirectional
from keras.layers import Convolution1D, GlobalMaxPooling1D,SpatialDropout1D
from keras.layers import Concatenate
from keras.layers import Merge
from keras.optimizers import Adam, Adadelta
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2


# In[3]:

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier


# In[5]:

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


# In[6]:

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [ seq_dic[aa] for aa in seq]


# In[7]:

prot_len = 2500


# In[8]:

def parse_data(train_dir, protein_dir, with_label=True, prot_len=2500, prot_vec=False):
    print "Parsing {0} , {1} with length {2}, type {3}".format(*[train_dir, protein_dir, prot_len, prot_vec])
    protein_col = "protein"
    drug_col = "drug"
    name_cols = [protein_col, drug_col]
    fp_cols = range(0,2048)
    col_names = name_cols + fp_cols
    if with_label:
        label_col = "label"
        col_names += [label_col]
    train_data = pd.read_table(train_dir,names=col_names)
    train_d = train_data[fp_cols].values
    if protein_dir.split(".")[-1]=='csv':
        protein_df = pd.read_csv(protein_dir, index_col="protein")
    else:
        protein_df = pd.read_table(protein_dir, index_col="protein")
    if prot_vec!="Convolution":
       prot_dic= protein_df[prot_vec].map(lambda seq: seq.split("\t")).to_dict()
       train_p = np.array(list(train_data[protein_col].map(lambda seq: prot_dic[seq])), dtype=np.float64)
    else:
       protein_df["seq_num_vec"] = protein_df.seq.map(lambda seq : encodeSeq(seq, seq_dic))
       prot_dic = protein_df.seq_num_vec.to_dict()
       train_p =  sequence.pad_sequences(train_data[protein_col].map(lambda seq:prot_dic[seq]), maxlen=prot_len)

    if with_label:
        label =  train_data[label_col].values
        print "\tPositive data : %d" %(sum(train_data[label_col]))
        print "\tNegative data : %d" %(train_data.shape[0] - sum(train_data[label_col]))
        return train_p, train_d, label
    else:
        return train_p, train_d


# In[13]:

class Drug_Target_Prediction(object):
    
    
    def PLayer(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            #model_p = Convolution1D(filters=filters, kernel_size=size, padding='valid', activity_regularizer=l2(regularizer_param), kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='valid', kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f
    
    
    def modelv(self, dropout, drug_layers, protein_strides, filters, fc_layers, prot_vec=False, prot_len=2500, activation='relu', hidden_protein_layer=None, initializer="glorot_normal"):
	def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)
        regularizer_param = 0.0001
        input_d = Input(shape=(2048,))
        input_p = Input(shape=(prot_len,))
        params_dic = {"kernel_initializer": initializer,
                      "activity_regularizer": l2(regularizer_param),
                      "kernel_regularizer": l2(regularizer_param),
			}
        input_layer_d = input_d
        input_layer_p = input_p
        if drug_layers is not None:
	   drug_layers = return_tuple(drug_layers)        
           for layer_size in drug_layers:
               model_d = Dense(layer_size, **params_dic)(input_layer_d)
               model_d = BatchNormalization()(model_d)
               model_d = Activation(activation)(model_d)
               model_d = Dropout(dropout)(model_d)
               input_layer_d = model_d
        if prot_vec!="Convolution":
           if protein_strides is not None:
              for layer_size in protein_strides:
                  model_p = Dense(layer_size, **params_dic)(input_layer_p)
                  model_p = BatchNormalization()(model_p)
                  model_p = Activation(activation)(model_p)
                  model_p = Dropout(dropout)(model_p)
                  input_layer_p = model_p
        else:
           model_p = Embedding(26,20)(input_p)
           model_p = SpatialDropout1D(0.2)(model_p)
	   model_ps = [self.PLayer(stride_size, filters, activation, initializer, regularizer_param)(model_p) for stride_size in protein_strides]
           if len(model_ps)!=1:
              model_p = Concatenate(axis=1)(model_ps)
           else:
              model_p = model_ps[0]
	if hidden_protein_layer:
           model_p = Dense(hidden_protein_layer, **params_dic)(model_p)
           model_p = BatchNormalization()(model_p)
           model_p = Activation(activation)(model_p)
           model_p = Dropout(dropout)(model_p)

        model_t = Concatenate(axis=1)([model_d,model_p])
        input_dim = filters*len(protein_strides) + drug_layers[-1]
        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for fc_layer in fc_layers:
                model_t = Dense(units=fc_layer, input_dim = input_dim, 
                                **params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(activation)(model_t)
                #model_t = Dropout(dropout)(model_t)
                input_dim = fc_layer
        model_t = Dense(1,activation='sigmoid', **params_dic)(model_t)

        model_f = Model(inputs=[input_d, input_p], outputs = model_t)

        return model_f
    
    
    def __init__(self, dropout=0.2, drug_layers=(1024,512), protein_strides = (10,15,20,25), filters=64, learning_rate=1e-3, decay=0.0,
                fc_layers=None, prot_vec=None, prot_len=2500, activation="relu", hidden_protein_layer=None):
        self.__dropout = dropout
        self.__drugs_layer = drug_layers
        self.__protein_strides = protein_strides
        self.__filters = filters
        self.__fc_layers = fc_layers
        self.__learning_rate = learning_rate
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__activation = activation
        self.__hidden_protein_layer = hidden_protein_layer
        self.__decay = decay
        print("learning rate : %f"%learning_rate)
        print("training type : %s"%prot_vec)
        print("protein length: %d"%prot_len)
        print("drotout ratio : %f"%dropout)
        print("activation    : %s"%activation)
        print("decay rate    : %f"%self.__decay)
        self.__model_t = self.modelv(self.__dropout, self.__drugs_layer, self.__protein_strides, self.__filters, self.__fc_layers, prot_vec=self.__prot_vec, prot_len=self.__prot_len, activation=self.__activation, hidden_protein_layer=self.__hidden_protein_layer)

        opt = Adam(lr=learning_rate, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        K.get_session().run(tf.global_variables_initializer())

    def fit(self, train_d, train_p, label, n_epoch=10, batch_size=32):
        for _ in xrange(n_epoch):
            history = self.__model_t.fit([train_d,train_p],label, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1,initial_epoch=_)
            print(self.__model_t.optimizer.lr)
        return self.__model_t
    
    def summary(self):
        self.__model_t.summary()
    
    def validation(self, train_d, train_p, label, output_file=None, n_epoch=10, prot_vec=None, batch_size=32, **kwargs):
        
        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ["window_sizes", "drug_layers", "fc_layers", "learning_rate"]])
            result_df = pd.DataFrame(data = [[self.__protein_strides, self.__drugs_layer, self.__fc_layers, self.__learning_rate]]*n_epoch, columns=param_tuple)
            result_df["epoch"] = range(1,n_epoch+1)
        result_dic = {dataset: {"AUC":[], "AUPR": [], "opt_threshold(AUPR)":[], "opt_threshold(AUC)":[] }for dataset in kwargs}
        
        for _ in xrange(n_epoch):
            history = self.__model_t.fit([train_d,train_p],label, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1, initial_epoch=_)
            print(K.eval(self.__model_t.optimizer.lr))
            for dataset in kwargs:
                print "\tPredction of " + dataset
                test_p = kwargs[dataset][0]
                test_d = kwargs[dataset][1]
                test_label = kwargs[dataset][2]
                prediction = self.__model_t.predict([test_d,test_p])
                fpr, tpr, thresholds_AUC =  roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label,prediction)
                distance = (1-fpr)**2+(1-tpr)**2
                EERs = (1-recall)/(1-precision)
                positive = sum(test_label)
                negative = test_label.shape[0]-positive
                ratio = negative/positive
                opt_t_AUC = thresholds_AUC[np.argmin(distance)]
                opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
                AUPR = auc(recall,precision)
                print "\tArea Under ROC Curve(AUC): %0.3f" % AUC
                print "\tArea Under PR Curve(AUPR): %0.3f" % AUPR
                print "\tOptimal threshold(AUC) : %0.3f " % opt_t_AUC
                print "\tOptimal threshold(AUPR) : %0.3f" % opt_t_AUPR
                print "================================================="
                result_dic[dataset]["AUC"].append(AUC)
                result_dic[dataset]["AUPR"].append(AUPR)
                result_dic[dataset]["opt_threshold(AUC)"].append(opt_t_AUC)
                result_dic[dataset]["opt_threshold(AUPR)"].append(opt_t_AUPR)
        if output_file:
            for dataset in kwargs:
                result_df[dataset, "AUC"] = result_dic[dataset]["AUC"]
                result_df[dataset, "AUPR"] = result_dic[dataset]["AUPR"]
                result_df[dataset, "opt_threshold(AUC)"] = result_dic[dataset]["opt_threshold(AUC)"]
                result_df[dataset, "opt_threshold(AUPR)"] = result_dic[dataset]["opt_threshold(AUPR)"]
            print "save to " + output_file
            print result_df
            result_df.to_csv(output_file, index=False)

    def predict(self, **kwargs):
        results_dic = {}
        for dataset in kwargs:
            result_dic = {}
            test_p = kwargs[dataset][0]
            test_d = kwargs[dataset][1]
            result_dic["label"] = kwargs[dataset][2]
            result_dic["predicted"] = self.__model_t.predict([test_d, test_p])
            results_dic[dataset] = result_dic
        return results_dic
    
    def save(self, output_file):
        self.__model_t.save(output_file)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="training data with [seq_index, fingerprint] ")
    parser.add_argument("seq_dir", help="sequence information with index for training data" )
    parser.add_argument("--test-name", '-n', help="name of test data set", nargs="*")
    parser.add_argument("--test-dir", "-t", help="test data with [seq_index, fingerprint, label]", nargs="*")
    parser.add_argument("--test-seq-dir", '-s', help="sequence informations with index for test data", nargs="*")
    parser.add_argument("--learning-rate", '-r',help="learning late", default=1e-4, type=float)
    parser.add_argument("--window-sizes", '-w', help="window sizes for model", default="10,15,20,25")
    parser.add_argument("--drug-layers", '-d', help="inner drugs layer for model")
    parser.add_argument("--fc-layers", '-f',help="fully connected layer for model")
    parser.add_argument("--n-epoch", '-e', help="the number of epoch", type=int, default=10)
    parser.add_argument("--validation", "-v", help="validation with validation set", action="store_true")
    parser.add_argument("--predict", "-p", help="predict independent test set", action="store_true")
    parser.add_argument("--threshold", "-T", help="threshold for prediction", type=float, default=0.5)
    parser.add_argument("--prot-vec", "-V", help="If protein type is vector, write prtein vector type", type=str)
    parser.add_argument("--prot-len", "-l", help="protein vector length", default=2500, type=int)
    parser.add_argument("--save-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    parser.add_argument("--activation", "-a", help='activation function', type=str)
    parser.add_argument("--hidden-protein-layer","-H", help="hidden dense layer for protein", default=None)
    parser.add_argument("--dropout", "-D", help="dropout ratio", default=0.2, type=float)
    parser.add_argument("--n-filters", "-F", help="number of filters", default=64, type=int)
    parser.add_argument("--batch-size", "-b", help="batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="decay rate", default=0.0, type=float)
    args = parser.parse_args()
    train_dir = args.train_dir
    seq_dir = args.seq_dir
    test_names = args.test_name
    tests = args.test_dir
    test_seqs = args.test_seq_dir
    test_sets = zip(test_names, tests, test_seqs)
    activation = args.activation
    n_epoch = args.n_epoch
    prot_len = args.prot_len
    filters = args.n_filters
    batch_size = args.batch_size
    decay = args.decay
    window_sizes = [int(size) for size in args.window_sizes.split(",")]
    if args.hidden_protein_layer: hidden_protein_layer = int(args.hidden_protein_layer)
    else: hidden_protein_layer = args.hidden_protein_layer
    dropout = args.dropout
    output_file = args.output
    if args.drug_layers is not None: drug_layers=[int(layer) for layer in args.drug_layers.split(",")]; 
    else: drug_layers=None
    if args.fc_layers is not None: fc_layers=[int(layer) for layer in args.fc_layers.split(",")]; 
    else: fc_layers=None
    learning_rate = args.learning_rate
    prot_vec = args.prot_vec
    dti_prediction_model = Drug_Target_Prediction(drug_layers=drug_layers,protein_strides=window_sizes, filters=filters, decay=decay,
                                                  fc_layers=fc_layers, learning_rate=learning_rate, prot_vec=prot_vec, prot_len=prot_len, activation=activation, hidden_protein_layer=hidden_protein_layer, dropout=dropout)
    train_p, train_d, label = parse_data(train_dir, seq_dir, with_label=True, prot_len=prot_len, prot_vec=prot_vec )
    test_dic = {test_name:parse_data(test, test_seq, with_label=True, prot_len=prot_len, prot_vec=prot_vec) for test_name, test, test_seq in test_sets}
    print dti_prediction_model.summary()
    if args.validation:
       print "validtion"
       dti_prediction_model.validation(train_d, train_p, label, output_file=output_file, n_epoch=n_epoch, prot_vec=prot_vec, batch_size=batch_size, **test_dic)
    elif args.predict:
       print "prediction"
       dti_prediction_model.fit(train_d, train_p, label, n_epoch=n_epoch, batch_size=batch_size)
       test_predicted = dti_prediction_model.predict(**test_dic)
       result_df = pd.DataFrame()
       result_columns = []
       for dataset in test_predicted:
           temp_df = pd.DataFrame()
           value = test_predicted[dataset]["predicted"]
           value = np.squeeze(value)
           print dataset+str(value.shape)
           temp_df[dataset,'predicted'] = value
           temp_df[dataset, 'label'] = np.squeeze(test_predicted[dataset]['label'])
           result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
           result_columns.append((dataset, "predicted"))
           result_columns.append((dataset, "label"))
       print result_df.head()
       result_df.columns = pd.MultiIndex.from_tuples(result_columns)
       print "save to %s"%output_file
       result_df.to_csv(output_file, index=False)
    if args.save_model:
       dti_prediction_model.save(args.save_model)
       """
       for dataset in test_predicted:
           label = test_predicted[dataset]["label"]
           predicted = test_predicted[dataset]["predicted"]
           predicted[predicted>=args.threshold] = 1
           predicted[predicted<args.threshold] = 0
           precision, recall, f1_score, support = precision_recall_fscore_support(label, predicted)
           accuracy = accuracy_score(label, predicted)
           print dataset+ " result : "
           print "accuracy : " + str(accuracy)
           print "precision : " + str(precision)
           print "recall : " + str(recall)
           print "F1-score : " + str(f1_score)
       """    
           
    exit()
