# DeepConvDTI

## Overview 

This Python script is used to train, validate, test deep learning model for prediction of drug-target interaction (DTI) Deep learning model will be built
by Keras with tensorflow. You can set almost hyper-parameters as you want, See below parameter description DTI, drug and protein data must be written as csv
file format. And feature should be tab-delimited format for script to parse data. Basically, this script builds convolutional neural network on sequence.
If you don't want convolutional neural network but traditional dense layers on provide protein feature, specify type of feature and feature length. 

## Requirement
  tensorflow > 1.0
  keras > 2.0 numpy
  pandas 
  scikit-learn  

## Usage 

  usage: DTI_deep.py [-h] [--test-name [TEST_NAME [TEST_NAME ...]]]
                     [--test-dti-dir [TEST_DTI_DIR [TEST_DTI_DIR ...]]]
                     [--test-drug-dir [TEST_DRUG_DIR [TEST_DRUG_DIR ...]]]
                     [--test-protein-dir [TEST_PROTEIN_DIR [TEST_PROTEIN_DIR ...]]]
                     [--with-label WITH_LABEL]
                     [--window-sizes [WINDOW_SIZES [WINDOW_SIZES ...]]]
                     [--protein-layers [PROTEIN_LAYERS [PROTEIN_LAYERS ...]]]
                     [--drug-layers [DRUG_LAYERS [DRUG_LAYERS ...]]]
                     [--fc-layers [FC_LAYERS [FC_LAYERS ...]]]
                     [--learning-rate LEARNING_RATE] [--n-epoch N_EPOCH]
                     [--prot-vec PROT_VEC] [--prot-len PROT_LEN]
                     [--drug-vec DRUG_VEC] [--drug-len DRUG_LEN]
                     [--activation ACTIVATION] [--dropout DROPOUT]
                     [--n-filters N_FILTERS] [--batch-size BATCH_SIZE]
                     [--decay DECAY] [--validation] [--predict]
                     [--save-model SAVE_MODEL] [--output OUTPUT]
                     dti_dir drug_dir protein_dir


## Data Specification

All training, validation, test should follow specification to be parsed correctly by DeepConvDTI

  * Model takes 3 types data as a set, Drug-target interaction data, target protein data, compound data.

  * They should be ''.csv'' format.

  * For feature column, each dimension of features in columns should be delimited with tab (''\t'')

After three data are correctly listed, target protein data and compound data will be joined with drug-target data, generating DTI feature.

### Drug target interaction data

Drug target interaction data should be at least 2 columns ''Protein_ID'' and ''Compound_ID'',

and should have ''Label'' column except ''--test'' case. ''Label'' colmun has to label ''0'' as negative and ''1'' as positive.

^  Protein_ID  ^  Compound_ID  ^  Label  ^
|  PID001      |  CID001       |  0      |
|  ...         |  ...          |  ...    |
|  PID100      |  CID100       |  1      |

### Target protein data 

Because DeepConvDTI focuses on convolution on protein sequence, protein data specification is little different from other data.

If ''Sequence'' column is specified in data and ''--prot-vec'' is ''Convolution'', it will execute convolution on protein.

Or if you specify other type of column with ''--prot-vec''(i.e. ''Prot2Vec''), it will construct dense network

''Protein_ID'' column will be used as forein key from ''Protein_ID'' from Drug-target interaction data.

^  Protein_ID  ^  Sequence      ^  Prot2Vec                  ^
|  PID001      |  MALAC....ACC  |  0.539\t-0.579\t...\t0.39  |

### Compound data

Basically same with Target protein data, but no ''Convolution''.

''Compoun_ID'' column will be used as forein key from ''Compoun_ID'' from Drug-target interaction data.

^  Compound_ID  ^  morgan_r2        ^
|  CID001       |  0\t1\t...\t0\t1  |



## Parameter specification

  positional arguments:
    dti_dir               Training DTI information [drug, target, label]
    drug_dir              Training drug information [drug, SMILES,[feature_name,
                          ..]]
    protein_dir           Training protein information [protein, seq,
                          [feature_name]]

  optional arguments:
    -h, --help            show this help message and exit
    --test-name [TEST_NAME [TEST_NAME ...]], -n [TEST_NAME [TEST_NAME ...]]
                          Name of test data sets
    --test-dti-dir [TEST_DTI_DIR [TEST_DTI_DIR ...]], -i [TEST_DTI_DIR [TEST_DTI_DIR ...]]
                          Test dti [drug, target, [label]]
    --test-drug-dir [TEST_DRUG_DIR [TEST_DRUG_DIR ...]], -d [TEST_DRUG_DIR [TEST_DRUG_DIR ...]]
                          Test drug information [drug, SMILES,[feature_name,
                          ..]]
    --test-protein-dir [TEST_PROTEIN_DIR [TEST_PROTEIN_DIR ...]], -t [TEST_PROTEIN_DIR [TEST_PROTEIN_DIR ...]]
                          Test Protein information [protein, seq,
                          [feature_name]]
    --with-label WITH_LABEL, -W WITH_LABEL
                          Existence of label information in test DTI
    --window-sizes [WINDOW_SIZES [WINDOW_SIZES ...]], -w [WINDOW_SIZES [WINDOW_SIZES ...]]
                          Window sizes for model (only works for Convolution)
    --protein-layers [PROTEIN_LAYERS [PROTEIN_LAYERS ...]], -p [PROTEIN_LAYERS [PROTEIN_LAYERS ...]]
                          Dense layers for protein
    --drug-layers [DRUG_LAYERS [DRUG_LAYERS ...]], -c [DRUG_LAYERS [DRUG_LAYERS ...]]
                          Dense layers for drugs
    --fc-layers [FC_LAYERS [FC_LAYERS ...]], -f [FC_LAYERS [FC_LAYERS ...]]
                          Dense layers for concatenated layers of drug and
                          target layer
    --learning-rate LEARNING_RATE, -r LEARNING_RATE
                          Learning late for training
    --n-epoch N_EPOCH, -e N_EPOCH
                          The number of epochs for training or validation
    --prot-vec PROT_VEC, -v PROT_VEC
                          Type of protein feature, if Convolution, it will
                          execute conlvolution on sequeunce
    --prot-len PROT_LEN, -l PROT_LEN
                          Protein vector length
    --drug-vec DRUG_VEC, -V DRUG_VEC
                          Type of drug feature
    --drug-len DRUG_LEN, -L DRUG_LEN
                          Drug vector length
    --activation ACTIVATION, -a ACTIVATION
                          Activation function of model
    --dropout DROPOUT, -D DROPOUT
                          Dropout ratio
    --n-filters N_FILTERS, -F N_FILTERS
                          Number of filters for convolution layer, only works
                          for Convolution
    --batch-size BATCH_SIZE, -b BATCH_SIZE
                          Batch size
    --decay DECAY, -y DECAY
                          Learning rate decay
    --validation          Excute validation with independent data, will give AUC
                          and AUPR (No prediction result)
    --predict             Predict interactions of independent test set
    --save-model SAVE_MODEL, -m SAVE_MODEL
                          save model
    --output OUTPUT, -o OUTPUT
                          Prediction output

## Contact 
[[mailto://dlsrnsladlek@gist.ac.kr|dlsrnsladlek@gist.ac.kr]]

