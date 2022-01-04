import numpy as np
import pandas as pd
from keras.preprocessing import sequence
import keras
from keras import backend as K
from keras.models import load_model
import argparse
import h5py

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0] 
    else:
        return [seq_dic[aa] for aa in seq]

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
               prot_len=2500, prot_vec="Convolution",
               drug_vec="Convolution", drug_len=2048):

    print("Parsing {0} , {1}, {2} with length {3}, type {4}".format(*[dti_dir ,drug_dir, protein_dir, prot_len, prot_vec]))

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")


    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    drug_feature = np.stack(dti_df[drug_vec].map(lambda fp: fp.split("\t")))
    if prot_vec=="Convolution":
        protein_feature = sequence.pad_sequences(dti_df["encoded_sequence"].values, prot_len)
    else:
        protein_feature = np.stack(dti_df[prot_vec].map(lambda fp: fp.split("\t")))
    if with_label:
        label = dti_df[label_col].values
        print("\tPositive data : %d" %(sum(dti_df[label_col])))
        print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
        return {"protein_feature": protein_feature, "drug_feature": drug_feature, "label": label,
                "Compound_ID":dti_df["Compound_ID"].tolist(), "Protein_ID":dti_df["Protein_ID"].tolist()}
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature,
                "Compound_ID":dti_df["Compound_ID"].tolist(), "Protein_ID":dti_df["Protein_ID"].tolist()}



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true", default=False)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)
    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="morgan_fp")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=2048, type=int)
    args = parser.parse_args()
    
    model = args.model

    test_names = args.test_name
    tests = args.test_dti_dir
    test_proteins = args.test_protein_dir
    test_drugs = args.test_drug_dir
    test_sets = zip(test_names, tests, test_drugs, test_proteins)
    with_label = args.with_label
    output_file = args.output


    f = h5py.File(model, 'r+')

    try:
        f.__delitem__("optimizer_weights")
    except:
        print("optimizer_weights are already deleted")

    f.close()

    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
    }
    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, with_label=with_label, **type_params)
                for test_name, test_dti, test_drug, test_protein in test_sets}

    loaded_model = load_model(model)
    print("prediction")
    result_df = pd.DataFrame()
    result_columns = []
    for dataset in test_dic:
        temp_df = pd.DataFrame()
        prediction_dic = test_dic[dataset]
        N = int(np.ceil(prediction_dic["drug_feature"].shape[0]/50))
        d_splitted = np.array_split(prediction_dic["drug_feature"], N)
        p_splitted = np.array_split(prediction_dic["protein_feature"], N)
        predicted = sum([np.squeeze(loaded_model.predict([d,p])).tolist() for d,p in zip(d_splitted, p_splitted)], [])
        temp_df[dataset, 'predicted'] = predicted
        temp_df[dataset, 'Compound_ID'] = prediction_dic["Compound_ID"]
        temp_df[dataset, 'Protein_ID'] = prediction_dic["Protein_ID"]
        if with_label:
           temp_df[dataset, 'label'] = np.squeeze(test_dic[dataset]['label'])
        result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
        result_columns.append((dataset, "predicted"))
        result_columns.append((dataset, "Compound_ID"))
        result_columns.append((dataset, "Protein_ID"))
        if with_label:
           result_columns.append((dataset, "label"))
    result_df.columns = pd.MultiIndex.from_tuples(result_columns)
    print("save to %s"%output_file)
    result_df.to_csv(output_file, index=False)
    '''
    predicted = loaded_model.predict([prediction_dic["drug_feature"],prediction_dic["protein_feature"]])
    dti_dic = prediction_dic['dti']
    dti_dic["predicted"] = predicted
    dti_dic.to_csv(output)
    '''
