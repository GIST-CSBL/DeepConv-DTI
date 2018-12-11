
import itertools

BASE_DIR = "/home/share/dlsrnsi/DTI/Deep_DTI/"
OUTPUT_DIR = BASE_DIR + "Result_performance/MATADOR/"
VALIDATION_OUTPUT_DIR = BASE_DIR+"Result_validation/ratio_2/"
PREDICTION_DIR = BASE_DIR+"Prediction_result/MATADOR/"
MODEL_DIR = BASE_DIR+"Model/MATADOR/"
mail_address = "dlsrnsladlek@naver.com"
workdir: BASE_DIR

def get_arg_sets(max_list):
    max_index = len(max_list)
    for i in range(1,max_index+1):
        return_var = max_list[0:i]
        yield ",".join([str(var) for var in return_var])


max_window = [10,15,20,25,30,35]
window_set = get_arg_sets(max_window)
window_arg_set = list(iter(window_set))

max_drug_layer = [1024,512,256]
drug_layer_set = get_arg_sets(max_drug_layer)
drug_layer_arg_set = list(iter(drug_layer_set))

max_fc_layer = [256,128,64]
fc_layer_set = get_arg_sets(max_fc_layer)
fc_layer_arg_set = list(iter(fc_layer_set))

learning_rates = [str(0.001)]
lengthes = ["2500"]
types = ["Convolution"]
n_epochs = ["25"]
thresholds = ["0.15"]

param_keys = ["window_size", "drug_layer", "fc_layer", "learning_rate", "type", "length", "threshold", "epoch"]
paramset_dic = {str(i):{param_key: param for param_key, param in zip(param_keys, param_set)} for i, param_set in enumerate(itertools.product(window_arg_set, drug_layer_arg_set, fc_layer_arg_set, learning_rates, types,lengthes, thresholds, n_epochs))}

training_dir = "/DAS_Storage1/Drug_AI_project/training_dataset/training_dataset/"
training_wildcard = "{number}.csv"
training_drug = "/DAS_Storage1/Drug_AI_project/training_dataset/merged_compound.csv"
training_seq = "/DAS_Storage1/Drug_AI_project/training_dataset/merged_protein.csv"

validation_names = ["MATADOR"]
validation_dirs = ["/DAS_Storage1/Drug_AI_project/validation_dataset/validation_dti.csv"]
validation_drugs = ["/DAS_Storage1/Drug_AI_project/validation_dataset/validation_compound.csv"]
validation_seqs = ["/DAS_Storage1/Drug_AI_project/validation_dataset/validation_protein.csv"]

test_names = ["PubChem", "PubChem_unseen_drug", "PubChem_unseen_target", "PubChem_unseen_both" ,"KinaseSARfari" ]
test_dirs = ["/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_dti.csv", "/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_dti_new_compound.csv", "/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_dti_new_protein.csv", "/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_dti_both_new.csv","/DAS_Storage1/Drug_AI_project/test_dataset/KinaseSARfari/test_dti.csv"]
test_drugs = ["/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_compound.csv","/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_compound.csv","/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_compound.csv","/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_compound.csv","/DAS_Storage1/Drug_AI_project/test_dataset/KinaseSARfari/test_compound.csv"]
test_seqs = ["/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_protein.csv","/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_protein.csv","/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_protein.csv","/DAS_Storage1/Drug_AI_project/test_dataset/PubChem/test_protein.csv","/DAS_Storage1/Drug_AI_project/test_dataset/KinaseSARfari/test_protein.csv"]

test_names = ["MATADOR"]
test_dirs = ["/DAS_Storage1/Drug_AI_project/validation_dataset/validation_dti.csv"]
test_drugs = ["/DAS_Storage1/Drug_AI_project/validation_dataset/validation_compound.csv"]
test_seqs = ["/DAS_Storage1/Drug_AI_project/validation_dataset/validation_protein.csv"]


training_sets, = glob_wildcards(training_dir+training_wildcard)
print(training_sets)
#training_sets = [1]



def return_params(wildcards) -> dict:
    param_dic =  paramset_dic[wildcards.paramset_key]
    return param_dic
paramset_list = ["127", "AAC","123"]
paramset_list = ["Similarity"]
paramset_dic["127_2500_64"] ={'window_size': '10 15 20 25 30', 'protein_length': '2500', 'fc_layer': '256', 'threshold': '0.2', 'drug_layer': '512 128', 'protein_type': 'Convolution', 'learning_rate': '0.0001', 'epoch': '20',  "decay":"0.0001","activation":"elu", "dropout":"0.00", "hidden_layer":"128"}
paramset_dic["127_2500_more_decay"] ={'window_size': '10 15 20 25 30', 'protein_length': '2500', 'fc_layer': '256', 'threshold': '0.2', 'drug_layer': '512 128', 'protein_type': 'Convolution', 'learning_rate': '0.0001', 'epoch': '25',  "decay":"0.1","activation":"elu", "dropout":"0.00", "hidden_layer":"128"}
paramset_dic["127_2750"] ={'window_size': '10 15 20 25 30', 'protein_length': '2750', 'fc_layer': '256', 'threshold': '0.2', 'drug_layer': '512 128', 'protein_type': 'Convolution', 'learning_rate': '0.0001', 'epoch': '25',  "decay":"0.0001","activation":"elu", "dropout":"0.00", "hidden_layer":"128"}
paramset_dic["AAC"] = {'drug_layer': '1024 512', 'fc_layer':'512', 'hidden_layer':'1024 512', "window_size":"0", 'learning_rate':'0.0001',"protein_type":"AAComposition", "protein_length":'8420', "epoch":"10", "threshold":"0.18", "activation":"elu", "dropout":"0.00", "decay":"0.00"}
paramset_dic["Similarity"] ={'window_size': '0', 'protein_length': '3675', 'fc_layer': '256', 'threshold': '0.12', 'drug_layer': '512 128', 'protein_type': 'Similarity', 'learning_rate': '0.0001', 'epoch': '15',  "decay":"0.0001","activation":"elu", "dropout":"0.00", "hidden_layer":"512 128"}
paramset_dic["CTD"] ={'window_size': '0', 'protein_length': '147', 'fc_layer': '512', 'threshold': '0.15', 'drug_layer': '1024 512', 'protein_type': 'CTD', 'learning_rate': '0.0001', 'epoch': '20',  "decay":"0.0001","activation":"elu", "dropout":"0.00", "hidden_layer":"64"}
paramset_dic["127_128"] ={'window_size': '10 15 20 25 30', 'protein_length': '2500', 'fc_layer': '128', 'threshold': '0.2', 'drug_layer': '512 128', 'protein_type': 'Convolution', 'learning_rate': '0.0001', 'epoch': '20',  "decay":"0.0001","activation":"elu", "dropout":"0.00", "hidden_layer":"128"}
paramset_list = ["Similarity", "CTD", "127_128"]
#paramset_list = ["127_2500_more_decay"]
#paramset_list = ["Similarity"]
for param_key in paramset_list:
    print(param_key,paramset_dic[param_key])
#training_sets = range(0,3)

localrules: evaluate_performance

rule all:
#     input: expand("/DAS_Storage1/dlsrnsi/DTI/Deep_DTI/DTI_for_target/P28223/Prediction/{paramset_key}_{number}.csv", paramset_key=paramset_list, number=training_sets)
     input: expand(OUTPUT_DIR+"{paramset_key}_{number}.csv", paramset_key=paramset_list, number=training_sets)
#      input: expand(VALIDATION_OUTPUT_DIR+"{paramset_key}_{number}.csv", paramset_key=paramset_list, number=training_sets)

            
onsuccess:
     print("Workflow finished, no error")
     shell("mail -s 'Workflow finished without error' "+mail_address+" < {log}")

onerror:
     print("An error occurred")
     shell("mail -s 'An error occurred' "+mail_address+" < {log}")

rule validation_DTI_deep:
     input: training_seq=training_seq, validation_dir=validation_dirs, validation_seq=validation_seqs, training_set=training_dir+training_wildcard, training_drug=training_drug, validation_drug=validation_drugs
     params:  return_params, test_names=validation_names, n_epoch=40
     output: VALIDATION_OUTPUT_DIR+"{paramset_key}_{number}.csv"
     run:         
         param_dic = params[0]         
         shell("""python2 DTI_deep.py {input.training_set} {input.training_drug} {input.training_seq} --validation -n {params.test_names} -i {input.validation_dir} -t {input.validation_seq} -d {input.validation_drug} -o {output} -e {params.n_epoch} -b 64 -F 128 """+""" -p {hidden_layer} -w {window_size} -c {drug_layer} -f {fc_layer} -r {learning_rate} -v {protein_type} -l {protein_length} -D {dropout} -a {activation} -y {decay} -V morgan_fp_r2 """.format(**param_dic))

rule run_dti_deep:
     input: training_seq=training_seq, test_dtis = test_dirs, test_drugs = test_drugs, test_seqs=test_seqs, training_set=training_dir+training_wildcard, training_drug=training_drug
     params:  return_params, test_names=test_names
     output: prediction = PREDICTION_DIR+"{paramset_key}_{number}.csv", model = MODEL_DIR+"{paramset_key}_{number}.h5"
     run:
         param_dic = params[0]
         cmd = """python2 DTI_deep.py {input.training_set} {input.training_drug} {input.training_seq} -n {params.test_names} -i {input.test_dtis} -d {input.test_drugs} -t {input.test_seqs} --predict -o {output.prediction} -m {output.model} -b 64 -F 128"""+""" -w {window_size} -c {drug_layer} -f {fc_layer} -r {learning_rate} -v {protein_type} -l {protein_length} -e {epoch} -D {dropout} -a {activation} -y {decay} -p {hidden_layer} -V morgan_fp_r2""".format(**param_dic)
         print(cmd)
         shell(cmd)



rule evaluate_performance:
     input: prediction =  PREDICTION_DIR+"{paramset_key}_{number}.csv"
     output: performance = OUTPUT_DIR+"{paramset_key}_{number}.csv"
     params: return_params, test_names=test_names
     run:
         param_dic = params[0] 
         cmd = "python2 evaluate_performance.py {input.prediction} -o {output.performance} -n {params.test_names}" + " -T {threshold}".format(**param_dic)
         print(cmd)
         shell(cmd)

rule predict_with_model:
     input: model=MODEL_DIR+"{paramset_key}_{number}.h5", dti = "/DAS_Storage1/dlsrnsi/DTI/Deep_DTI/DTI_for_target/P28223/P28223.csv", drug=training_drug, target=training_seq
     output: "/DAS_Storage1/dlsrnsi/DTI/Deep_DTI/DTI_for_target/P28223/Prediction/{paramset_key}_{number}.csv"
     shell: "python2 predict_with_model.py {input.model} {input.dti} {input.drug} {input.target} {output}"
