import itertools

BASE_DIR = "/home/share/dlsrnsi/DTI/Deep_DTI/"
OUTPUT_DIR = BASE_DIR + "Result_performance/ratio_2/"
VALIDATION_OUTPUT_DIR = BASE_DIR+"Result_validation/ratio_2/"
PREDICTION_DIR = BASE_DIR+"Prediction_result/ratio_2/"
MODEL_DIR = BASE_DIR+"Model/"
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

training_dir = "/DAS_Storage1/dlsrnsi/DTI/training_set_ratio_2/"
training_wildcard = "{number}.tsv"

validation_names = ["MATADOR"]
validation_dirs = ["/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/MATADOR/validation_set_with_high_negative.tsv"]
validation_seqs = ["/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/MATADOR/matador_protein.csv"]

test_names = ["PubChem", "PubChem_unseen_drug", "PubChem_unseen_target", "PubChem_unseen_both" ,"KinaseSARfari" ]
test_dirs = ["/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/PubChem/test_set.tsv", "/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/PubChem/test_set_unseen_compound.tsv", "/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/PubChem/test_set_unseen_protein.tsv", "/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/PubChem/test_set_unseen_both.tsv","/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/KinaseSARfari/test_set.tsv"]
test_seqs = ["/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/PubChem/protein.csv", "/home/share/dlsrnsi/DTI/Deep_DTI/validation_set/KinaseSARfari/proteins.csv"]



training_sets, = glob_wildcards(training_dir+training_wildcard)
#training_sets = [1]
training_seq = "/home/share/dlsrnsi/DTI/Deep_DTI/training_set/final_result/proteins.csv"



def return_params(wildcards) -> dict:
    param_dic =  paramset_dic[wildcards.paramset_key]
    return param_dic
paramset_dic["54"] = {'drug_layer': '1024,512', 'fc_layer': '256', 'window_size': '15,20,25,30', 'learning_rate': '0.0001'}
paramset_dic["55"] = {'drug_layer': '1024,512,256', 'fc_layer': '256', 'window_size': '15,20,25,30', 'learning_rate': '0.0001'}
paramset_list = ["30","54","55"]
paramset_list = ["30", "39"]
paramset_list = ["39", "AAC", "CTD","30","39_hidden_128"]
paramset_list = ["30_hidden_128", "39_hidden_128"]


paramset_dic["CTD"] = {'drug_layer': '1024,512', 'fc_layer': '512', 'window_size': '64', 'learning_rate': '0.0001', "type":"CTD", "length":'147',"epoch":"20", "threshold":"0.1"}
paramset_dic["AAC"] = {'drug_layer': '1024,512', 'fc_layer':'512', 'window_size':'1024,512', 'learning_rate':'0.0001',"type":"AAComposition", "length":'8420', "epoch":"15", "threshold":"0.15"}
paramset_dic["60"] = {"drug_layer": '1024,512', 'fc_layer':'512,256', 'window_size': '5,10,15,20,25', 'learning_rate': '0.001', 'type':"Convolution","length":"2500", "epoch":"10","threshold":"0.20"}
paramset_dic["39_hidden_128"] ={'window_size': '10,15,20,25,30', 'length': '2500', 'fc_layer': '128', 'threshold': '0.15', 'drug_layer': '512,128', 'type': 'Convolution', 'learning_rate': '0.0001', 'epoch': '25', 'hidden_protein_layer':'128'}
paramset_dic["30_hidden_128"] ={'window_size': '10,15,20,25,30', 'length': '2500', 'fc_layer': '128', 'threshold': '0.15', 'drug_layer': '512,128', 'type': 'Convolution', 'learning_rate': '0.0001', 'epoch': '25', 'hidden_protein_layer':'128'}

for param_key in paramset_list:
    print(param_key,paramset_dic[param_key])
rule all:
     input: expand(OUTPUT_DIR+"{paramset_key}_{number}.csv", paramset_key=paramset_list, number=training_sets)

            
onsuccess:
     print("Workflow finished, no error")
     shell("mail -s 'Workflow finished without error' "+mail_address+" < {log}")

onerror:
     print("An error occurred")
     shell("mail -s 'An error occurred' "+mail_address+" < {log}")

rule validation_DTI_deep:
     input: training_seq=training_seq, validation_dir=validation_dirs[0], validation_seq=validation_seqs[0], training_set=training_dir+training_wildcard 
     params:  return_params, test_names=validation_names, n_epoch=40
     output: VALIDATION_OUTPUT_DIR+"{paramset_key}_{number}.csv"
     run:         
         param_dic = params[0]         
         shell("""python DTI_deep.py {input.training_set} {input.training_seq} --validation -n {params.test_names} -t {input.validation_dir} -s {input.validation_seq} -o {output} -e {params.n_epoch} -a elu """+""" -w {window_size} -d {drug_layer} -f {fc_layer} -r {learning_rate} -V {type} -l {length} -H {hidden_protein_layer}""".format(**param_dic))

rule run_dti_deep:
     input: training_seq=training_seq, pubchem_test = test_dirs[0], pubchem_usc = test_dirs[1], pubchem_usp=test_dirs[2],pubchem_usb=test_dirs[3],kinasesarfari_test = test_dirs[4], pubchem_seq = test_seqs[0], kinsase_safari_seq = test_seqs[1], training_set=training_dir+training_wildcard 
     params:  return_params, test_names=test_names
     output: prediction = PREDICTION_DIR+"{paramset_key}_{number}.csv", model = MODEL_DIR+"{paramset_key}_{number}.h5"
     run:
         param_dic = params[0]
         cmd = """python DTI_deep.py {input.training_set} {input.training_seq} -n {params.test_names} -t {input.pubchem_test} {input.pubchem_usc} {input.pubchem_usp} {input.pubchem_usb} {input.kinasesarfari_test} -s {input.pubchem_seq} {input.pubchem_seq} {input.pubchem_seq} {input.pubchem_seq} {input.kinsase_safari_seq} --predict -o {output.prediction} -m {output.model} -a elu """+""" -w {window_size} -d {drug_layer} -f {fc_layer} -r {learning_rate} -V {type} -l {length} -T {threshold} -e {epoch} -H {hidden_protein_layer}""".format(**param_dic)
         print(cmd)
         shell(cmd)



rule evaluate_performance:
     input: prediction =  PREDICTION_DIR+"{paramset_key}_{number}.csv"
     output: performance = OUTPUT_DIR+"{paramset_key}_{number}.csv"
     params: return_params, test_names=test_names
     run:
         param_dic = params[0] 
         cmd = "python evaluate_performance.py {input.prediction} -o {output.performance} -n {params.test_names}" + " -T {threshold}".format(**param_dic)
         print(cmd)
         shell(cmd)
