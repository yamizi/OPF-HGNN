import sys
sys.path.append(".")
from utils.io import get_parser

from utils.logging import init_comet
from diff_distribution import run_case

parser = get_parser()
parser.add_argument('-s', '--source_cases',
                    help="Cases separated by +",
                    default="case9+case14+case30+case118",
                    type=str) 

def run(mutations = ["cost", "load_relative"], cases = ["case9","case14","case30","case118"],
        source_cases = ["case9","case14","case30","case118"], 
        nb_train = 8000,nb_val = 2000, dataset_type="y_no_OPF", device="cuda"):

    for mutation in mutations:
        for source_case in source_cases:
            for case in cases:
                if source_case==case:
                    continue
                experiment = init_comet({"source_case":source_case,"case":case, "mutation":mutation},"test_gen_topology")
                training_case=[[source_case,nb_train,0.7,[mutation]]]
                validation_case=[case,nb_val,0.7,[mutation]]
                path = "./output/test_gen_topology/"+mutation+"/"+case+"/"+source_case
                run_case(training_cases=training_case,validation_case=validation_case, plot=False,
                        title="Test generalization on "+mutation, save_path=path, dataset_type=dataset_type,
                        experiment=experiment,train_batch_size=128,val_batch_size=256,scale=False,
                        device=device)


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    source_cases = args.source_cases.split("+")
    
    run(mutations,cases, source_cases, args.nb_train,args.nb_val, device=args.device)
