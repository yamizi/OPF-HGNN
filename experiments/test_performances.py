import sys
sys.path.append(".")
from utils.io import get_parser
import hashlib
from utils.logging import init_comet
from runs.diff_distribution import run_case

parser = get_parser()

def run(mutations = ["cost", "load_relative"],cases = ["case9","case14","case30","case118"], 
        nb_train = 8000,nb_val = 2000, dataset_type="y_OPF", device="cuda", scale=0,
        opf=1, project_name="test_perf_v4",use_ray=1):

    for mutation in mutations:
        for case in cases:
            experiment = init_comet({"case":case, "mutation":mutation},project_name)
            training_case=[[case,nb_train,0.7,[mutation]]]
            validation_case=[case,nb_val,0.7,[mutation]]
            path = "./output/test_perf"
            hash_path = f"{training_case}_{validation_case}"
            hash_path = hashlib.md5(hash_path.encode()).hexdigest()

            run_case(training_cases=training_case,validation_case=validation_case, plot=False,
                     title="Test performance on "+mutation, save_path=path, dataset_type=dataset_type,
                     experiment=experiment,train_batch_size=256,val_batch_size=512,scale=scale,
                     device=device, opf=opf,use_ray=use_ray,uniqueid=hash_path)


if __name__ == "__main__":
    args = parser.parse_args()
    mutations = args.mutations.split("+")
    cases = args.cases.split("+")
    comet_name = args.comet_name if args.comet_name!="" else "test_perf_a1"
    run(mutations,cases, args.nb_train,args.nb_val, device=args.device, scale=args.scale,
        dataset_type=args.dataset_type,opf=args.opf, project_name=comet_name,use_ray =args.ray)
