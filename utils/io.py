import json
import argparse
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_json'):
            return obj.to_json(orient='records')
        return json.JSONEncoder.default(self, obj)
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mutations',
                            help="Mutations separated by +",
                            default="cost+load_relative",
                            type=str) 

    parser.add_argument('-c', '--cases',
                            help="Cases separated by +",
                            default="case9+case14",
                            type=str) 

    parser.add_argument('-d', '--device',
                            help="Device (cpu or cuda)",
                            default="cuda",
                            type=str) 

    parser.add_argument('-t','--nb_train', help="Number of graphs used in training", type=int, default=8000)
    parser.add_argument('-v','--nb_val', help="Number of graphs used in validation", type=int, default=2000)
    parser.add_argument('-s','--scale', help="Scaling features", type=int, default=0)

    parser.add_argument('-dt','--dataset_type', help="Which features to log", type=str, default="y_OPF")
    parser.add_argument('-n','--comet_name', help="Name of the comet project", type=str, default="")
    parser.add_argument('-o','--opf', help="Scaling features", type=int, default=1)
    parser.add_argument('-r', '--ray', help="Parallelize with ray", type=int, default=0)
    parser.add_argument('-e', '--epochs', help="Max epochs", type=int, default=500)
    return parser