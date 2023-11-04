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
                            default="cost+load+load_relative",
                            type=str) 

    parser.add_argument('-c', '--cases',
                            help="Cases separated by +",
                            default="case9+case14+case30+case118",
                            type=str) 

    parser.add_argument('-d', '--device',
                            help="Device (cpu or cuda)",
                            default="cpu",
                            type=str) 

    parser.add_argument('-t','--nb_train', help="Number of graphs used in training", type=int, default=800)
    parser.add_argument('-v','--nb_val', help="Number of graphs used in validation", type=int, default=200)
    parser.add_argument('-s','--scale', help="Scaling features", type=int, default=0)

    return parser