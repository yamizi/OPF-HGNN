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

    parser.add_argument('-bt', '--batch_train', help="Batch size used in training", type=int, default=256)
    parser.add_argument('-t', '--nb_train', help="Number of graphs used in training", type=int, default=8000)
    parser.add_argument('-v', '--nb_val', help="Number of graphs used in validation", type=int, default=2000)
    parser.add_argument('-s', '--scale', help="Scaling features", type=int, default=0)

    parser.add_argument('-dt', '--dataset_type', help="Which features to log", type=str, default="y_OPF")
    parser.add_argument('-n', '--comet_name', help="Name of the comet project", type=str, default="")
    parser.add_argument('-o', '--opf', help="Scaling features", type=int, default=1)
    parser.add_argument('-r', '--ray', help="Parallelize with ray", type=int, default=0)
    parser.add_argument('-cb', '--clamp_boundary',
                        help="Clamping output; 1 clamp training only, 2 clamp both training and evaluation, 3 clamp validation only", type=int,
                        default=1)
    parser.add_argument('-pl', '--use_physical_loss',
                        help="Whether to include physical loss; 1 report it only, 2 report it and minimize it", type=int,
                        default=1)

    parser.add_argument('-e', '--epochs', help="Max epochs", type=int, default=500)
    parser.add_argument('-hp', '--num_samples', help="Num samples in hyper-param optimization", type=int, default=250)

    parser.add_argument('-cv', '--cv_ratio', help="cross validation ratio", type=float, default=0.2)

    parser.add_argument('-lr', '--base_lr', help="Initial learning rate", type=float, default=0.1)
    parser.add_argument('-dlr', '--decay_lr', help="Learnng rate decay", type=float, default=0.52)
    parser.add_argument('-cl', '--cls', help="Graph layer classes", type=str, default="sage")
    parser.add_argument('-ag', '--aggr', help="Message passing aggregation", type=str, default="mean")
    parser.add_argument('-hc', '--hidden_channels', help="Hidden layers features seprated by :", type=str,
                        default="64:64")
    parser.add_argument('-we', '--weighting', help="Weighting strategy: uniform, relative, adatptive", type=str,
                        default="relative")

    return parser
