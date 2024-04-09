import numpy as np

MIN_PQ = -200
MAX_PQ = 1800


def normalizeCols(dataset, case="", columns=["p_mw", "q_mvar"], min_val=MIN_PQ, max_val=MAX_PQ):
    dts = []
    columns_to_affect = [e for e in columns if e in dataset.columns]
    if len(columns_to_affect) == 0:
        return dataset

    max_vals = dataset[columns_to_affect].max()
    min_vals = dataset[columns_to_affect].min()

    if case == "":
        min_vals = min_val
        max_vals = max_val
    elif case == "case13" or case == "case123":
        min_vals = min_val
        max_vals = 5000.0 / 3  # Base kVA
    elif case == "case8500":
        min_vals = min_val
        max_vals = 27.5 * 1000 / 3  # Base kVA

    dataset[columns_to_affect] = (dataset[columns_to_affect] - min_vals) / (max_vals - min_vals)
    return dataset
