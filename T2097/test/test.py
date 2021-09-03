import numpy as np
import pandas as pd

if __name__ == "__main__":
    path_out = "/opt/ml/image-classification-level1-20/T2097/test/output.csv" # output csv

    p = [
        [0.2, 0.2, 0.2], # mask
        [0.4, 0.15, 0.15], # gender
        [0.3, 0.3, 0.3] # age
    ]

    df = pd.DataFrame(p)
    df.columns = ['ImageID', 'ans', 'ans']
    df.to_csv(path_out, index=False)