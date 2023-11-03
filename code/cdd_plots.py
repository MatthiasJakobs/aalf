import numpy as np
import pandas as pd
from critdd import Diagrams
from os import remove

def create_cdd(ds_name):
    df_val = pd.read_csv(f'results/{ds_name}_val.csv')
    df_val = df_val.set_index('dataset_names')
    df_test = pd.read_csv(f'results/{ds_name}_test.csv')
    df_test = df_test.set_index('dataset_names')

    diagram = Diagrams(
        np.stack([df_val.to_numpy(), df_test.to_numpy()]),
        treatment_names = df_val.columns,
        diagram_names=['validation', 'test'],
        maximize_outcome = False
    )

    diagram.to_file(
        f"{ds_name}.pdf",
        preamble = "\n".join([ # colors are defined before \begin{document}
            "\\definecolor{color1}{HTML}{84B818}",
            "\\definecolor{color2}{HTML}{D18B12}",
            "\\definecolor{color3}{HTML}{1BB5B5}",
            "\\definecolor{color4}{HTML}{F85A3E}",
            "\\definecolor{color5}{HTML}{4B6CFC}",
        ]),
        axis_options = { # style the plot
            "cycle list": ",".join([ # define the markers for treatments
                "{color1,mark=*}",
                "{color2,mark=diamond*}",
                "{color3,mark=triangle,semithick}",
                "{color4,mark=square,semithick}",
                "{color5,mark=pentagon,semithick}",
            ]),
            "width": "\\axisdefaultwidth",
            "height": "0.5*\\axisdefaultheight",
            "title": f"{ds_name.replace('_', ' ')}"
        },
    )

    # Cleanup temp files
    remove(f'{ds_name}.aux')
    remove(f'{ds_name}.log')