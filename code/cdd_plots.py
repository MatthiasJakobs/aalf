import numpy as np
import pandas as pd
from critdd import Diagrams, Diagram
from os import remove

TREATMENT_DICT = {
    'linear': 'Linear',
    'nn': 'Neural Network',
    'selOpt': 'Optimal Selection',
}

DATASET_DICT = {
    'kdd_cup_nomissing': 'KDD Cup 2018',
    'weather': 'Weather',
    'pedestrian_counts': 'Pedestrian Counts',
    'london_smart_meters_nomissing': 'London Smart Meters',
}

def create_cdd(ds_name, drop_columns=None):
    df_test = pd.read_csv(f'results/{ds_name}_test.csv')
    df_test = df_test.set_index('dataset_names')

    # Test if two columns are identical. Heuristically via col-sum
    test_col_sum = df_test.sum(axis=0).to_numpy()
    if len(np.unique(test_col_sum)) != len(test_col_sum):
        print('At least two columns equal in test')
        print(df_test)

    if drop_columns is not None:
        df_test.drop(columns=drop_columns)

    diagram = Diagram(
        df_test.to_numpy(),
        treatment_names = df_test.columns,
        #diagram_names=['validation', 'test'],
        maximize_outcome = False
    )

    diagram.to_file(
        f"{ds_name}.pdf",
        preamble = "\n".join([ # colors are defined before \begin{document}
            "\\definecolor{color1}{HTML}{000000}",
            "\\definecolor{color2}{HTML}{00FF00}",
            "\\definecolor{color3}{HTML}{0000FF}",
            "\\definecolor{color4}{HTML}{FF0000}",
            "\\definecolor{color5}{HTML}{01FFFE}",
            "\\definecolor{color6}{HTML}{FFA6FE}",
            "\\definecolor{color7}{HTML}{FFDB66}",
            "\\definecolor{color8}{HTML}{006401}",
            "\\definecolor{color9}{HTML}{010067}",
            # "\\definecolor{color10}{HTML}{95003A}",
            # "\\definecolor{color11}{HTML}{007DB5}",
            # "\\definecolor{color12}{HTML}{FF00F6}",
            # "\\definecolor{color13}{HTML}{FFEEE8}",
            # "\\definecolor{color14}{HTML}{774D00}",
            # "\\definecolor{color15}{HTML}{90FB92}",
            # "\\definecolor{color16}{HTML}{0076FF}",
            # "\\definecolor{color17}{HTML}{D5FF00}",
            # "\\definecolor{color18}{HTML}{FF937E}",
            # "\\definecolor{color19}{HTML}{6A826C}",
            # "\\definecolor{color20}{HTML}{FF029D}",
            # "\\definecolor{color21}{HTML}{FE8900}",
            # "\\definecolor{color22}{HTML}{7A4782}",
            # "\\definecolor{color23}{HTML}{7E2DD2}",
            # "\\definecolor{color24}{HTML}{85A900}",
            # "\\definecolor{color25}{HTML}{FF0056}",
            # "\\definecolor{color26}{HTML}{A42400}",
            # "\\definecolor{color27}{HTML}{00AE7E}",
            # "\\definecolor{color28}{HTML}{683D3B}",
            # "\\definecolor{color29}{HTML}{BDC6FF}",
            # "\\definecolor{color30}{HTML}{263400}",
            # "\\definecolor{color31}{HTML}{BDD393}",
            # "\\definecolor{color32}{HTML}{00B917}",
            # "\\definecolor{color33}{HTML}{9E008E}",
            # "\\definecolor{color34}{HTML}{001544}",
            # "\\definecolor{color35}{HTML}{C28C9F}",
            # "\\definecolor{color36}{HTML}{FF74A3}",
            # "\\definecolor{color37}{HTML}{01D0FF}",
            # "\\definecolor{color38}{HTML}{004754}",
            # "\\definecolor{color39}{HTML}{E56FFE}",
            # "\\definecolor{color40}{HTML}{788231}",
            # "\\definecolor{color41}{HTML}{0E4CA1}",
            # "\\definecolor{color42}{HTML}{91D0CB}",
            # "\\definecolor{color43}{HTML}{BE9970}",
            # "\\definecolor{color44}{HTML}{968AE8}",
            # "\\definecolor{color45}{HTML}{BB8800}",
            # "\\definecolor{color46}{HTML}{43002C}",
            # "\\definecolor{color47}{HTML}{DEFF74}",
            # "\\definecolor{color48}{HTML}{00FFC6}",
            # "\\definecolor{color49}{HTML}{FFE502}",
            # "\\definecolor{color50}{HTML}{620E00}",
            # "\\definecolor{color51}{HTML}{008F9C}",
            # "\\definecolor{color52}{HTML}{98FF52}",
            # "\\definecolor{color53}{HTML}{7544B1}",
            # "\\definecolor{color54}{HTML}{B500FF}",
            # "\\definecolor{color55}{HTML}{00FF78}",
            # "\\definecolor{color56}{HTML}{FF6E41}",
            # "\\definecolor{color57}{HTML}{005F39}",
            # "\\definecolor{color58}{HTML}{6B6882}",
            # "\\definecolor{color59}{HTML}{5FAD4E}",
            # "\\definecolor{color60}{HTML}{A75740}",
            # "\\definecolor{color61}{HTML}{A5FFD2}",
            # "\\definecolor{color62}{HTML}{FFB167}",
            # "\\definecolor{color63}{HTML}{009BFF}",
            # "\\definecolor{color64}{HTML}{E85EBE}",
        ]),
        axis_options = { # style the plot
            "cycle list": ",".join([ # define the markers for treatments
                "{color1,mark=*}",
                "{color6,mark=*}",
                "{color3,mark=*}",
                # "{color4,mark=*}",
                # "{color5,mark=*}",
                "{color1,mark=diamond*}",
                "{color1,mark=triangle,semithick}",
                "{color6,mark=triangle,semithick}",
                "{color3,mark=triangle,semithick}",
                "{color4,mark=triangle,semithick}",
                "{color1,mark=square,semithick}",
                "{color4,mark=square,semithick}",
                "{color3,mark=square,semithick}",
                # "{color1,mark=pentagon,semithick}",
                # "{color4,mark=pentagon,semithick}",
                # "{color3,mark=pentagon,semithick}",
            ]),
            "width": "\\axisdefaultwidth",
            "height": "0.5*\\axisdefaultheight",
            "title": f"{ds_name.replace('_', ' ')}"
        },
    )

    # Cleanup temp files
    remove(f'{ds_name}.aux')
    remove(f'{ds_name}.log')

def create_all_cdd(drop_columns=None):
    #ds_names = ['london_smart_meters_nomissing', 'kdd_cup_nomissing', 'weather', 'pedestrian_counts']
    ds_names = ['london_smart_meters_nomissing', 'kdd_cup_nomissing']

    Xs = []

    for i, ds_name in enumerate(ds_names):

        df_test = pd.read_csv(f'results/{ds_name}_test.csv')
        df_test = df_test.set_index('dataset_names')

        if drop_columns is not None:
            df_test = df_test.drop(columns=drop_columns, errors='ignore')

        if i == 0:
            treatment_names = df_test.columns

        # Ensure same order of treatments
        df_test = df_test[treatment_names]

        Xs.append(df_test.to_numpy())

    treatment_names = [TREATMENT_DICT.get(key,key) for key in treatment_names.to_list()]
    ds_names = [DATASET_DICT.get(key,key) for key in ds_names]

    print(treatment_names)
    diagram = Diagrams(
        Xs,
        treatment_names = treatment_names,
        diagram_names=ds_names,
        maximize_outcome = False
    )

    diagram.to_file(
        f"all_datasets.pdf",
        preamble = "\n".join([ # colors are defined before \begin{document}
            "\\definecolor{color1}{HTML}{009ee3}",
            "\\definecolor{color2}{HTML}{e82e82}",
            "\\definecolor{color3}{HTML}{35cdb4}",
            "\\definecolor{color4}{HTML}{4a4ad8}",
            "\\definecolor{color5}{HTML}{ec6469}",
            "\\definecolor{color6}{HTML}{ffbc29}",
            # "\\definecolor{color1}{HTML}{000000}",
            # "\\definecolor{color2}{HTML}{00FF00}",
            # "\\definecolor{color3}{HTML}{0000FF}",
            # "\\definecolor{color4}{HTML}{FF0000}",
            # "\\definecolor{color5}{HTML}{01FFFE}",
            # "\\definecolor{color6}{HTML}{FFA6FE}",
            # "\\definecolor{color7}{HTML}{FFDB66}",
            # "\\definecolor{color8}{HTML}{006401}",
            # "\\definecolor{color9}{HTML}{010067}",
            # "\\definecolor{color10}{HTML}{95003A}",
            # "\\definecolor{color11}{HTML}{007DB5}",
            # "\\definecolor{color12}{HTML}{FF00F6}",
            # "\\definecolor{color13}{HTML}{FFEEE8}",
            # "\\definecolor{color14}{HTML}{774D00}",
            # "\\definecolor{color15}{HTML}{90FB92}",
            # "\\definecolor{color16}{HTML}{0076FF}",
            # "\\definecolor{color17}{HTML}{D5FF00}",
            # "\\definecolor{color18}{HTML}{FF937E}",
            # "\\definecolor{color19}{HTML}{6A826C}",
            # "\\definecolor{color20}{HTML}{FF029D}",
            # "\\definecolor{color21}{HTML}{FE8900}",
            # "\\definecolor{color22}{HTML}{7A4782}",
            # "\\definecolor{color23}{HTML}{7E2DD2}",
            # "\\definecolor{color24}{HTML}{85A900}",
            # "\\definecolor{color25}{HTML}{FF0056}",
            # "\\definecolor{color26}{HTML}{A42400}",
            # "\\definecolor{color27}{HTML}{00AE7E}",
            # "\\definecolor{color28}{HTML}{683D3B}",
            # "\\definecolor{color29}{HTML}{BDC6FF}",
            # "\\definecolor{color30}{HTML}{263400}",
            # "\\definecolor{color31}{HTML}{BDD393}",
            # "\\definecolor{color32}{HTML}{00B917}",
            # "\\definecolor{color33}{HTML}{9E008E}",
            # "\\definecolor{color34}{HTML}{001544}",
            # "\\definecolor{color35}{HTML}{C28C9F}",
            # "\\definecolor{color36}{HTML}{FF74A3}",
            # "\\definecolor{color37}{HTML}{01D0FF}",
            # "\\definecolor{color38}{HTML}{004754}",
            # "\\definecolor{color39}{HTML}{E56FFE}",
            # "\\definecolor{color40}{HTML}{788231}",
            # "\\definecolor{color41}{HTML}{0E4CA1}",
            # "\\definecolor{color42}{HTML}{91D0CB}",
            # "\\definecolor{color43}{HTML}{BE9970}",
            # "\\definecolor{color44}{HTML}{968AE8}",
            # "\\definecolor{color45}{HTML}{BB8800}",
            # "\\definecolor{color46}{HTML}{43002C}",
            # "\\definecolor{color47}{HTML}{DEFF74}",
            # "\\definecolor{color48}{HTML}{00FFC6}",
            # "\\definecolor{color49}{HTML}{FFE502}",
            # "\\definecolor{color50}{HTML}{620E00}",
            # "\\definecolor{color51}{HTML}{008F9C}",
            # "\\definecolor{color52}{HTML}{98FF52}",
            # "\\definecolor{color53}{HTML}{7544B1}",
            # "\\definecolor{color54}{HTML}{B500FF}",
            # "\\definecolor{color55}{HTML}{00FF78}",
            # "\\definecolor{color56}{HTML}{FF6E41}",
            # "\\definecolor{color57}{HTML}{005F39}",
            # "\\definecolor{color58}{HTML}{6B6882}",
            # "\\definecolor{color59}{HTML}{5FAD4E}",
            # "\\definecolor{color60}{HTML}{A75740}",
            # "\\definecolor{color61}{HTML}{A5FFD2}",
            # "\\definecolor{color62}{HTML}{FFB167}",
            # "\\definecolor{color63}{HTML}{009BFF}",
            # "\\definecolor{color64}{HTML}{E85EBE}",
        ]),
        axis_options = { # style the plot
            "cycle list": ",".join([ # define the markers for treatments
                "{color1,mark=*}",
                "{color2,mark=*}",
                "{color3,mark=*}",
                # "{color4,mark=*}",
                # "{color5,mark=*}",
                "{color4,mark=square}",
                "{color5,mark=square}",
                "{color6,mark=triangle,semithick}",
                # "{color6,mark=triangle,semithick}",
                # "{color3,mark=triangle,semithick}",
                # "{color4,mark=triangle,semithick}",
                # "{color1,mark=square,semithick}",
                # "{color4,mark=square,semithick}",
                # "{color3,mark=square,semithick}",
                # "{color1,mark=pentagon,semithick}",
                # "{color4,mark=pentagon,semithick}",
                # "{color3,mark=pentagon,semithick}",
            ]),
            "width": "\\axisdefaultwidth",
            "height": "0.5*\\axisdefaultheight",
            "title": f""
        },
    )

    # Cleanup temp files
    remove(f'all_datasets.aux')
    remove(f'all_datasets.log')

def create_cdd_overall(drop_columns=None):
    ds_names = ['london_smart_meters_nomissing', 'kdd_cup_nomissing', 'weather', 'pedestrian_counts']

    Xs = []

    for i, ds_name in enumerate(ds_names):

        df_test = pd.read_csv(f'results/{ds_name}_test.csv')
        df_test = df_test.set_index('dataset_names')

        if drop_columns is not None:
            df_test = df_test.drop(columns=drop_columns, errors='ignore')

        if i == 0:
            treatment_names = df_test.columns

        # Ensure same order of treatments
        df_test = df_test[treatment_names]


        Xs.append(df_test.to_numpy())


    treatment_names = [TREATMENT_DICT.get(key,key) for key in treatment_names.to_list()]
    ds_names = [DATASET_DICT.get(key,key) for key in ds_names]

    print(treatment_names)
    diagram = Diagram(
        np.concatenate(Xs, axis=0),
        treatment_names = treatment_names,
        maximize_outcome = False
    )

    diagram.to_file(
        f"total.pdf",
        preamble = "\n".join([ # colors are defined before \begin{document}
            "\\definecolor{color1}{HTML}{000000}",
            "\\definecolor{color2}{HTML}{00FF00}",
            "\\definecolor{color3}{HTML}{0000FF}",
            "\\definecolor{color4}{HTML}{FF0000}",
            "\\definecolor{color5}{HTML}{01FFFE}",
            "\\definecolor{color6}{HTML}{FFA6FE}",
            "\\definecolor{color7}{HTML}{FFDB66}",
            "\\definecolor{color8}{HTML}{006401}",
            "\\definecolor{color9}{HTML}{010067}",
            # "\\definecolor{color10}{HTML}{95003A}",
            # "\\definecolor{color11}{HTML}{007DB5}",
            # "\\definecolor{color12}{HTML}{FF00F6}",
            # "\\definecolor{color13}{HTML}{FFEEE8}",
            # "\\definecolor{color14}{HTML}{774D00}",
            # "\\definecolor{color15}{HTML}{90FB92}",
            # "\\definecolor{color16}{HTML}{0076FF}",
            # "\\definecolor{color17}{HTML}{D5FF00}",
            # "\\definecolor{color18}{HTML}{FF937E}",
            # "\\definecolor{color19}{HTML}{6A826C}",
            # "\\definecolor{color20}{HTML}{FF029D}",
            # "\\definecolor{color21}{HTML}{FE8900}",
            # "\\definecolor{color22}{HTML}{7A4782}",
            # "\\definecolor{color23}{HTML}{7E2DD2}",
            # "\\definecolor{color24}{HTML}{85A900}",
            # "\\definecolor{color25}{HTML}{FF0056}",
            # "\\definecolor{color26}{HTML}{A42400}",
            # "\\definecolor{color27}{HTML}{00AE7E}",
            # "\\definecolor{color28}{HTML}{683D3B}",
            # "\\definecolor{color29}{HTML}{BDC6FF}",
            # "\\definecolor{color30}{HTML}{263400}",
            # "\\definecolor{color31}{HTML}{BDD393}",
            # "\\definecolor{color32}{HTML}{00B917}",
            # "\\definecolor{color33}{HTML}{9E008E}",
            # "\\definecolor{color34}{HTML}{001544}",
            # "\\definecolor{color35}{HTML}{C28C9F}",
            # "\\definecolor{color36}{HTML}{FF74A3}",
            # "\\definecolor{color37}{HTML}{01D0FF}",
            # "\\definecolor{color38}{HTML}{004754}",
            # "\\definecolor{color39}{HTML}{E56FFE}",
            # "\\definecolor{color40}{HTML}{788231}",
            # "\\definecolor{color41}{HTML}{0E4CA1}",
            # "\\definecolor{color42}{HTML}{91D0CB}",
            # "\\definecolor{color43}{HTML}{BE9970}",
            # "\\definecolor{color44}{HTML}{968AE8}",
            # "\\definecolor{color45}{HTML}{BB8800}",
            # "\\definecolor{color46}{HTML}{43002C}",
            # "\\definecolor{color47}{HTML}{DEFF74}",
            # "\\definecolor{color48}{HTML}{00FFC6}",
            # "\\definecolor{color49}{HTML}{FFE502}",
            # "\\definecolor{color50}{HTML}{620E00}",
            # "\\definecolor{color51}{HTML}{008F9C}",
            # "\\definecolor{color52}{HTML}{98FF52}",
            # "\\definecolor{color53}{HTML}{7544B1}",
            # "\\definecolor{color54}{HTML}{B500FF}",
            # "\\definecolor{color55}{HTML}{00FF78}",
            # "\\definecolor{color56}{HTML}{FF6E41}",
            # "\\definecolor{color57}{HTML}{005F39}",
            # "\\definecolor{color58}{HTML}{6B6882}",
            # "\\definecolor{color59}{HTML}{5FAD4E}",
            # "\\definecolor{color60}{HTML}{A75740}",
            # "\\definecolor{color61}{HTML}{A5FFD2}",
            # "\\definecolor{color62}{HTML}{FFB167}",
            # "\\definecolor{color63}{HTML}{009BFF}",
            # "\\definecolor{color64}{HTML}{E85EBE}",
        ]),
        axis_options = { # style the plot
            "cycle list": ",".join([ # define the markers for treatments
                "{color1,mark=*}",
                "{color6,mark=*}",
                "{color3,mark=*}",
                # "{color4,mark=*}",
                # "{color5,mark=*}",
                "{color1,mark=diamond*}",
                "{color1,mark=triangle,semithick}",
                "{color6,mark=triangle,semithick}",
                "{color3,mark=triangle,semithick}",
                "{color4,mark=triangle,semithick}",
                "{color1,mark=square,semithick}",
                "{color4,mark=square,semithick}",
                "{color3,mark=square,semithick}",
                # "{color1,mark=pentagon,semithick}",
                # "{color4,mark=pentagon,semithick}",
                # "{color3,mark=pentagon,semithick}",
            ]),
            "width": "\\axisdefaultwidth",
            "height": "0.5*\\axisdefaultheight",
            "title": f"Over all datasets"
        },
    )

    # Cleanup temp files
    remove(f'total.aux')
    remove(f'total.log')

def main():
    # print('create weather cup cdd')
    # create_cdd('weather')
    # print('create kdd cup cdd')
    # create_cdd('kdd_cup_nomissing')
    # print('create pedestrian counts cdd')
    # create_cdd('pedestrian_counts')
    print('create london smart meters cdd')
    create_cdd('london_smart_meters_nomissing', drop_columns=['v4_0.5_calibrated'])
    exit()

    create_all_cdd(drop_columns=['selBinom0.9', 'selBinom0.95', 'selBinom0.99', 'v4_0.4', 'v4_0.3'])
    #create_cdd_overall(drop_columns=['selBinom0.9', 'selBinom0.95', 'selBinom0.99', 'v4_0.4', 'v4_0.3'])

if __name__ == '__main__':
    main()