"""
This is a boilerplate pipeline 'TratamentoDados'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes.filtrar_dados,
                inputs="dataset_kobe_dev",
                outputs="data_filtered",
                name="filtrar_dados_node",
            ),
            node(
                func=nodes.separar_treino_teste,
                inputs="data_filtered",
                outputs=["base_train", "base_test"],
                name="split_treino_teste_node",
            ),
        ]
    )