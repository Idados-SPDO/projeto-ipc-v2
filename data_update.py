import pandas as pd

def atualizar_base_incremental(df_atual: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    colunas_datas_atual = df_atual.columns[3:]
    colunas_datas_new = new_df.columns[3:]
    
    novas_colunas = [col for col in colunas_datas_new if col not in colunas_datas_atual]
    
    if novas_colunas:
        chaves = ["UF", "Código", "Descrição"]
        df_atual = df_atual.set_index(chaves)
        new_data = new_df.set_index(chaves)[novas_colunas]
        
        df_atual = df_atual.join(new_data, how="left")
        df_atual = df_atual.reset_index()
    
    return df_atual
