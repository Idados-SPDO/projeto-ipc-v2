import pandas as pd
import re
import datetime
import streamlit as st
from config import SHEET_NAMES

@st.cache_data
def read_excel_file(uploaded_file) -> pd.DataFrame:
    """Lê e processa o arquivo Excel principal com várias abas."""
    lista_dfs = []
    for sheet in SHEET_NAMES:
        df_sheet = pd.read_excel(
            uploaded_file,
            sheet_name=sheet,
            skiprows=6
        )
        df_sheet["Capital"] = sheet
        lista_dfs.append(df_sheet)
    
    df = pd.concat(lista_dfs, ignore_index=True)
    df = df.dropna(axis=1, how='all')
    df.rename(columns={'Capital': 'UF'}, inplace=True)
    df.columns = [re.sub(r'\s*\(Q.*\)', '', col) for col in df.columns]

    data_atual = datetime.datetime.now()
    cols_to_keep = []
    for col in df.columns:
        match = re.search(r'(\d{2}/\d{4})', col)
        if match:
            try:
                data_coluna = datetime.datetime.strptime(match.group(1), "%m/%Y")
                if data_coluna.year >= 2024 and data_coluna <= data_atual:
                    cols_to_keep.append(col)
            except Exception:
                pass
        else:
            cols_to_keep.append(col)
    df = df[cols_to_keep]
    return df

@st.cache_data
def read_excel_excess_file(upload_file) -> pd.DataFrame:
    """Lê e processa o arquivo Excel de excessões."""
    df_sheet = pd.read_excel(upload_file, sheet_name='itens com excessões')
    df_excecao = df_sheet[df_sheet["excessão"].notna()][["DESCRIÇÃO"]]
    return df_excecao



def read_excel_service_file(upload_file) -> pd.DataFrame:
    """Lê e processa o arquivo Excel de serviços."""
    df_sheet = pd.read_excel(upload_file, sheet_name='itens com excessões')
    df_servicos = df_sheet[
        (df_sheet["serviços?"] == "Serviço") & 
        (df_sheet["excessão"].isna())
    ][["DESCRIÇÃO"]]
    return df_servicos

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

@st.cache_data
def read_excel_weight_file(upload_file) -> pd.DataFrame:
    xls = pd.ExcelFile(upload_file)
    sheets = xls.sheet_names
    list_dfs = []

    for sheet in sheets:
        df_sheet = pd.read_excel(upload_file, sheet_name=sheet, skiprows=1)
        df_sem_coluna = df_sheet.drop('Cód.Série - Nr.Índice', axis=1)
        novas_colunas = df_sem_coluna.columns.tolist()
        for i in range(1, len(novas_colunas)):
            novas_colunas[i] = novas_colunas[i].split(' - ')[0]
        df_sem_coluna.columns = novas_colunas
        df_sem_coluna["Cód.Estrutura"] = df_sem_coluna["Cód.Estrutura"].astype(str)
        df_sem_coluna['Descrição'] = df_sem_coluna['Descrição'].str.replace(r'^\.*', '', regex=True)
        df_sem_coluna = df_sem_coluna.iloc[1:]
        df_sem_coluna["UF"] = sheet
        
        cols = df_sem_coluna.columns.tolist()
        if "Descrição" in cols:
            cols.remove("UF")
            idx = cols.index("Descrição")
            cols.insert(idx+1, "UF")
            df_sem_coluna = df_sem_coluna[cols]
        
        list_dfs.append(df_sem_coluna)
    
    return pd.concat(list_dfs)

