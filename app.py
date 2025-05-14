import streamlit as st
import pandas as pd
import duckdb
import re
import numpy as np

from config import *
from utils import to_excel, highlight_values, get_criticidade
from data_processing import (
    read_excel_file,
    read_excel_excess_file,
    read_excel_weight_file,
    read_excel_service_file
)
from visualizations import plot_time_series
from data_update import atualizar_base_incremental

# ----- Funções de Banco de Dados e Carregamento dos Dados -----
def load_database(df: pd.DataFrame, table_name: str, con):
    """Registra o DataFrame no DuckDB, criando (ou recriando) a tabela."""
    con.register("df_excel", df)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_excel")

def prepare_base_data(con, table_name, excess_table_name, weight_table_name, service_table_name):
    """
    Consulta as tabelas no banco de dados e prepara os DataFrames:
      - df: Dados originais (incluindo agregação nacional - BR).
      - df_melted: Dados no formato 'long' para análise de séries.
      - df_excess: Itens de exceção.
      - df_weight: Tabela de ponderações.
      - colunas_datas: Colunas com datas.
    """
    query = f"SELECT * FROM {table_name}"
    excess_query = f"SELECT * FROM {excess_table_name}"
    weight_query = f"SELECT * FROM {weight_table_name}"
    service_query = f"SELECT * FROM {service_table_name}"
    
    df = con.execute(query).fetchdf()
    df_excess = con.execute(excess_query).fetchdf()
    df_weight = con.execute(weight_query).fetchdf()
    df_service = con.execute(service_query).fetchdf()
    
    # Consolida os dados com a agregação nacional (BR)
    colunas_datas = df.columns[3:]
    df_br = df.groupby(["Código", "Descrição"], as_index=False)[colunas_datas].sum()
    df_br["UF"] = "BR"
    df_br = df_br[["UF", "Código", "Descrição"] + list(colunas_datas)]
    df = pd.concat([df, df_br], ignore_index=True)
    
    # Transforma os dados para formato 'long'
    df_melted = df.melt(
        id_vars=["UF", "Código", "Descrição"],
        value_vars=colunas_datas,
        var_name="Data",
        value_name="Valor"
    )
    df_melted["Valor"] = pd.to_numeric(df_melted["Valor"], errors="coerce")
    df_melted["CodigoDescricao"] = (
        df_melted["Código"].astype(str) + " - " + df_melted["Descrição"].astype(str)
    )
    
    # Verifica exceções (itens que não entram no cálculo) pela coluna CodigoDescricao
    if not df_excess.empty:
        excess_set = set(df_excess["DESCRIÇÃO"].dropna())
        df_melted["Exceção"] = df_melted["CodigoDescricao"].apply(
            lambda x: any(exc in x for exc in excess_set)
        )
    else:
        df_melted["Exceção"] = False
        
    if not df_service.empty:
        service_set = set(df_service["DESCRIÇÃO"].dropna())
        df_melted["Serviço"] = df_melted["CodigoDescricao"].apply(
            lambda x: any(serv in x for serv in service_set)
        )
    else:
        df_melted["Serviço"] = False

    return df, df_melted, df_excess, df_weight, df_service,colunas_datas

def compute_comparative_data(df_melted):
    """
    Agrupa os dados (exceto os itens de exceção) para exibição na visão de status.
    Calcula totais, SuperCrítico, Crítico, Aceitável, Suficiente e Exceção.
    """
    mask_status = (~df_melted["Exceção"]) & (~df_melted["Serviço"])
    df_totais = df_melted.groupby(["UF", "Data"])["Valor"].count().rename("Total")
    df_super_critico = df_melted[mask_status & (df_melted["Valor"] <= 25)] \
                        .groupby(["UF", "Data"])["Valor"].count().rename("SuperCrítico")
    df_critico = df_melted[mask_status & (df_melted["Valor"].between(26, 55))] \
                   .groupby(["UF", "Data"])["Valor"].count().rename("Crítico")
    df_aceitavel = df_melted[mask_status & (df_melted["Valor"].between(56, 100))] \
                    .groupby(["UF", "Data"])["Valor"].count().rename("Aceitável")
    df_suficiente = df_melted[mask_status & (df_melted["Valor"] > 100)] \
                     .groupby(["UF", "Data"])["Valor"].count().rename("Suficiente")
    df_excessao = df_melted[df_melted["Exceção"]] \
                   .groupby(["UF", "Data"])["Valor"].count().rename("Exceção")
    df_servico = df_melted[df_melted["Serviço"]] \
               .groupby(["UF", "Data"])["Valor"].count().rename("Serviços")
    
    df_comparativo = pd.concat([df_totais, df_super_critico, df_critico,
                                df_aceitavel, df_suficiente, df_excessao, df_servico], axis=1) \
                      .fillna(0).reset_index()
    
    df_comparativo["Data_dt"] = pd.to_datetime(df_comparativo["Data"], format="%m/%Y", errors="coerce")
    return df_comparativo

def prepare_quantity_table(df, df_excess, df_service):
    """
    Prepara o DataFrame que será usado na aba "Controle de Cotações".
    Cria as colunas: CodigoDescricao, UF (normalizada), Grupo e Exceção.
    """
    df_tab = df.copy()
    df_tab["CodigoDescricao"] = df_tab["Código"].astype(str) + " - " + df_tab["Descrição"].astype(str)
    df_tab["UF"] = df_tab["UF"].str.strip().str.upper()
    df_tab["Grupo"] = df_tab["Código"].astype(str).str[:4]

    if not df_excess.empty:
        excess_set = set(df_excess["DESCRIÇÃO"].dropna())
        df_tab["Exceção"] = df_tab["CodigoDescricao"].apply(lambda x: any(exc in x for exc in excess_set))
    else:
        df_tab["Exceção"] = False
        
    if not df_service.empty:
        service_set = set(df_service["DESCRIÇÃO"].dropna())
        df_tab["Serviço"] = df_tab["CodigoDescricao"].apply(lambda x: any(serv in x for serv in service_set))
    else:
        df_tab["Serviço"] = False

    return df_tab

# ----- Funções de Estilo para as Tabelas -----
def style_quantidade(val):
    if pd.isna(val):
        return ""
    critic = get_criticidade(val)
    style_map = {
        "SuperCrítico": "background-color: #ff4d4d; color: black;",
        "Crítico": "background-color: #ffa500; color: black;",
        "Aceitável": "background-color: #FCDA51; color: black;",
        "Suficiente": "background-color: #66cc66; color: black;",
        "Exceção": "background-color: gray; color: black;"
    }
    return style_map.get(critic, "")

def style_locked_quantidade(val):
    try:
        numeric_val = float(val)
        return style_quantidade(numeric_val)
    except Exception:
        return ""
    
            
def compute_statistics_from_pivot(pivot_df):
                """
                Recebe um pivot table (com índices = CodigoDescricao e colunas = UFs) contendo valores numéricos.
                Converte para formato longo, filtra os valores cujas classificações, segundo get_criticidade, 
                sejam diferentes de "Exceção" e "Suficiente" e retorna os valores do primeiro quartil, 
                mediana e terceiro quartil agrupados por UF.
                """
                # Converte de pivot para formato longo
                id_col = pivot_df.index.name if pivot_df.index.name is not None else "index"
                df_long = pivot_df.reset_index().melt(id_vars=id_col, var_name="UF", value_name="Valor")

                # Remove linhas que tenham o hífen (ou seja, em que não haja valor numérico)
                df_long = df_long[df_long["Valor"] != '-']
                
                # Converte os valores para numérico
                df_long["Valor"] = pd.to_numeric(df_long["Valor"], errors="coerce")
                
                # Cria uma nova coluna com a classificação usando get_criticidade
                df_long["Criticidade"] = df_long["Valor"].apply(get_criticidade)
                df_long = df_long.dropna(subset=["Valor"])
                
                # Filtra para manter apenas os valores que NÃO sejam "Exceção" ou "Suficiente"
                df_valid = df_long[~df_long["Criticidade"].isin(["Exceção", "Suficiente"])]
                
                # Para cada UF, calcula:
                #   - Q1 (25%): primeiro quartil
                #   - Mediana (50%)
                #   - Q3 (75%): terceiro quartil
                stats = df_valid.groupby("UF")["Valor"].agg(
                    Q1=lambda x: x.quantile(0.25),
                    Median="median",
                    Mean = "mean",
                    Q3=lambda x: x.quantile(0.75)
                )
                return stats


# ----- Funções para Filtragem e Construção da Tabela -----
def filter_quantity_data(df_tab ,input_capital, selected_item, selected_group, selected_criticidade,date_cols):
    df_filtered = df_tab.copy()
    
    if input_capital:
        df_filtered = df_filtered[df_filtered["UF"].isin(input_capital)]
    if selected_item:
        df_filtered = df_filtered[df_filtered["CodigoDescricao"].isin(selected_item)]
    if selected_group:
        df_filtered = df_filtered[df_filtered["Grupo"].isin(selected_group)]
    
    if selected_criticidade:
        # Se date_cols for uma string, converte para lista para iterar corretamente.
        cols_to_check = [date_cols] if isinstance(date_cols, str) else date_cols
        def row_matches(row):
            if row["Exceção"] and "Exceção" in selected_criticidade:
                return True
            if row.get("Serviço", False) and "Serviços" in selected_criticidade:
                return True
            if not row["Exceção"] and not row.get("Serviço", False):
                for col in cols_to_check:
                    val = row.get(col)
                    if pd.notnull(val):
                        if get_criticidade(val) in selected_criticidade:
                            return True
            return False

        df_filtered = df_filtered[df_filtered.apply(row_matches, axis=1)]
    
    return df_filtered

def filter_weight_data(df_tab ,input_capital, selected_item, selected_group):
    df_filtered = df_tab.copy()
    
    if input_capital:
        df_filtered = df_filtered[df_filtered["UF"].isin(input_capital)]
    if selected_item:
        df_filtered = df_filtered[df_filtered["CodigoDescricao"].isin(selected_item)]
    if selected_group:
        df_filtered = df_filtered[df_filtered["Grupo"].isin(selected_group)]

    
    return df_filtered

def build_index_pivot_table(df, locked_date, df_base):
    """
    Constrói uma tabela pivot de índices a partir da base consolidada (df) e
    utiliza o mapeamento definido abaixo. Usa a tabela df_base (geralmente o df_tab)
    para recuperar o flag 'Exceção' de cada CodigoDescricao.
    """
    # Seleciona as colunas necessárias e renomeia locked_date para 'Valor'
    df_locked = df[["UF", "CodigoDescricao", locked_date, "Exceção"]].copy()
    df_locked.rename(columns={locked_date: "Valor"}, inplace=True)
    # Cria a tabela pivot com a quantidade (Valor)
    df_pivot = df_locked.pivot(index="CodigoDescricao", columns="UF", values="Valor")
    df_pivot = df_pivot.fillna('-')
    if "BR" in df_pivot.columns:
        df_pivot = df_pivot.drop("BR", axis=1)
    
    # Cria um mapeamento de CódigoDescrição para o flag de exceção (usando df_base)
    exception_mapping = df_base.drop_duplicates("CodigoDescricao").set_index("CodigoDescricao")["Exceção"]
    
    # Define o mapeamento de criticidade para os índices (para valores numéricos)
    criticidade_mapping = {
        "Suficiente": 1,
        "Aceitável": 2,
        "Crítico": 3,
        "SuperCrítico": 4,
    }
    
    # Função que converte um valor em seu índice, considerando se é exceção
    def cell_to_index(val, codigo):
        if exception_mapping.get(codigo, False):
            return 0  # Exceção terá índice 0
        if isinstance(val, (int, float)):
            crit = get_criticidade(val)
            return criticidade_mapping.get(crit, np.nan)
        return np.nan

    # Aplica a função para cada célula do pivot, considerando a linha (CodigoDescricao)
    df_index = df_pivot.copy()
    for codigo in df_index.index:
        for col in df_index.columns:
            df_index.loc[codigo, col] = cell_to_index(df_index.loc[codigo, col], codigo)
    return df_index

    

def build_pivot_table(df, locked_date):
    df_locked = df[["UF", "CodigoDescricao", locked_date, "Exceção"]].copy()
    df_locked.rename(columns={locked_date: "Valor"}, inplace=True)
    df_pivot = df_locked.pivot(index="CodigoDescricao", columns="UF", values="Valor")
    df_pivot = df_pivot.fillna('-')
    # Converter valores numéricos sem casas decimais para int
    df_pivot = df_pivot.applymap(lambda x: int(x) if isinstance(x, float) and x.is_integer() else x)
    # Remove a coluna BR, se existir
    if "BR" in df_pivot.columns:
        df_pivot = df_pivot.drop("BR", axis=1)
    df_pivot.index.name = "CodigoDescricao"
    return df_pivot

def build_weight_pivot_table(df, locked_date):
    df_locked = df[["UF", "CodigoDescricao", locked_date]].copy()
    df_locked.rename(columns={locked_date: "Valor"}, inplace=True)
    df_pivot = df_locked.pivot(index="CodigoDescricao", columns="UF", values="Valor")
    df_pivot = df_pivot.fillna('-')
    # Converter valores numéricos sem casas decimais para int
    df_pivot = df_pivot.applymap(lambda x: int(x) if isinstance(x, float) and x.is_integer() else x)
    # Remove a coluna BR, se existir
    if "BR" in df_pivot.columns:
        df_pivot = df_pivot.drop("BR", axis=1)
    # Define o nome do índice
    df_pivot.index.name = "CodigoDescricao"
    
    return df_pivot



def process_weight_data(df_weight: pd.DataFrame):
    """
    Processa a base de ponderações:
      - Filtra os itens com Cód.Estrutura com tamanho >= 6.
      - Formata as colunas numéricas e cria a coluna 'CodigoDescricao' e 'Grupo'.
      - Retorna o DataFrame filtrado, a lista de colunas de data e o dicionário com os limites para formatação.
    """

    df_weight_filtrado = df_weight[df_weight["Cód.Estrutura"].str.len() >= 6].copy()

    df_weight_filtrado["CodigoDescricao"] = (
        df_weight_filtrado["Cód.Estrutura"].astype(str) + " - " + df_weight_filtrado["Descrição"].astype(str)
    ).str.strip()
    
    # Reordena as colunas para garantir que "UF" fique na posição correta
    colunas = df_weight_filtrado.columns.tolist()
    colunas.remove("CodigoDescricao")
    idx_uf = colunas.index("UF")
    colunas.insert(idx_uf, "CodigoDescricao")
    df_weight_filtrado = df_weight_filtrado[colunas]
    
    # Cria a coluna 'Grupo' com os 4 primeiros dígitos do código
    df_weight_filtrado['Cód_Estrutura_4'] = df_weight_filtrado['Cód.Estrutura'].str[:4]
    df_weight_filtrado["Grupo"] = df_weight_filtrado["CodigoDescricao"].str.split(" - ").str[0].str[:4]
    
    # Identifica as colunas de data com padrão 'mm/aaaa'
    date_cols_weight = [col for col in df_weight_filtrado.columns if re.match(r'\d{2}/\d{4}', col)]
    # Converte as colunas de data para numérico
    for col in date_cols_weight:
        df_weight_filtrado[col] = (
            df_weight_filtrado[col]
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df_weight_filtrado[col] = pd.to_numeric(df_weight_filtrado[col], errors='coerce')

    capital_to_uf = {
            "Belo Horizonte": "MG",
            "Brasília": "DF",
            "Porto Alegre": "RS",
            "Recife": "PE",
            "Rio de Janeiro": "RJ",
            "Salvador": "BA",
            "São Paulo": "SP",
            "BR": "BR"
        }
    df_weight_filtrado["UF"] = df_weight_filtrado["UF"].map(capital_to_uf).fillna(df_weight_filtrado["UF"])
    return df_weight_filtrado


# ----- Funções de Interface e Exibição -----
def create_legend():
    """Cria a legenda exibida na sidebar da aplicação."""
    legend_markdown = """
    <style>
        .legend-table {
            width: 90%;
            border-collapse: collapse;
            font-size: 12px;
            margin: auto;
        }
        .legend-table th, .legend-table td {
            border: 1px solid #ddd;
            padding: 4px;
            text-align: center;
        }
        .legend-table th {
            background-color: #f4f4f4;
        }
        .super-critico { background-color: #ff4d4d; color: black; }
        .critico      { background-color: #ffa500; color: black; }
        .aceitavel    { background-color: #FCDA51; color: black; }
        .suficiente   { background-color: #66cc66; color: black; }
        .excessao     { background-color: gray; color: black; }
        .servicos     { background-color: #3C5096; color white}
    </style>
    <p>Legenda de Cores:</p>
    <table class="legend-table">
        <tr>
            <th>Categoria</th>
            <th>Descrição</th>
            <th>Cor</th>
        </tr>
        <tr>
            <td>Supercrítico</td>
            <td>Qtde. de cotações ≤ 25</td>
            <td class="super-critico"></td>
        </tr>
        <tr>
            <td>Crítico</td>
            <td>Qtde. de cotações entre 26 e 55</td>
            <td class="critico"></td>
        </tr>
        <tr>
            <td>Aceitável</td>
            <td>Qtde. de cotações entre 56 e 100</td>
            <td class="aceitavel"></td>
        </tr>
        <tr>
            <td>Suficiente</td>
            <td>Qtde. de cotações > 100</td>
            <td class="suficiente"></td>
        </tr>
        <tr>
            <td>Exceção</td>
            <td>Itens que não entram no cálculo do IPC</td>
            <td class="excessao"></td>
        </tr>
        <tr>
            <td>Serviço</td>
            <td>Itens de serviços (Criticidade por quantidade diferente dos demais itens) </td>
            <td class="servicos"></td>
        </tr>
        
    </table>
    <p>Legenda de Prioridades:</p>
    <table class="legend-table">
        <tr>
            <th>Prioridade</th>
            <th>Descrição</th>
        </tr>
        <tr>
            <td>Prioridade 1</td>
            <td>Ponderação acima de 1,00%</td>
        </tr>
        <tr>
            <td>Prioridade 2</td>
            <td>Ponderação entre 0,40% e 1,00%</td>
        </tr>
        <tr>
            <td>Prioridade 3</td>
            <td>Ponderação ≤ 0,40%</td>
        </tr>
    </table>
    """
    st.sidebar.markdown(legend_markdown, unsafe_allow_html=True)

def display_visao_geral(tab, df_comparativo, target_date, df_tab, df_weight, colunas_datas):
    with tab:
        sub_tab_status, sub_tab_controle = st.tabs([
             "Visão de Status por Quantidade",
             "Controle de Cotações"
        ])
        
        with sub_tab_status:
            st.write(f"### Visão de Status por Quantidade - {target_date}")
            df_recent = df_comparativo[ df_comparativo["Data_dt"] == df_comparativo["Data_dt"].max() ]
            st.dataframe(
                df_recent[["UF", "Data", "Total", "SuperCrítico", "Crítico",
                           "Aceitável", "Suficiente", "Exceção", "Serviços"]]
            )
        
        with sub_tab_controle:
            st.write("### Controle de Cotações")
            capitais = sorted([uf for uf in df_tab["UF"].unique() if uf != "BR"] )
            col1, col2, col3 = st.columns(3)
            selected_capital = col1.selectbox(
                "Selecione a UF (ou BR):",
                options=capitais,
                index=0,
                key="quant_capital"
            )
            input_capital = [selected_capital] if selected_capital else []
            selected_item = col2.multiselect(
                "Selecione os itens:",
                df_tab["CodigoDescricao"].unique(),
                key="quant_item"
            )
            all_dates = list(colunas_datas)
            default_date = all_dates[-1] if len(all_dates) > 0 else None
            selected_date = col3.selectbox(
                "Data de referência:",
                options=all_dates,
                index=len(all_dates)-1,
                disabled=True
            )

            col4, col5, col6 = st.columns(3)
            selected_criticidade = col4.multiselect(
                "Selecione a criticidade:",
                options=["SuperCrítico", "Crítico", "Aceitável", "Suficiente", "Exceção", "Serviços"],
                key="quant_criticidade"
            )
            selected_group = col5.multiselect(
                "Selecione o grupo:",
                sorted(df_tab["Grupo"].unique()),
                key="quant_group"
            )
            selected_prioridade = col6.multiselect(
                "Selecione a prioridade:",
                options=["Prioridade 1", "Prioridade 2", "Prioridade 3"],
                key="quant_prioridade"
            )

            st.write("### Tabela Consolidada")
            df_filtrado = filter_quantity_data(
                df_tab,
                input_capital,
                selected_item,
                selected_group,
                selected_criticidade,
                selected_date or all_dates
            )
            locked_date = default_date  # Data de referência fixada
            
            # Obtém a tabela pivot de quantidade (tabela consolidada)
            df_pivot = build_pivot_table(df_filtrado, locked_date)
            # Aplica o estilo (como no código original)
            exception_map = df_tab.set_index("CodigoDescricao")["Exceção"].to_dict()
            service_map = df_tab.set_index("CodigoDescricao")["Serviço"].to_dict()
            
            df_weight_base = process_weight_data(df_weight)
            df_weight_filtrado = filter_weight_data(df_weight_base, input_capital, selected_item, selected_group)
            
            df_weight_pivot = build_weight_pivot_table(df_weight_filtrado, locked_date)
            
            
            df_q = df_pivot.reset_index()
            df_q["Código"] = df_q["CodigoDescricao"].str.split(" - ").str[0]
            df_w = df_weight_pivot.reset_index()
            df_w["Código"] = df_w["CodigoDescricao"].str.split(" - ").str[0]
            merge = pd.merge(df_q, df_w, on="Código", how="inner")
            
            cols_x = [c for c in merge.columns if c.endswith('_x') and c != 'CodigoDescricao_x']
            cols_y = [c for c in merge.columns if c.endswith('_y') and c != 'CodigoDescricao_y']

            df_sel = merge[['CodigoDescricao_x'] + cols_x + cols_y].copy()
            

            rename_map = {'CodigoDescricao_x': 'CodigoDescricao'}

            for c in cols_x:
                uf = c[:-2] 
                rename_map[c] = f'{uf}_qtd'

            for c in cols_y:
                uf = c[:-2] 
                rename_map[c] = f'{uf}_pond'
                
            df_sel.rename(columns=rename_map, inplace=True)
            
            qtd_cols = [c for c in df_sel.columns if c.endswith("_qtd")]
            pond_cols = [c for c in df_sel.columns if c.endswith("_pond")]
            
            if len(qtd_cols) != 1:
                st.error(
                    "Erro: esperava exatamente 1 coluna de quantidade, mas encontrou: "
                    + str(qtd_cols)
                )
            else:
                qtd_col = qtd_cols[0]
                pond_col = pond_cols[0]
                # 3) Cria Criticidade, mas força "Exceção" quando apropriado
                df_sel["Criticidade"] = df_sel.apply(
                    lambda row: "Exceção" 
                                if exception_map.get(row["CodigoDescricao"], False) 
                                else get_criticidade(row[qtd_col]),
                    axis=1
                )
                df_sel["Prioridade"] = df_sel[pond_col].apply(
                    lambda x: "Prioridade 1" if x > 1
                            else ("Prioridade 2" if x > 0.4 else "Prioridade 3")
                )

             
                severity_map = {
                    "SuperCrítico": 4,
                    "Crítico":      3,
                    "Aceitável":    2,
                    "Suficiente":   1,
                    "Exceção":      0, 
                }
                df_sel["CriticidadeNivel"] = df_sel["Criticidade"].map(severity_map)
                
                if selected_prioridade:
                    df_sel = df_sel[df_sel["Prioridade"].isin(selected_prioridade)]

                # 5) Ordena primeiro por CriticidadeNivel, depois por quantidade
                df_sorted = df_sel.sort_values(
                    by=["CriticidadeNivel", pond_col],
                    ascending=[False, False]
                ).reset_index(drop=True)
                

                df_sorted["Falta p/ Cobertura Mínima"] = df_sorted[qtd_col].apply(lambda x: max(0, 100 - x))

                # 6) Exclui qualquer coluna que termine com '_pond' antes de exibir
                cols_display = [
                    c for c in df_sorted.columns
                    if c not in ("Criticidade", "CriticidadeNivel")
                    and not c.endswith("_pond")
                ]
                # Mantém 'CodigoDescricao' sempre à frente
                ordered = ["CodigoDescricao"] + [
                    c for c in cols_display if c != "CodigoDescricao"
                ]
                df_display = df_sorted[ordered]


                def style_full_row(row):
                    styles = []
                    for col in df_display.columns:
                        # só estiliza as colunas de quantidade
                        if col.endswith("_qtd"):
                            if service_map.get(row["CodigoDescricao"], False):
                                # serviços: roxo
                                styles.append("background-color: #3C5096; color: white;")
                            else:
                                # exceções: cinza, senão usa style_quantidade
                                if exception_map.get(row["CodigoDescricao"], False):
                                    styles.append("background-color: gray; color: black;")
                                else:
                                    styles.append(style_quantidade(row[col]))
                        else:
                            styles.append("")
                    return styles
                styled = df_display.style.apply(style_full_row, axis=1)
                

                st.dataframe(styled)
                
                # 8) Botão de download
                st.download_button(
                    label="📥 Baixar Tabela Consolidada",
                    data=to_excel(styled, "Tabela_Consolidada"),
                    file_name="tabela_consolidada.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
def display_series_historica(tab, df, colunas_datas):
    with tab:
        df_series = df.melt(id_vars=["UF", "Código", "Descrição"], value_vars=colunas_datas,
                            var_name="Data", value_name="Valor")
        df_series["Valor"] = pd.to_numeric(df_series["Valor"], errors="coerce")
        df_series["Data_clean"] = pd.to_datetime(
            df_series["Data"].str.extract(r'(\d{2}/\d{4})')[0], format="%m/%Y")
        df_series["CodigoDescricao"] = df_series["Código"].astype(str) + " - " + df_series["Descrição"].astype(str)
        df_series["Grupo"] = df_series["Código"].astype(str).str[:4]
        
        capitais_series = sorted(df["UF"].unique())
        col1, col2, col3 = st.columns(3)
        selected_capitais = col1.multiselect("Selecione a UF/BR:", capitais_series, key="series_uf")
        group_options = sorted(df_series["Grupo"].unique())
        selected_group = col2.multiselect("Selecione o grupo:", group_options, key="series_group")

        if selected_group:
            unique_items = sorted(df_series[df_series["Grupo"].isin(selected_group)]["CodigoDescricao"].unique())
        else:
            unique_items = sorted(df_series["CodigoDescricao"].unique())
        selected_items = col3.multiselect("Selecione o item:", unique_items, key="series_item")
        
        if not selected_capitais or (not selected_items and not selected_group):
            st.error("Por favor, selecione ao menos uma região e um item ou grupo para visualizar a série histórica.")
        else:
            df_series_filtered = df_series[df_series["UF"].isin(selected_capitais)]
            if selected_group:
                df_series_filtered = df_series_filtered[df_series_filtered["Grupo"].isin(selected_group)]
            elif selected_items:
                df_series_filtered = df_series_filtered[df_series_filtered["CodigoDescricao"].isin(selected_items)]
            if not df_series_filtered.empty:
                df_pivot = df_series_filtered.pivot_table(index="Data_clean",
                                                          columns=["UF", "CodigoDescricao"],
                                                          values="Valor", aggfunc="mean")
                plot_time_series(df_pivot)
            else:
                st.warning("Selecione pelo menos uma região e/ou item para visualizar a série histórica.")
        





# ----- Função Principal -----
def main():
    st.title("Leitor de Controle de Cotações - IPC")
    st.text(
        "Esta aplicação web permite consultar, analisar e visualizar dados de cotações "
        "e ponderações por subitem e região. Utilize filtros dinâmicos, gráficos interativos "
        "e a opção de download de planilhas para obter insights e sinalizar a necessidade "
        "de ampliação de amostras."
    )
    #uploaded_file = st.sidebar.file_uploader("Atualize sua Base de Cotações:", type=["xls", "xlsx"])
    #uploaded_excess_file = st.sidebar.file_uploader("Atualize sua Base de Excessões:", type=["xls", "xlsx"])
    
    create_legend()
    
    db_path = "ipc.db"
    excess_table_name = "excessoes"
    service_table_name = "servicos"
    table_name = "controle_cotacoes"
    con = duckdb.connect(db_path)
    #if uploaded_file is not None:
    #    df_novo = read_excel_file(uploaded_file)
    #    
    #    try:
    #        query = f"SELECT * FROM {table_name}"
    #        df_atual = con.execute(query).fetchdf()
    #    except Exception:
    #        df_atual = pd.DataFrame()  
    #   
    #    if not df_atual.empty:
    #        df_atualizada = atualizar_base_incremental(df_atual, df_novo)
     #   else:
     #       df_atualizada = df_novo
     #   
     #   load_database(df_atualizada, table_name, con)
        
    
    #if uploaded_excess_file is not None:
    #    df_excess_excel = read_excel_excess_file(uploaded_excess_file)
    #    df_service_excel = read_excel_service_file(uploaded_excess_file)
    #    
    #    con.register("df_excess_excel", df_excess_excel)
    #    con.execute(f"DROP TABLE IF EXISTS {excess_table_name}")
    #    con.execute(f"CREATE TABLE {excess_table_name} AS SELECT * FROM df_excess_excel")
    #    
    #    con.register("df_service_excel", df_service_excel)
    #    con.execute(f"DROP TABLE IF EXISTS {service_table_name}")
    #    con.execute(f"CREATE TABLE {service_table_name} AS SELECT * FROM df_service_excel")
    
    
    df, df_melted, df_excess, df_weight, df_service ,colunas_datas = prepare_base_data(
        con,
        table_name="controle_cotacoes",
        excess_table_name="excessoes",
        weight_table_name="ponderacoes",
        service_table_name="servicos"
    )
    df_comparativo = compute_comparative_data(df_melted)
    df_comparativo = df_comparativo.sort_values(by='Data_dt')
    
    target_date = df_comparativo["Data"].iloc[-1]
    df_tab = prepare_quantity_table(df, df_excess, df_service)
    tab1, tab2 = st.tabs(["Visão Geral", "Série Histórica"])
    display_visao_geral(
        tab1,
        df_comparativo,
        target_date,
        df_tab,
        df_weight,
        colunas_datas
    )
    display_series_historica(tab2, df, colunas_datas)

if __name__ == "__main__":
    main()
