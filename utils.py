import pandas as pd
from io import BytesIO
import re
import datetime

def to_excel(df: pd.DataFrame, sheet_name: str) -> bytes:
    """Converte um DataFrame para um arquivo Excel em memória."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def highlight_values(cell_value):
    """Retorna o estilo de formatação de célula com base no valor."""
    try:
        valor = float(cell_value)
        if valor <= 25:
            return "background-color: #ff4d4d; color: black" 
        elif valor <= 55:
            return "background-color: #ffa500; color: black" 
        elif valor <= 100:
            return "background-color: #FCDA51; color: black"  
        else:
            return "background-color: #66cc66; color: black" 
    except:
        return ""

def get_criticidade(valor):
    """Retorna a criticidade de um valor numérico."""
    try:
        valor = float(valor)
        if valor <= 25:
            return "SuperCrítico"
        elif valor <= 55:
            return "Crítico"
        elif valor <= 100:
            return "Aceitável"
        else:
            return "Suficiente"
    except:
        return None
