import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_bar_chart(df_bar_series: pd.DataFrame, last_date) -> None:
    """Cria um gráfico de barras para o último mês."""
    fig, ax = plt.subplots(figsize=(8, 6))
    df_bar_series.plot(kind="bar", ax=ax)
    ax.set_xlabel("UF")
    ax.set_ylabel("Quantidade de cotações")
    ax.set_title(f"Última Data: {last_date.strftime('%m/%Y')}")
    st.pyplot(fig)

def plot_time_series(df_pivot: pd.DataFrame) -> None:
    """Cria um gráfico de série histórica para os itens selecionados com regiões demarcadas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plota as séries históricas
    for col in df_pivot.columns:
        ax.plot(df_pivot.index, df_pivot[col], marker="o", label=f"{col[0]} - {col[1]}")
    
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade de Cotações")
    ax.set_title("Série Histórica")
    
    max_val = df_pivot.max().max() * 1.1 
    
    ax.axhspan(0, 25, facecolor='#ff4d4d', alpha=0.3, label="Super Crítico (<=25)")
    ax.axhspan(25, 55, facecolor='#ffa500', alpha=0.3, label="Crítico (26-55)")
    ax.axhspan(55, 100, facecolor='#FCDA51', alpha=0.3, label="Aceitável (56-100)")
    ax.axhspan(100, max_val, facecolor='#66cc66', alpha=0.3, label="Suficiente (>100)")
    
    ax.set_ylim(0, max_val)  
    
    ax.legend(title="Região - Produto", loc="best")
    st.pyplot(fig)
