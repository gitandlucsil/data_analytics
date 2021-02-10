import pandas as pd
import plotly.express as px

def grafico_colune_categoria(coluna, tabela):
    fig = px.histogram(tabela, x=coluna, color='Categoria')
    fig.show()

clientes_df = pd.read_csv('ClientesBanco.csv', encoding='latin1')
clientes_df = clientes_df.drop('CLIENTNUM', axis=1)
#Tratamento e visão geral dos dados
clientes_df = clientes_df.dropna()
#print(clientes_df.info())
#print(clientes_df.describe())
print(clientes_df['Categoria'].value_counts())
print(clientes_df['Categoria'].value_counts(normalize=True))
#Analisando os dados
for coluna in clientes_df:
    grafico_colune_categoria(coluna, clientes_df)
