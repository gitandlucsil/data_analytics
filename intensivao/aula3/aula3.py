import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
#Importando a base de dados
meses = {'jan': 1,
        'fev': 2,
        'mar': 3,
        'abr': 4,
        'mai': 5,
        'jun': 6,
        'jul': 7,
        'ago': 8,
        'set': 9,
        'out': 10,
        'nov': 11,
        'dez': 12}
caminho_bases = pathlib.Path('dataset')
base_airbnb = pd.DataFrame()
for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    mes = meses[nome_mes]
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))
    df = pd.read_csv(caminho_bases/arquivo.name, low_memory=False)
    df['ano'] = ano
    df['mes'] = mes
    base_airbnb = base_airbnb.append(df)
#print(base_airbnb)
#Tratamentos
#print(list(base_airbnb.columns))
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')
colunas = ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count',
'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'guests_included', 
'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating', 
'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
'review_scores_location', 'review_scores_value', 'instant_bookable', 'is_business_travel_ready',
'cancellation_policy', 'ano', 'mes']
base_airbnb = base_airbnb.loc[:, colunas]
#print(list(base_airbnb.columns))
#print(base_airbnb)
#Tratar os valores faltando
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() > 70000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
#print(base_airbnb.isnull().sum())
base_airbnb = base_airbnb.dropna()
#print(base_airbnb.shape)
#print(base_airbnb.isnull().sum())
#Verificar os tipos de dados em cada coluna
#print(base_airbnb.dtypes)
#print('-'*60)
#print(base_airbnb.iloc[0])
#price
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)
#extra people
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)
#verificando os tipos
#print(base_airbnb.dtypes)
#Analise exploratoria e tratar outliers
plt.figure(figsize=(15, 10))
sns.heatmap(base_airbnb.corr(), annot=True, cmap='Greens')
#print(base_airbnb.corr())
#Definição de funções para analise de outliers
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude

def excluir_outliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    lim_inf, lim_sup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas

def diagrama_caixa(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)

def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.distplot(coluna, hist=True)

def grafico_barra(coluna):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))
#price
#diagrama_caixa(base_airbnb['price'])
#histograma(base_airbnb['price'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'price')
print('{} linhas removidas'.format(linhas_removidas))
#histograma(base_airbnb['price'])
print(base_airbnb.shape)
#extra_people
#diagrama_caixa(base_airbnb['extra_people'])
#histograma(base_airbnb['extra_people'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'extra_people')
print('{} linhas removidas'.format(linhas_removidas))
#histograma(base_airbnb['extra_people'])
print(base_airbnb.shape)
#host_listings_count
#diagrama_caixa(base_airbnb['host_listings_count'])
#grafico_barra(base_airbnb['host_listings_count'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'host_listings_count')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#accommodates
#diagrama_caixa(base_airbnb['accommodates'])
#grafico_barra(base_airbnb['accommodates'])
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'accommodates')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#bathrooms
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bathrooms')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#bedrooms
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'bedrooms')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#beds
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'beds')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#guests_included
base_airbnb = base_airbnb.drop('guests_included', axis = 1)
print(base_airbnb.shape)
#minimum_nights
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'minimum_nights')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#maximum_nights
base_airbnb = base_airbnb.drop('maximum_nights', axis = 1)
print(base_airbnb.shape)
#number_of_reviews
base_airbnb = base_airbnb.drop('number_of_reviews', axis = 1)
print(base_airbnb.shape)
#Tratamento de colunas de valores de texto
#print(base_airbnb['property_type'].value_counts())
tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []
for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 500:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type']==tipo, 'property_type'] = 'Outros'
print(base_airbnb['property_type'].value_counts())
#room_type
print(base_airbnb['room_type'].value_counts())
#bed_type
print(base_airbnb['bed_type'].value_counts())
# agrupando categorias de bed_type
tabela_bed = base_airbnb['bed_type'].value_counts()
colunas_agrupar = []
for tipo in tabela_bed.index:
    if tabela_bed[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type']==tipo, 'bed_type'] = 'Outros'
print(base_airbnb['bed_type'].value_counts())
#cancellation_policy
print(base_airbnb['cancellation_policy'].value_counts())
# agrupando categorias de cancellation_policy
tabela_cancellation = base_airbnb['cancellation_policy'].value_counts()
colunas_agrupar = []
for tipo in tabela_cancellation.index:
    if tabela_cancellation[tipo] < 10000:
        colunas_agrupar.append(tipo)
print(colunas_agrupar)
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy']==tipo, 'cancellation_policy'] = 'Outros'
print(base_airbnb['cancellation_policy'].value_counts())
#amenities
print(base_airbnb['amenities'].iloc[1].split(','))
print(len(base_airbnb['amenities'].iloc[1].split(',')))
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb = base_airbnb.drop('amenities', axis = 1)
print(base_airbnb.shape)
base_airbnb, linhas_removidas = excluir_outliers(base_airbnb, 'n_amenities')
print('{} linhas removidas'.format(linhas_removidas))
print(base_airbnb.shape)
#Visualizacao de Mapa de Propriedades
amostra = base_airbnb.sample(n=5000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.latitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                            center=centro_mapa, zoom=10,
                            mapbox_style='stamen-terrain')
#mapa.show()
#Enconding
colunas_tf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_tf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='t', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna]=='f', coluna] = 0
colunas_categorias = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_categorias)
print(base_airbnb_cod.head())
#Modelo de Previsão
def avaliar_modelo(nome_modelo, y_test, previsao):
    r2 = r2_score(y_test, previsao)
    RSME = np.sqrt(mean_squared_error(y_test, previsao))
    return f'Modelo {nome_modelo}:\nR2:{r2:.2%}\nRSME:{RSME:.2f}'
    
modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()
modelos = {
    'RandomForest': modelo_rf,
    'LinearRegression': modelo_lr,
    'ExtraTreesRegressor': modelo_et
}
y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=10)
for nome_modelo, modelo in modelos.items():
    #treinar
    modelo.fit(X_train, y_train)
    #testar
    previsao = modelo.predict(X_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))