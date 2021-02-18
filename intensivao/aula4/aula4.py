from selenium import webdriver
import pandas as pd
import time

clientes_df = pd.read_excel('Clientes Pagamento.xlsx', dtype={'Cliente':object})
#print(clientes_df)
driver = webdriver.Chrome()