import pandas as pd
import smtplib
import email.message

def enviar_email(resumo_loja, loja):
    server = smtplib.SMTP('smtp.gmail.com:587')
    email_content = f'''
    <p>Caro Andre,</p>
    {resumo_loja.to_html()}
    <p>Bom dia!</p>'''

    msg = email.message.Message()
    msg['Subject'] = f'Resumo - Loja: {loja}'
    msg['From'] = 'andlucsil1221@gmail.com'
    msg['To'] = 'andre_2_silva@hotmail.com'
    password = input('Informe a senha do seu email: ')
    msg.add_header('Content-Type', 'text/html')
    msg.set_payload(email_content)

    sender = smtplib.SMTP('smtp.gmail.com:587')
    sender.starttls()
    sender.login(msg['From'], password)
    sender.sendmail(msg['From'], msg['To'], msg.as_string().encode('utf-8'))

df = pd.read_excel(r'Vendas.xlsx')
#print(df)
#Mostra o faturamento por loja
faturamento = df [['ID Loja', 'Valor Final']].groupby('ID Loja').sum()
faturamento = faturamento.sort_values(by='Valor Final', ascending=False)
#print(faturamento)
#Mostra a quantidade vendida por loja
quantidade = df [['ID Loja', 'Quantidade']].groupby('ID Loja').sum()
quantidade = quantidade.sort_values(by='Quantidade', ascending=False)
#print(quantidade)
#Calculando o ticket medios dos produtos por loja
ticket_medio = (faturamento['Valor Final']/quantidade['Quantidade']).to_frame()
ticket_medio = ticket_medio.rename(columns={0:'Ticket Medio'})
ticket_medio = ticket_medio.sort_values(by='Ticket Medio', ascending=False)
#print(ticket_medio)
#Criando o relatorio por loja
lojas = df['ID Loja'].unique()
for loja in lojas:
    tabela_loja = df.loc[df['ID Loja'] == loja, ['ID Loja', 'Quantidade', 'Valor Final']]
    resumo_loja = tabela_loja.groupby('ID Loja').sum()
    resumo_loja['Ticket Medio'] = resumo_loja['Valor Final']/resumo_loja['Quantidade']
    enviar_email(resumo_loja, loja)
#email para a diretoria
tabela_diretoria = faturamento.join(quantidade).join(ticket_medio)
enviar_email(tabela_diretoria, 'Todas as lojas')