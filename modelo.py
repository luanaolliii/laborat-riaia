import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
data = pd.read_csv('dados.csv')

# Selecionar as features e o target
X = data[['tamanho', 'ano_construcao', 'numero_quartos']]
y = data['preco']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro quadrático médio: {mse}')

# Salvar o modelo
import joblib
joblib.dump(modelo, 'modelo_previsao.pkl')
