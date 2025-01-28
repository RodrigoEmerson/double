import json
import time
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.estimator import keras

# Configura o driver do Chrome
driver = webdriver.Chrome()

# Abre a página desejada
url = 'https://historicosblaze.com/blaze/doubles'
driver.get(url)

# Define os valores possíveis e a sequência esperada
valores_possiveis = [14, 2, 13, 3, 12, 4, 0, 11, 5, 10, 6, 9, 7, 8, 1]
sequencia_esperada = [14, 2, 13, 3, 12, 4, 0, 11, 5, 10, 6, 9, 7, 8, 1]

# Inicializa o arquivo JSON com uma lista vazia, caso ele não exista
try:
    with open('dados.json', 'r') as f:
        dados = json.load(f)
except FileNotFoundError:
    dados = {'valores': [""],}
    with open('dados.json', 'w') as f:
        json.dump(dados, f, indent=2)

# Inicializa a rede neural
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(15, activation='softmax')) # altera para 15 unidades e softmax
model.compile(loss='categorical_crossentropy', optimizer='adam') # altera a função de perda para a categorical_crossentropy

acertos = 0
tentativas = 0
dados_treinamento = {'caracteristicas': [], 'classe': []}


while True:

    print("-"*60)

    # Espera a página carregar
    driver.implicitly_wait(10)

    # Extrai o conteúdo HTML da página
    html = driver.page_source

    # Analisa o conteúdo HTML com o BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Encontra a div desejada
    div_valor = soup.find('div', {'class': 'number'})

    # Verifica se a div foi encontrada
    if div_valor is not None:
        # Extrai o valor da div
        valor = int(div_valor.text)

        # Verifica se o valor é um dos valores possíveis
        if valor not in valores_possiveis:
            print(f'Valor {valor} não é um valor possível')
            driver.refresh()
            time.sleep(30)
            continue

        # Imprime o valor
        print(valor)

        # Adiciona o valor aos dados existentes
        dados['valores'].append(valor)

        # Converte os últimos 5 valores em um vetor de características
        ultimos_valores = dados['valores'][-5:]
        vetor_caracteristicas = np.array(ultimos_valores, dtype=np.float32).reshape(1, 5)

        # Usa a rede neural para prever as probabilidades de cada possível próximo valor
        proximo_valor_probabilidades = model.predict(vetor_caracteristicas)

        # Escolhe o valor com a maior probabilidade como próximo valor possível
        proximo_valor = valores_possiveis[np.argmax(proximo_valor_probabilidades)]

        # Converte as probabilidades em uma lista ordenada de valores possíveis
        proximo_valor_ordem = np.argsort(proximo_valor_probabilidades)[0][::-1]
        proximo_valor_possivel = [valores_possiveis[i] for i in proximo_valor_ordem]

        # Imprime o valor possível e sua probabilidade
        print(f'Próximo valor possível: {proximo_valor} (probabilidade: {proximo_valor_probabilidades[0][np.argmax(proximo_valor_probabilidades)]})')

        # Verifica se o valor está na sequência esperada
        if valor == sequencia_esperada[len(dados['valores']) % len(sequencia_esperada)]:
            acertos += 1
            dados_treinamento['caracteristicas'].append(ultimos_valores)
            dados_treinamento['classe'].append(proximo_valor)
            if acertos % 10 == 0:  # atualiza o modelo a cada 10 acertos
                X = np.array(dados_treinamento['caracteristicas'], dtype=np.float32)
                y = np.array(dados_treinamento['classe'], dtype=np.int32)
                y = keras.utils.to_categorical(y, num_classes=len(valores_possiveis))
                model.fit(X, y, epochs=10, batch_size=32)
                print('Modelo atualizado')
        else:
            print('Valor errado!')
        tentativas += 1

        # Imprime a porcentagem de acertos
        if tentativas > 0:
            porcentagem_acertos = acertos / tentativas * 100
            print(f'Porcentagem de acertos: {porcentagem_acertos:.2f}%')

        """# Imprime os valores possíveis e suas respectivas probabilidades
        for i, valor in enumerate(proximo_valor_possivel):
            print(
                f'Próximo valor possível {i + 1}: {valor} (probabilidade: {proximo_valor_probabilidades[0][proximo_valor_ordem[i]]})')
"""

        # Salva os dados atualizados em um arquivo JSON
        with open('dados.json', 'w') as f:
            json.dump(dados, f)



    else:
        print("A div não foi encontrada")

        # Atualiza a página
        driver.refresh()


    # Aguarda 30 segundos antes de executar o loop novamente
    time.sleep(30)
    # Atualiza a página
    driver.refresh()


# Fecha o driver
driver.quit()

