import json
import time
import numpy as np
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from bs4 import BeautifulSoup
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def carregar_dados_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        dados = {'valores': [""]}
        with open(filepath, 'w') as f:
            json.dump(dados, f, indent=2)
        return dados

def salvar_dados_json(filepath, dados):
    with open(filepath, 'w') as f:
        json.dump(dados, f)

def criar_modelo(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def extrair_valor_atual(soup):
    div_valor = soup.find('div', {'class': 'number'})
    if div_valor is not None:
        try:
            return int(div_valor.text)
        except ValueError:
            return None
    return None

def main():
    # Parâmetros configuráveis
    url = 'https://historicosblaze.com/blaze/doubles'
    json_path = 'dados.json'
    valores_possiveis = [14, 2, 13, 3, 12, 4, 0, 11, 5, 10, 6, 9, 7, 8, 1]
    sequencia_esperada = valores_possiveis
    tempo_espera = 30

    driver = webdriver.Chrome()
    driver.get(url)
    dados = carregar_dados_json(json_path)
    model = criar_modelo(input_dim=5, num_classes=len(valores_possiveis))
    acertos = tentativas = 0
    dados_treinamento = {'caracteristicas': [], 'classe': []}

    try:
        while True:
            print("-"*60)
            driver.implicitly_wait(10)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            valor = extrair_valor_atual(soup)

            if valor is None or valor not in valores_possiveis:
                print(f'Valor inválido: {valor}')
                driver.refresh()
                time.sleep(tempo_espera)
                continue

            print(valor)
            dados['valores'].append(valor)
            if len(dados['valores']) < 6:
                salvar_dados_json(json_path, dados)
                driver.refresh()
                time.sleep(tempo_espera)
                continue

            ultimos_valores = dados['valores'][-6:-1]
            vetor_caracteristicas = np.array(ultimos_valores, dtype=np.float32).reshape(1, 5)
            proximo_valor_probabilidades = model.predict(vetor_caracteristicas)
            proximo_valor = valores_possiveis[np.argmax(proximo_valor_probabilidades)]
            print(f'Próximo valor possível: {proximo_valor} (probabilidade: {proximo_valor_probabilidades[0][np.argmax(proximo_valor_probabilidades)]})')

            if valor == sequencia_esperada[len(dados['valores']) % len(sequencia_esperada)]:
                acertos += 1
                dados_treinamento['caracteristicas'].append(ultimos_valores)
                dados_treinamento['classe'].append(proximo_valor)
                if acertos % 10 == 0:
                    X = np.array(dados_treinamento['caracteristicas'], dtype=np.float32)
                    y = np.array(dados_treinamento['classe'], dtype=np.int32)
                    y = to_categorical(y, num_classes=len(valores_possiveis))
                    model.fit(X, y, epochs=10, batch_size=32)
                    print('Modelo atualizado')
            else:
                print('Valor errado!')
            tentativas += 1

            if tentativas > 0:
                porcentagem_acertos = acertos / tentativas * 100
                print(f'Porcentagem de acertos: {porcentagem_acertos:.2f}%')

            salvar_dados_json(json_path, dados)
            time.sleep(tempo_espera)
            driver.refresh()
    except KeyboardInterrupt:
        print("Execução interrompida pelo usuário.")
    except WebDriverException as e:
        print(f"Erro no WebDriver: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
        
        
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
"""