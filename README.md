# double_adivinhar

## Descrição

Este projeto utiliza Python, Selenium, BeautifulSoup e uma rede neural (Keras) para coletar dados de resultados do jogo "Double" do site Blaze, armazenar os resultados em um arquivo JSON e tentar prever o próximo valor com base nos últimos resultados utilizando aprendizado de máquina.

## Funcionalidades

- Coleta automática dos resultados do jogo Double via web scraping.
- Armazenamento dos resultados em arquivo JSON.
- Treinamento de uma rede neural para prever o próximo valor com base nos últimos 5 resultados.
- Atualização automática do modelo conforme mais dados são coletados.
- Exibição de estatísticas de acerto durante a execução.

## Pré-requisitos

- Python 3.7 ou superior
- Google Chrome instalado
- ChromeDriver compatível com a versão do seu Chrome
- Bibliotecas Python:
  - selenium
  - beautifulsoup4
  - numpy
  - keras
  - tensorflow

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/RodrigoEmerson/double.git
   cd double
   ```
2. Instale as dependências:
   ```bash
   pip install selenium beautifulsoup4 numpy keras tensorflow
   ```
3. Baixe o [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) compatível com seu navegador e coloque-o no PATH do sistema ou no mesmo diretório do script.

## Como Executar

Execute o script principal:

```bash
python double_adivinhar.py
```

O script abrirá o navegador, coletará os resultados e tentará prever o próximo valor automaticamente.

## Exemplo de Uso

Saída esperada no terminal:

```
------------------------------------------------------------
14
Próximo valor possível: 2 (probabilidade: 0.08)
Porcentagem de acertos: 60.00%
------------------------------------------------------------
2
Próximo valor possível: 13 (probabilidade: 0.12)
Porcentagem de acertos: 62.50%
...
```

## Observações

- O script depende da estrutura da página do site Blaze. Mudanças no site podem exigir ajustes no código.
- O modelo de rede neural é simples e serve como exemplo; para melhores resultados, ajuste hiperparâmetros e arquitetura.
- O ChromeDriver deve ser compatível com a versão do seu navegador Chrome.
- Para interromper a execução, pressione `Ctrl+C` no terminal.

## Autor

Desenvolvido por [Rodrigo](https://github.com/RodrigoEmerson)

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
