# Regressão — Preço de anúncios do Airbnb em NYC

**Por:** Carolina Amorim e Lara Abduni

Projeto de regressão com o objetivo de prever o preço por noite (`price`, em US$) de anúncios do Airbnb em Nova York a partir de características de localização, tipo de acomodação, disponibilidade e histórico de avaliações. O notebook compara modelos de referência e regressão, valida o desempenho e discute limitações importantes para qualquer uso prático.

## Dataset

[New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (Kaggle) — 48.895 anúncios coletados em 2019 e 16 colunas. O arquivo `AB_NYC_2019.csv` já está incluído neste diretório; não é necessário baixar dados durante a execução.

## Documentação

A documentação do projeto está organizada com MkDocs Material e preparada para publicação no GitHub Pages:

[Documentação](https://carolinaoamorim.github.io/NYCAirbnb-Regression/)

Para testar localmente ou publicar atualizações, veja [Setup e Execução](docs/setup.md).

## Estrutura da análise

1. **Carregamento e inspeção inicial** — dimensões, tipos, valores ausentes, alvo e separação entre treino e teste.
2. **Preparação dos dados** — remoção de preços zerados e colunas identificadoras, tratamento de ausências e codificação das categorias em um pipeline.
3. **Seleção de modelos** — comparação entre um regressor de referência, Ridge e HistGradientBoosting, com validação cruzada de cinco partes.
4. **Avaliação e interpretação** — comparação de métricas no conjunto de teste, análise de resíduos, importância por permutação e SHAP.
5. **Discussão** — limitações do modelo e dos dados, além de recomendações para uma eventual operacionalização.

## Principais resultados

- O `HistGradientBoostingRegressor` ajustado obteve RMSE médio de validação cruzada de **US$ 72,73**, abaixo dos **US$ 103,12** do regressor de referência.
- No teste, o modelo final obteve **RMSE de US$ 72,77**, **MAE de US$ 44,35** e **R² de 0,50**. A avaliação exclui preços acima do percentil 99 do conjunto de treino (US$ 799), conforme o notebook.
- `room_type` foi a variável com maior importância por permutação; localização (`longitude`) e `minimum_nights` também contribuíram para o desempenho.
- Os dados representam o mercado de 2019 e a validação aleatória pode ser otimista para anúncios novos em regiões ou grupos de anfitriões não vistos.

## Tecnologias utilizadas

Python, com Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy e SHAP. A documentação usa MkDocs Material.

## Como executar

1. No diretório `NYCAirbnb-Regression`, instale as dependências: `pip install -r requirements.txt`.
2. Abra `nycairbnb.ipynb` no Jupyter ou VS Code.
3. Execute todas as células com **Restart & Run All**. O CSV deve permanecer ao lado do notebook.

Para instalar as dependências e servir a documentação localmente, siga as instruções em [Setup e Execução](docs/setup.md).
