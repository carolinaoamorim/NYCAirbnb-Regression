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

Após documentar a preparação dos dados, o relatório acompanha as quatro seções da APS2:

1. **Seleção e implementação dos modelos** — Dummy, Ridge e HistGradientBoosting, validação cruzada, buscas de hiperparâmetros, transformação do alvo e curvas de aprendizado.
2. **Avaliação do melhor modelo no teste** — métricas, valores reais versus preditos, resíduos versus preditos e distribuição dos resíduos.
3. **Interpretação e explicabilidade** — três coeficientes do Ridge, três importâncias por permutação do ensemble, SHAP global e três previsões individuais.
4. **Discussão crítica** — duas limitações do modelo, duas dos dados e duas ações para operacionalização e manutenção, com referências bibliográficas.

As 12 figuras do relatório são exportadas diretamente das saídas salvas do notebook por `python scripts/generate_docs_figures.py`. O script não produz visualizações adicionais. Cada figura tem análise, e as decisões de pré-processamento são vinculadas às evidências efetivamente apresentadas no código.

## Principais resultados

- O `HistGradientBoostingRegressor` ajustado obteve RMSE médio de validação cruzada de **US$ 72,73**, abaixo dos **US$ 103,12** do regressor de referência.
- No teste, o modelo final obteve **RMSE de US$ 72,77**, **MAE de US$ 44,35** e **R² de 0,50**. A avaliação exclui preços acima do percentil 99 do conjunto de treino (US$ 798,76, impresso como US$ 799), conforme o notebook.
- `room_type` foi a variável com maior importância por permutação; localização (`longitude`) e `minimum_nights` também contribuíram para o desempenho.
- Os dados representam o mercado de 2019 e a validação aleatória pode ser otimista para anúncios novos em regiões ou grupos de anfitriões não vistos.

## Tecnologias utilizadas

Python, com Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy e SHAP. A documentação usa MkDocs Material.

## Como executar

1. No diretório `NYCAirbnb-Regression`, instale as dependências: `pip install -r requirements.txt`.
2. Abra `nycairbnb.ipynb` no Jupyter ou VS Code.
3. Execute todas as células com **Restart & Run All**. O CSV deve permanecer ao lado do notebook.

Para instalar as dependências e servir a documentação localmente, siga as instruções em [Setup e Execução](docs/setup.md).
