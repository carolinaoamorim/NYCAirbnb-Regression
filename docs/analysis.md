# APS2 — Regressão do preço de anúncios em NYC

**Objetivo:** prever o preço anunciado por noite (`price`, em US$). As quatro seções numeradas seguem o roteiro da APS2 para regressão. Todas as figuras abaixo são os próprios PNGs salvos em `nycairbnb.ipynb`, exportados sem redesenho; as tabelas e interpretações usam as saídas desse notebook.

## Dados e pipeline herdado da APS1

O arquivo contém **48.895 registros e 16 colunas**. A inspeção inicial executa `.describe()` apenas sobre `price`, uma variável numérica: mediana de US$ 106, percentil 90 de US$ 269, percentil 99 arredondado de US$ 799 e máximo de US$ 10.000. Essa inspeção documenta o arquivo; os ajustes das transformações e as comparações de modelos usam o treino.

O código exclui os 11 preços iguais a zero e as colunas `id`, `host_id`, `name`, `host_name` e `last_review`. Em seguida, separa 80% para treino e 20% para teste, com `random_state=42`. O percentil 99 calculado **somente no treino** é US$ **798,76** (o `print` do notebook arredonda para US$ 799). Aplicando esse mesmo limite aos dois conjuntos, restam **38.715 anúncios no treino** e **9.677 no teste**, com dez preditores.

### Evidência e decisão de pré-processamento

| Evidência efetivamente apresentada no notebook | Decisão implementada | Justificativa e alcance |
|---|---|---|
| `df.head()` mostra identificadores, nomes e a data `last_review`. | Remover `id`, `host_id`, `name`, `host_name` e `last_review`. | Identificadores não descrevem o imóvel; texto e data exigiriam engenharia específica. Contagem e frequência de avaliações permanecem, mas não substituem a informação de recência da data. |
| A contagem de ausências informa 10.052 nulos em `reviews_per_month`; o `assert` confirma a equivalência com `number_of_reviews == 0`. | Imputar `reviews_per_month` com zero. | Zero representa ausência de avaliações nesse arquivo; não é uma imputação pela média. |
| A inspeção informa 11 preços zero e grande distância entre mediana, P99 e máximo. | Excluir preços zero e limitar o alvo pelo P99 do treino. | Define um problema restrito a diárias positivas até o limite. Valores altos não foram demonstrados como erros; o corte reduz sua influência no RMSE e exclui parte do mercado da avaliação. |
| A inspeção de ausências não aponta nulos nas outras seis variáveis numéricas usadas. | Manter imputador pela mediana como proteção e aplicar `StandardScaler`. | A imputação é preventiva; a escala torna a penalização L2 do Ridge comparável entre variáveis numéricas. Não é uma conclusão de um mapa de correlação. |
| `neighbourhood_group`, `room_type` e `neighbourhood` são categorias nominais, sem ordem numérica. | Aplicar OHE e agrupar categorias com menos de 50 registros no ajuste. | Evita impor uma ordem artificial. O limiar 50 é uma escolha metodológica para controlar a dimensão, sem alegar que foi otimizado por uma figura. |
| O histograma duplo da [seção 1.2](#12-mudanca-no-pipeline-transformacao-do-alvo) mostra a cauda do preço no treino. | Testar `log1p(price)` e refazer as buscas. | A transformação é uma hipótese de melhoria; sua adoção depende do RMSE na mesma validação. |

`ColumnTransformer` e `Pipeline` ajustam imputadores, escalas e categorias dentro de cada fold. O encoder usa `min_frequency=50` nas três colunas categóricas; o agrupamento afeta principalmente bairros. No ajuste sobre todo o treino, são produzidas **104 features**. As três colunas originais são multiclasse; não há OHE de uma variável originalmente binária nem PCA após OHE. Manter as categorias no Ridge preserva a simetria da penalização, mas exige cautela na interpretação dos coeficientes, conforme a [documentação do encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

O corte do alvo é definido antes da validação cruzada, usando o treino completo: a CV descreve esse recorte previamente filtrado. O teste não entra nas buscas; a aplicação do limite ao seu alvo restringe a população avaliada. As métricas não estimam o desempenho em todos os preços originais.

## 1. Seleção e implementação dos modelos

São comparados `DummyRegressor(strategy="mean")`, **Ridge com regularização L2** e **HistGradientBoostingRegressor**, ensemble de árvores por boosting. O Dummy estabelece a referência constante; o Ridge verifica uma relação linear regularizada; o ensemble permite interações e relações não lineares. O mesmo pré-processamento é mantido para comparar os modelos.

A métrica principal é **RMSE, em US$**: penaliza mais os erros grandes e mantém a unidade do alvo. MAE e R² complementam a leitura, sem determinar a escolha. Todas as comparações usam `KFold(n_splits=5, shuffle=True, random_state=42)` no treino. O scorer negativo do Scikit-learn é convertido em RMSE positivo nas tabelas.

| Modelo inicial | RMSE médio, US$ | Desvio entre folds, US$ | MAE médio, US$ | R² médio |
|---|---:|---:|---:|---:|
| Dummy (média) | 103,12 | 2,43 | 73,18 | −0,00 |
| Ridge (`alpha=1`) | 79,66 | 2,15 | 49,83 | 0,40 |
| HistGradientBoosting padrão | 72,88 | 1,85 | 44,44 | 0,50 |

O ensemble tem o menor RMSE inicial. A vantagem sobre o Ridge é compatível com ganhos ao modelar interações e não linearidades; a comparação, isoladamente, não identifica qual interação produz a diferença.

### 1.1 Seleção de hiperparâmetros

O **Ridge** usa `GridSearchCV` com 13 valores de `alpha`, de `10⁻³` a `10³`, em escala logarítmica. O **HistGradientBoosting** usa `RandomizedSearchCV` com 15 candidatos e semente 42: cinco hiperparâmetros tornam uma grade completa mais custosa. As duas buscas usam o mesmo RMSE e os mesmos cinco folds.

<figure markdown>
  ![RMSE médio de validação cruzada do Ridge para cada alpha testado.](assets/images/ridge_alpha.png)
  <figcaption>Figura 1 — Curva de validação do Ridge, produzida pela célula da busca em grade. Menor RMSE é melhor.</figcaption>
</figure>

O mínimo ocorre em **`alpha=1`**, com RMSE de **US$ 79,66**, igual ao padrão. A curva mostra piora com penalizações mais fortes na região testada. Ela sustenta a escolha de `alpha`; não demonstra que outras variáveis seriam dispensáveis nem que a regularização é inútil.

| Hiperparâmetro do ensemble | Espaço da busca | Melhor valor impresso |
|---|---|---:|
| `learning_rate` | Loguniforme entre 0,02 e 0,3 | 0,04843 |
| `max_iter` | Inteiros de 150 a 599 | 364 |
| `max_leaf_nodes` | Inteiros de 15 a 79 | 76 |
| `min_samples_leaf` | Inteiros de 10 a 99 | 49 |
| `l2_regularization` | Loguniforme entre 0,001 e 10 | 0,8111 |

O ensemble ajustado obtém **RMSE de US$ 72,73**, ante US$ 72,88 do padrão. O ganho de US$ 0,15 é pequeno em relação à variação entre folds e não demonstra superioridade estatística. É o menor erro observado nessa busca, que não esgota o espaço de hiperparâmetros.

### 1.2 Mudança no pipeline: transformação do alvo

<figure markdown>
  ![Histogramas do preço de treino e de log de um mais o preço, lado a lado.](assets/images/price_log_transform.png)
  <figcaption>Figura 2 — Mesmo alvo de treino antes e depois de `log1p`, após o recorte de preço. A transformação comprime a cauda à direita.</figcaption>
</figure>

O painel original concentra anúncios nos preços menores e apresenta cauda à direita; o transformado fica menos assimétrico. Isso motiva **testar**, sem garantir melhora, o `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`. O ajuste ocorre em log e as previsões retornam a dólares antes do cálculo das métricas.

As buscas são refeitas para o ensemble e para o Ridge, com o mesmo espaço, orçamento, folds e RMSE. No ensemble com log, os melhores valores impressos são `learning_rate=0,1513`, `max_iter=516`, `max_leaf_nodes=67`, `min_samples_leaf=33` e `l2_regularization=2,389`. No Ridge com log, `alpha=0,001`.

| Modelo | RMSE médio, US$ | Desvio entre folds, US$ | MAE médio, US$ | R² médio |
|---|---:|---:|---:|---:|
| Dummy (média) | 103,12 | 2,43 | 73,18 | −0,00 |
| Ridge padrão | 79,66 | 2,15 | 49,83 | 0,40 |
| HistGradientBoosting padrão | 72,88 | 1,85 | 44,44 | 0,50 |
| Ridge ajustado | 79,66 | 2,15 | 49,83 | 0,40 |
| HistGradientBoosting ajustado | **72,73** | 1,80 | 44,26 | 0,50 |
| Ridge ajustado com log | 80,54 | 2,39 | 46,20 | 0,39 |
| HistGradientBoosting ajustado com log | 73,70 | 1,99 | **42,14** | 0,49 |

<figure markdown>
  ![RMSE médio e desvio entre folds dos sete modelos e variantes avaliados.](assets/images/model_comparison.png)
  <figcaption>Figura 3 — Comparação final da validação cruzada. As barras de erro mostram um desvio padrão entre folds, não intervalos de confiança.</figcaption>
</figure>

Para o ensemble, o log reduz o MAE de US$ 44,26 para US$ 42,14, mas eleva o RMSE de US$ 72,73 para US$ 73,70. No Ridge, o comportamento é semelhante. O erro quadrático em log é um objetivo diferente do erro quadrático em dólares; inverter a média prevista em log não recupera, em geral, a média condicional do preço. Os resultados são compatíveis com essa diferença, mas não isolam sua causa.

**Decisão:** manter o **HistGradientBoosting ajustado sem log**, com redução aproximada de **29,5% no RMSE de CV** em relação ao Dummy. O log foi implementado, retunado e rejeitado pelo critério definido. O escore da busca é uma estimativa usada na seleção; a avaliação independente vem na seção 2.

### 1.3 Curvas de aprendizado do melhor modelo

<figure markdown>
  ![Curvas de RMSE de treino e validação do ensemble selecionado em seis tamanhos de treino.](assets/images/learning_curves.png)
  <figcaption>Figura 4 — `learning_curve` do HistGradientBoosting ajustado, com os mesmos cinco folds e seis tamanhos. As faixas representam um desvio padrão entre folds.</figcaption>
</figure>

Com **3.097 exemplos**, os RMSE são **US$ 44,41 no treino** e **US$ 79,61 na validação**, mostrando grande diferença de generalização. Com **30.972 exemplos por fold**, chegam a **US$ 64,77** e **US$ 72,74**. A validação ainda melhora de US$ 73,85 para US$ 72,74 entre 19.822 e 30.972 exemplos; os ganhos observados diminuem.

Há um cuidado específico: o código mantém `early_stopping="auto"`, que passa a ativar a parada antecipada acima de 10.000 exemplos. Isso muda o regime de ajuste entre os tamanhos de 8.672 e 14.247 e pode contribuir para o salto do erro de treino. Portanto, a figura não isola apenas o efeito do volume de dados e não permite concluir ausência de sobreajuste, predomínio de viés ou inutilidade de coletar mais dados. Essa regra consta na [documentação do HistGradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

## 2. Avaliação do melhor modelo junto ao conjunto de teste

O modelo escolhido pela CV, já reajustado no treino completo pela busca, prevê os **9.677 anúncios** do teste filtrado. Essas métricas são calculadas após a seleção; as explicações posteriores não são usadas para trocar o modelo.

| Métrica | HistGradientBoosting ajustado | Dummy de referência |
|---|---:|---:|
| RMSE, US$ | **72,77** | 102,50 |
| MAE, US$ | 44,35 | — |
| R² | 0,50 | — |

O RMSE fica próximo dos US$ 72,73 de CV e cai cerca de **29,0%** frente ao Dummy no teste. Essa proximidade é compatível com a estimativa de validação neste recorte, sem provar ausência de sobreajuste. R² de 0,50 indica redução de aproximadamente metade da soma dos erros quadráticos em relação à referência pela média do próprio teste.

<figure markdown>
  ![Três painéis: valores reais versus preditos, resíduos versus preditos e histograma dos resíduos no teste.](assets/images/test_diagnostics.png)
  <figcaption>Figura 5 — Os três diagnósticos exigidos para regressão. Resíduo = preço real − preço predito; valores positivos indicam subestimação.</figcaption>
</figure>

- **Real versus predito:** há concentração próxima à diagonal nos preços baixos, mas muitos anúncios caros ficam abaixo dela. O modelo tende a subestimar a parte mais cara mesmo dentro do recorte avaliado.
- **Resíduos versus preditos:** o espalhamento muda ao longo das previsões e há grandes resíduos positivos, sobretudo na faixa intermediária de valores previstos. O padrão não é uma nuvem uniforme em torno de zero; não se deve descrever um funil monotônico em toda a extensão do eixo. As faixas diagonais são compatíveis com preços reais repetidos, pois, para um preço real fixo, o resíduo diminui quando a previsão aumenta.
- **Distribuição dos resíduos:** média de **US$ 0,46**, mediana de **−US$ 10,06** e assimetria de **2,53**. Uma média próxima de zero convive com erros positivos grandes; a mediana negativa indica superestimação no centro da distribuição. **9,9%** dos anúncios têm erro absoluto superior a US$ 100.

A tabela produzida na célula seguinte detalha o erro por preço real:

| Faixa de preço real, US$ | Anúncios | MAE, US$ | Resíduo médio, US$ |
|---|---:|---:|---:|
| (0; 75] | 3.082 | 21,9 | −19,7 |
| (75; 150] | 3.692 | 35,7 | −21,5 |
| (150; 250] | 1.940 | 45,8 | 5,9 |
| (250; 400] | 712 | 101,7 | 90,2 |
| (400; 798,76] | 251 | 273,4 | 273,1 |

Até US$ 150 predomina superestimação média; acima de US$ 400 a subestimação média chega a US$ 273,1. O erro agregado esconde diferenças relevantes entre segmentos. A tabela é diagnóstica, pois a faixa de preço real só é conhecida após observar o alvo.

## 3. Interpretação e explicabilidade

### 3.1 Modelo linear: três maiores coeficientes em magnitude

<figure markdown>
  ![Quinze coeficientes do Ridge ajustado de maior magnitude, com cores indicando o sinal.](assets/images/ridge_coefficients.png)
  <figcaption>Figura 6 — Coeficientes do Ridge na escala do alvo em dólares. Preditores numéricos estão padronizados; categorias são indicadores OHE. Azul indica sinal positivo e vermelho, negativo.</figcaption>
</figure>

| Feature | Coeficiente, US$ | Análise |
|---|---:|---|
| `neighbourhood_Tribeca` | +103,0 | Maior componente positivo entre os indicadores; contribui para previsões maiores quando esse indicador está ativo. |
| `neighbourhood_NoHo` | +82,4 | Segundo maior componente positivo; mostra que o ajuste usa informação de bairro além do distrito e das coordenadas. |
| `room_type_Entire home/apt` | +67,8 | Indicador de imóvel inteiro tem contribuição positiva. No mesmo bloco, `room_type_Shared room` tem coeficiente −47,1. |

Como o OHE mantém todas as categorias (`drop=None`), **não existe uma categoria de referência omitida**. Os valores são componentes da parametrização regularizada, não aumentos isolados em relação a um imóvel de referência. Por exemplo, o contraste entre os componentes de imóvel inteiro e quarto compartilhado é aproximadamente US$ 114,9, mantendo os demais termos fixos; não constitui efeito causal.

Para uma variável numérica, o coeficiente representa o termo associado a um desvio padrão da feature. Bairro, distrito e coordenadas compartilham informação, e não é possível trocar livremente um bairro mantendo todas as características geográficas coerentes. Logo, não cabe afirmar que mudar para Tribeca causaria aumento de US$ 103.

### 3.2 Ensemble: três features de maior importância

O HistGradientBoosting não oferece `feature_importances_`. O código usa **importância por permutação** em uma amostra reprodutível de **3.000 anúncios do teste**, com cinco repetições e o mesmo scorer de RMSE, permutando uma coluna original por vez.

<figure markdown>
  ![Aumento médio do RMSE ao permutar cada uma das dez features originais, com desvio entre repetições.](assets/images/permutation_importance.png)
  <figcaption>Figura 7 — Importância por permutação do ensemble selecionado. Maior aumento de RMSE indica maior dependência preditiva nessa amostra.</figcaption>
</figure>

| Feature | Aumento médio do RMSE, US$ | Desvio, US$ | Análise |
|---|---:|---:|---|
| `room_type` | 30,36 | 0,83 | Maior perda ao embaralhar: o modelo depende fortemente do tipo de acomodação. |
| `longitude` | 11,33 | 0,47 | A posição espacial contém informação importante mesmo com bairro, distrito e latitude disponíveis. |
| `minimum_nights` | 11,26 | 0,53 | A regra de estadia mínima também contribui para prever o preço; a permutação não informa a direção nem demonstra um limiar específico. |

Longitude e noites mínimas têm importâncias próximas; a diferença de US$ 0,07 não sustenta uma ordem robusta entre elas. `availability_365` vem a seguir (+US$ 9,59). Informação geográfica compartilhada ajuda a explicar por que bairro e distrito podem ter importância individual menor. Os desvios são entre permutações, não intervalos de confiança populacionais.

### 3.3 SHAP: importância global

O notebook aplica `TreeExplainer` ao ensemble sobre os dados transformados de **1.500 anúncios**, retirados da amostra anterior. Como o vencedor usa o alvo original, as contribuições estão em **US$**. O valor-base é cerca de **US$ 137,5**, definido pelo explicador, e não deve ser confundido com a média das 1.500 previsões.

<figure markdown>
  ![Ranking global de média do valor absoluto SHAP por feature transformada.](assets/images/shap_global.png)
  <figcaption>Figura 8 — Barras de média de |SHAP|. Mostram magnitude média de contribuição, sem indicar sua direção.</figcaption>
</figure>

As três maiores magnitudes são **`room_type_Entire home/apt` (US$ 51,15)**, **`longitude` (US$ 20,52)** e **`availability_365` (US$ 15,88)**. Latitude (US$ 12,62) e noites mínimas (US$ 9,66) também se destacam. O valor absoluto não informa se uma feature encarece ou barateia cada anúncio. O ranking usa colunas transformadas, enquanto a permutação usa as colunas originais; além disso, as duas medidas respondem a perguntas diferentes.

<figure markdown>
  ![Beeswarm SHAP com direção das contribuições e cores para valores altos e baixos das features transformadas.](assets/images/shap_global_detail.png)
  <figcaption>Figura 9 — Cada ponto é um anúncio. A posição horizontal indica contribuição em US$; vermelho representa valor alto da feature e azul, baixo. Nas numéricas, a cor usa valores padronizados.</figcaption>
</figure>

O indicador de imóvel inteiro ativo aparece principalmente com contribuições positivas. Longitudes mais altas tendem a contribuições negativas; disponibilidade alta tende a contribuições positivas, e noites mínimas altas aparecem entre contribuições negativas. Há dispersão e exceções: são padrões do modelo, sem demonstrar causalidade nem confirmar que disponibilidade alta identifica um anfitrião profissional.

### 3.4 SHAP: três predições individuais

As escolhas seguem o **menor, mediano e maior preço previsto dentro da amostra SHAP**, não extremos de toda a base. Os gráficos de cascata decompõem a previsão: valor-base + todas as contribuições SHAP = previsão, salvo arredondamento. Os números abaixo vêm do Top-5 impresso na célula correspondente.

<figure markdown>
  ![Cascata SHAP do quarto compartilhado em Williamsburg com previsão de aproximadamente 26 dólares.](assets/images/shap_examples_low.png)
  <figcaption>Figura 10 — Menor previsão da amostra: registro 47212, quarto compartilhado em Williamsburg, Brooklyn. Real: US$ 35; previsto: aproximadamente US$ 26.</figcaption>
</figure>

Partindo de US$ 137,5, a ausência do indicador de imóvel inteiro contribui **−US$ 69,2**, `minimum_nights=30` contribui **−US$ 24,3** e longitude, **−US$ 12,0**. O indicador de quarto privado contribui −US$ 9,9, enquanto `availability_365=341` acrescenta US$ 9,3. As demais features completam a soma até a previsão. O modelo subestima em cerca de US$ 9; a interpretação é específica desse anúncio.

<figure markdown>
  ![Cascata SHAP do apartamento inteiro no Harlem com previsão de aproximadamente 123 dólares.](assets/images/shap_examples_median.png)
  <figcaption>Figura 11 — Previsão mediana da amostra: registro 31558, imóvel inteiro no Harlem, Manhattan. Real: US$ 100; previsto: aproximadamente US$ 123.</figcaption>
</figure>

O indicador de imóvel inteiro acrescenta **US$ 40,5**, mas disponibilidade zero (**−US$ 23,7**), longitude (**−US$ 22,1**) e latitude (**−US$ 10,9**) compensam esse movimento. Noites mínimas contribuem +US$ 6,7. A superestimação é de cerca de US$ 23, equivalente a 23% do preço observado; sua aceitabilidade depende do uso, que não foi definido no projeto.

<figure markdown>
  ![Cascata SHAP do quarto privado em Midtown com previsão de aproximadamente 516 dólares.](assets/images/shap_examples_high.png)
  <figcaption>Figura 12 — Maior previsão da amostra: registro 19136, quarto privado em Midtown, Manhattan. Real: US$ 499; previsto: aproximadamente US$ 516.</figcaption>
</figure>

O indicador de Midtown soma **US$ 131,7**, a disponibilidade de 365 dias, **US$ 87,4**, e a contagem de 12 anúncios do anfitrião, **US$ 69,8**. Longitude acrescenta US$ 55,2, enquanto a ausência de imóvel inteiro contribui −US$ 31,4. O modelo superestima em cerca de US$ 17. A combinação supera localmente o efeito negativo do tipo de quarto, mas não comprova que o anúncio seja hotel ou aparthotel.

## 4. Discussão crítica

### 4.1 Duas limitações do melhor modelo

1. **Ausência de intervalos de previsão.** Os diagnósticos mostram erros diferentes por faixa, mas a saída do ensemble é um único preço. Isso dificulta comunicar o risco de cada sugestão. Uma extensão seria calibrar intervalos conformais com dados separados: [Lei et al. (2018)](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) apresentam cobertura marginal sob hipóteses como permutabilidade; ela não é garantia para cada faixa nem para dados com mudança de distribuição. Essa extensão não foi implementada.
2. **Generalização para grupos novos não medida.** A divisão aleatória não impede anúncios do mesmo anfitrião ou de regiões próximas em treino e validação. O desempenho para anfitriões ou regiões inéditos pode diferir do teste atual. [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881) discutem validação com estrutura espacial e hierárquica. Uma avaliação complementar usaria grupos de anfitrião, preservando `host_id` apenas para dividir, ou blocos espaciais; o notebook não executa essa comparação.

### 4.2 Duas limitações dos dados

1. **Preço anunciado sem transações.** O CSV não contém reservas e valores efetivamente pagos. Assim, não permite validar receita, demanda ou sucesso comercial de uma recomendação. A literatura de precificação do Airbnb, como [Gibbs et al. (2018)](https://doi.org/10.1080/10548408.2017.1308292), analisa associações entre atributos e preços anunciados; aplicá-las a vendas é uma extrapolação. Essa distinção decorre também do esquema do arquivo e não é corrigida pelo aumento de complexidade do modelo.
2. **Uma fotografia de 2019, sem acompanhamento temporal.** O arquivo não acompanha as mesmas ofertas ao longo das datas, impedindo medir sazonalidade ou estabilidade posterior das relações. `availability_365=0` também não distingue, por si só, reserva de bloqueio do calendário. A mudança da relação entre entradas e alvo ao longo do tempo é discutida por [Gama et al. (2014)](https://doi.org/10.1145/2523813); o risco aqui é uma hipótese de aplicação, pois o arquivo não permite medir drift temporal.

### 4.3 Duas ações para operacionalização e manutenção

1. **Versionar e servir o pipeline completo com validação de entradas.** Salvar conjuntamente transformações e modelo, registrar versões, esquema e recorte de preço; validar tipos, coordenadas e noites mínimas e registrar categorias desconhecidas. Testar o artefato antes de disponibilizá-lo e manter uma versão anterior para reversão. A proposta se apoia nos testes e monitoramento de [Breck et al. (2017)](https://research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf) e nas dependências de sistemas de ML discutidas por [Sculley et al. (2015)](https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). Não há API ou artefato de produção implementado no notebook.
2. **Monitorar entradas e erros e condicionar o retreino a evidências.** Acompanhar categorias novas, ausências, distribuição das features e RMSE/MAE por segmento quando houver novos preços anunciados observados, mantendo a definição do alvo. Comparar um candidato com o modelo vigente em dados recentes e separados antes de substituí-lo. Se o objetivo passar a preço transacionado, será necessário redefinir alvo, base e validação. A manutenção responde ao problema de drift descrito por [Gama et al. (2014)](https://doi.org/10.1145/2523813); os alertas e a rotina são propostas, não resultados executados.

As referências bibliográficas completas estão em [Referências](references.md).
