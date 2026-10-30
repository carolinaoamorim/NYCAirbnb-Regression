<section class="hero" markdown>
<p class="eyebrow">APS Machine Learning • Documentação aberta</p>

# Regressão Airbnb NYC

Análise e modelagem do preço por noite de anúncios do Airbnb em Nova York, com avaliação reprodutível, interpretação dos modelos e discussão crítica.

<div class="hero-actions" markdown>
[Ver análise](analysis.md){ .md-button .md-button--primary }
[Conhecer o dataset](dataset.md){ .md-button }
[Executar localmente](setup.md){ .md-button }
</div>
</section>

## Visão Geral

Este projeto usa o **New York City Airbnb Open Data** para estimar o preço por noite (`price`, em dólares) a partir de dados de localização, tipo de acomodação, disponibilidade e avaliações. A etapa de modelagem compara um regressor de referência, Ridge e HistGradientBoosting.

<div class="metric-grid" markdown>
<div class="metric-card" markdown>
<span class="metric-label">Anúncios</span>
<strong>48.895</strong>
Registros no arquivo original de 2019.
</div>

<div class="metric-card" markdown>
<span class="metric-label">Colunas</span>
<strong>16</strong>
Atributos de anúncio, anfitrião e preço.
</div>

<div class="metric-card" markdown>
<span class="metric-label">RMSE no teste</span>
<strong>US$ 72,67</strong>
HistGradientBoosting ajustado, após o corte de valores extremos descrito na análise.
</div>

<div class="metric-card" markdown>
<span class="metric-label">R² no teste</span>
<strong>0,50</strong>
Desempenho no conjunto de teste mantido fora dos ajustes.
</div>
</div>

## Resumo Executivo

| Pergunta | Resposta da análise |
|---|---|
| Qual é o problema? | Regressão do preço por noite (`price`), em dólares. |
| Qual é a referência? | `DummyRegressor` que prevê a média do preço no treino. |
| Qual modelo teve melhor RMSE? | `HistGradientBoostingRegressor` ajustado, com RMSE médio de US$ 72,47 na validação cruzada. |
| Como foi o resultado no teste? | RMSE de US$ 72,67, MAE de US$ 44,31 e R² de 0,50 (29% abaixo do Dummy). |
| O que exige cuidado? | O erro cresce muito nos anúncios caros; as limitações do modelo e dos dados estão na [seção 4](analysis.md#4-discussao-critica). |

## Organização conforme o roteiro da APS2

1. **Seleção e implementação:** Dummy, Ridge e ensemble, buscas de hiperparâmetros, mudança no alvo e curvas de aprendizado.
2. **Avaliação no teste:** métricas e os três diagnósticos exigidos para regressão.
3. **Interpretação e explicabilidade:** três coeficientes do Ridge, três importâncias do ensemble, SHAP global e três casos individuais.
4. **Discussão crítica:** duas limitações do modelo, duas dos dados e duas ações de operacionalização e manutenção, com referências.

O [relatório](analysis.md) segue a mesma numeração de seções e de figuras (1 a 14) do notebook `nycairbnb.ipynb` e usa exclusivamente as figuras salvas nele, com análise para cada uma e vínculo explícito entre figura e decisão de pré-processamento.

## Principais Achados

<div class="highlight-grid" markdown>
<div class="highlight-card" markdown>
<span class="status-pill">Achado 1</span>

### O ensemble supera a referência

O HistGradientBoosting ajustado reduziu o RMSE de validação cruzada de US$ 103,12 (Dummy) para US$ 72,47.
</div>

<div class="highlight-card" markdown>
<span class="status-pill">Achado 2</span>

### Tipo de quarto se destaca

Na análise por permutação, `room_type` foi a variável mais importante, seguida por `longitude` e `minimum_nights`.
</div>

<div class="highlight-card" markdown>
<span class="status-pill">Achado 3</span>

### O erro se concentra nos anúncios caros

Até US$ 150 o erro médio absoluto fica entre US$ 22 e 36; acima de US$ 400 o modelo subestima, em média, US$ 267.
</div>
</div>

!!! note "Escopo da métrica"
    O notebook calcula o percentil 99 do preço somente no treino (US$ 798,76, impresso como US$ 799) e remove valores acima desse limite tanto do treino quanto do teste. As métricas desta documentação correspondem a essa avaliação filtrada, não a todos os anúncios originais.

## Autoria

**Carolina Amorim** e **Lara Abduni**<br>
Projeto de Machine Learning com foco em regressão e avaliação de modelos.
