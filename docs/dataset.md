# Dataset

## Fonte

O projeto utiliza o [**New York City Airbnb Open Data**](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), publicado no Kaggle. O arquivo local `AB_NYC_2019.csv` reúne anúncios de Airbnb em Nova York e está incluído no projeto.

!!! info "Variável-alvo"
    A coluna `price` representa o preço anunciado por noite, em dólares americanos. Ela não informa o valor efetivamente pago, a ocupação do anúncio ou os descontos e taxas aplicados a uma reserva.

## Características Gerais

| Métrica | Valor |
|---|---:|
| Registros no arquivo original | 48.895 |
| Colunas | 16 |
| Variável-alvo | `price` |
| Preços iguais a zero | 11 |
| Registros sem `last_review` | 10.052 |
| Registros sem `reviews_per_month` | 10.052 |
| Registros sem `name` | 16 |
| Registros sem `host_name` | 21 |
| Distritos (`neighbourhood_group`) | 5 |
| Tipos de quarto (`room_type`) | 3 |

## Dicionário de Dados

| Coluna | Descrição |
|---|---|
| `id` | Identificador do anúncio |
| `name` | Nome ou descrição curta do anúncio |
| `host_id` | Identificador do anfitrião |
| `host_name` | Nome do anfitrião |
| `neighbourhood_group` | Distrito de Nova York |
| `neighbourhood` | Bairro do anúncio |
| `latitude` | Latitude da localização |
| `longitude` | Longitude da localização |
| `room_type` | Tipo de acomodação anunciado |
| `price` | Preço anunciado por noite, em dólares |
| `minimum_nights` | Número mínimo de noites por reserva |
| `number_of_reviews` | Total de avaliações do anúncio |
| `last_review` | Data da avaliação mais recente |
| `reviews_per_month` | Média mensal de avaliações |
| `calculated_host_listings_count` | Quantidade de anúncios associados ao anfitrião |
| `availability_365` | Dias de disponibilidade informados para os próximos 365 dias |

## Estatísticas do Alvo Apresentadas no Notebook

A célula de inspeção executa `df["price"].describe(percentiles=[.5, .9, .99]).round(1)`. O resumo abaixo reproduz essa saída sobre o arquivo original, antes da exclusão de preços zero. Não inclui categorias nem estatísticas de identificadores.

| Estatística de `price` | Valor |
|---|---:|
| Contagem | 48.895 |
| Média, US$ | 152,7 |
| Desvio padrão, US$ | 240,2 |
| Mínimo, US$ | 0,0 |
| Mediana, US$ | 106,0 |
| Percentil 90, US$ | 269,0 |
| Percentil 99, US$ | 799,0 |
| Máximo, US$ | 10.000,0 |

A média superior à mediana e o máximo distante do P99 evidenciam a cauda à direita. O corte usado na modelagem é recalculado apenas no treino, e não copiado desse resumo do arquivo completo. A [análise](analysis.md) explicita o vínculo entre cada evidência e a decisão de pré-processamento.

## Valores Ausentes e Pontos de Atenção

| Ponto observado | Tratamento no notebook | Observação |
|---|---|---|
| `reviews_per_month` ausente em anúncios sem avaliações | Substituído por zero | O notebook verifica que os valores ausentes nessa coluna correspondem a `number_of_reviews = 0`. |
| Coordenadas, mínimo de noites, avaliações e disponibilidade | Imputação preventiva pela mediana no pipeline | Não há nulos nessas colunas na inspeção apresentada. O imputador é ajustado em cada treino, sem antecipar informações do teste. |
| `name`, `host_name`, `last_review` | Colunas removidas | Não entram no modelo final. |
| `id` e `host_id` | Colunas removidas dos preditores | São identificadores, não características generalizáveis do anúncio. |
| `price = 0` | Registros removidos | Preço zero não é útil para o alvo de regressão. |
| Preços extremos | Corte no percentil 99 do treino | O limite de US$ 798,76 (impresso como US$ 799) é calculado no treino e aplicado também ao teste. Isso restringe o escopo das métricas. |
| Bairros pouco frequentes | Agrupados no encoder | Bairros com menos de 50 anúncios no treino são tratados como infrequentes. |

## Leitura Inicial

O dataset combina variáveis numéricas, categóricas, localização e informações incompletas sobre avaliações. Isso possibilita comparar modelos lineares e não lineares, mas exige separar os identificadores dos preditores, tratar as ausências dentro de um pipeline e interpretar os resultados à luz do período e da forma como o preço foi coletado.
