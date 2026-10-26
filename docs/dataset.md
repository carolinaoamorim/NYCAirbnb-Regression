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

## Estatísticas das Variáveis Numéricas

Resumo calculado sobre o CSV original. `reviews_per_month` tem 10.052 observações ausentes; por isso, suas estatísticas consideram apenas os 38.843 valores disponíveis. Os identificadores `id` e `host_id` aparecem para completar a inspeção do esquema, mas não são usados como preditores.

| Variável | Média | Mediana | Desvio padrão |
|---|---:|---:|---:|
| `id` | 19.017.143,24 | 19.677.284 | 10.983.108,39 |
| `host_id` | 67.620.010,65 | 30.793.816 | 78.610.967,03 |
| `latitude` | 40,7289 | 40,7231 | 0,0545 |
| `longitude` | -73,9522 | -73,9557 | 0,0462 |
| `price` (US$) | 152,72 | 106 | 240,15 |
| `minimum_nights` | 7,03 | 3 | 20,51 |
| `number_of_reviews` | 23,27 | 5 | 44,55 |
| `reviews_per_month` | 1,37 | 0,72 | 1,68 |
| `calculated_host_listings_count` | 7,14 | 1 | 32,95 |
| `availability_365` | 112,78 | 45 | 131,62 |

Médias bem acima das medianas e desvios padrão elevados em preço, noites mínimas, avaliações e disponibilidade indicam distribuições assimétricas e/ou dispersas. A média dos identificadores não tem interpretação estatística útil. As features numéricas mais assimétricas e os preços extremos justificam complementar médias com medianas, visualizações robustas e modelos avaliados no mesmo recorte de preço.

## Valores Ausentes e Pontos de Atenção

| Ponto observado | Tratamento no notebook | Observação |
|---|---|---|
| `reviews_per_month` ausente em anúncios sem avaliações | Substituído por zero | O notebook verifica que os valores ausentes nessa coluna correspondem a `number_of_reviews = 0`. |
| Coordenadas, mínimo de noites, avaliações e disponibilidade | Imputação pela mediana no pipeline | O imputador é ajustado dentro do pipeline para não antecipar informações do teste. |
| `name`, `host_name`, `last_review` | Colunas removidas | Não entram no modelo final. |
| `id` e `host_id` | Colunas removidas dos preditores | São identificadores, não características generalizáveis do anúncio. |
| `price = 0` | Registros removidos | Preço zero não é útil para o alvo de regressão. |
| Preços extremos | Corte no percentil 99 do treino | O limite de US$ 799 é calculado no treino e aplicado também ao teste. |
| Bairros pouco frequentes | Agrupados no encoder | Bairros com menos de 50 anúncios no treino são tratados como infrequentes. |

## Distribuição dos Anúncios

O arquivo contém anúncios nos cinco distritos abaixo. Manhattan e Brooklyn concentram a maior parte dos registros.

| Distrito | Anúncios |
|---|---:|
| Manhattan | 21.661 |
| Brooklyn | 20.104 |
| Queens | 5.666 |
| Bronx | 1.091 |
| Staten Island | 373 |

Os tipos de quarto mais frequentes são **Entire home/apt** (25.409 anúncios), **Private room** (22.326) e **Shared room** (1.160).

!!! warning "Defasagem temporal"
    Os registros representam um retrato de 2019. Mudanças posteriores no mercado de hospedagem e nas regras locais de Nova York limitam a validade de previsões atuais sem dados recentes e nova avaliação.

## Leitura Inicial

O dataset combina variáveis numéricas, categóricas, localização e informações incompletas sobre avaliações. Isso possibilita comparar modelos lineares e não lineares, mas exige separar os identificadores dos preditores, tratar as ausências dentro de um pipeline e interpretar os resultados à luz do período e da forma como o preço foi coletado.
