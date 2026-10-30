# Dataset

## Fonte

O projeto utiliza o [**New York City Airbnb Open Data**](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), publicado no Kaggle. O arquivo local `AB_NYC_2019.csv` reúne anúncios de Airbnb em Nova York e está incluído no projeto.

!!! info "Variável-alvo"
    A coluna `price` é o preço por noite do anúncio, em dólares americanos.

## Características Gerais

Apenas informações estruturais do arquivo. Estatísticas e figuras usadas nas decisões são calculadas **somente no
treino**, depois do split, e estão na [seção 0 da análise](analysis.md#0-dados-e-pipeline-de-pre-processamento).

| Métrica | Valor |
|---|---:|
| Registros no arquivo original | 48.895 |
| Colunas | 16 |
| Variável-alvo | `price` |
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
| `room_type` | Tipo de acomodação |
| `price` | Preço por noite, em dólares |
| `minimum_nights` | Número mínimo de noites por reserva |
| `number_of_reviews` | Total de avaliações do anúncio |
| `last_review` | Data da avaliação mais recente |
| `reviews_per_month` | Média mensal de avaliações |
| `calculated_host_listings_count` | Quantidade de anúncios associados ao anfitrião |
| `availability_365` | Dias de disponibilidade informados para os próximos 365 dias |

## Tratamentos Aplicados

| Ponto observado (no treino) | Tratamento no notebook | Evidência |
|---|---|---|
| `reviews_per_month` ausente em anúncios sem avaliações | Substituído por zero | 8.012 ausentes no treino, todos com `number_of_reviews = 0` (verificado no notebook). |
| Coordenadas, mínimo de noites, avaliações e disponibilidade | Imputação preventiva pela mediana no pipeline | Sem nulos no treino; o imputador é preventivo e ajustado em cada fold. |
| `name`, `host_name`, `last_review` | Colunas removidas | Não entram no modelo final. |
| `id` e `host_id` | Colunas removidas dos preditores | São identificadores, não características generalizáveis do anúncio. |
| `price = 0` | Registros removidos | Regra de validade (11 anúncios), aplicada antes do split. |
| Preços extremos | Corte no percentil 99 do treino | Figura 1: 1% dos anúncios concentra 83% dos desvios quadráticos. Limite de US$ 798,76, calculado no treino e aplicado também ao teste. |
| Bairros pouco frequentes | Agrupados no encoder | Figura 2: 129 de 217 bairros têm menos de 50 anúncios no treino (4,8% dos anúncios) e são agrupados como infrequentes. |

## Leitura Inicial

O dataset combina variáveis numéricas, categóricas, localização e informações incompletas sobre avaliações. Isso possibilita comparar modelos lineares e não lineares, mas exige separar os identificadores dos preditores, tratar as ausências dentro de um pipeline e controlar a dimensão do One-Hot Encoding dos bairros.
