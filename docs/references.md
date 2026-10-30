# Referências

## Dataset

- GOMONOV, D. [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data). Kaggle, 2019. Arquivo usado: `AB_NYC_2019.csv`.

## Bibliografia (seções 1.3, 3.3 e 4)

- BRECK, E.; CAI, S.; NIELSEN, E.; SALIB, M.; SCULLEY, D. [The ML test score: a rubric for ML production readiness and technical debt reduction](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/). In: *IEEE International Conference on Big Data*, 2017. Fundamenta testes e monitoramento para disponibilizar o pipeline.
- DUAN, N. [Smearing estimate: a nonparametric retransformation method](https://doi.org/10.1080/01621459.1983.10478017). *Journal of the American Statistical Association*, v. 78, n. 383, p. 605–610, 1983. Fundamenta a leitura do efeito da transformação log do alvo (seção 1.3).
- GAMA, J.; ŽLIOBAITĖ, I.; BIFET, A.; PECHENIZKIY, M.; BOUCHACHIA, A. [A survey on concept drift adaptation](https://doi.org/10.1145/2523813). *ACM Computing Surveys*, v. 46, n. 4, art. 44, 2014. Fundamenta a discussão sobre mudança temporal e manutenção.
- GIBBS, C.; GUTTENTAG, D.; GRETZEL, U.; MORTON, J.; GOODWILL, A. [Pricing in the sharing economy: a hedonic pricing model applied to Airbnb listings](https://doi.org/10.1080/10548408.2017.1308292). *Journal of Travel & Tourism Marketing*, v. 35, n. 1, p. 46–56, 2018. Contextualiza a modelagem de preços anunciados a partir de atributos de anúncios.
- KAUFMAN, S.; ROSSET, S.; PERLICH, C.; STITELMAN, O. [Leakage in data mining: formulation, detection, and avoidance](https://doi.org/10.1145/2382577.2382579). *ACM Transactions on Knowledge Discovery from Data*, v. 6, n. 4, art. 15, 2012. Fundamenta o risco de vazamento entre anúncios agrupados.
- KOENKER, R.; BASSETT, G. [Regression quantiles](https://doi.org/10.2307/1913643). *Econometrica*, v. 46, n. 1, p. 33–50, 1978. Fundamenta a proposta de regressão quantílica.
- LEI, J.; G'SELL, M.; RINALDO, A.; TIBSHIRANI, R. J.; WASSERMAN, L. [Distribution-free predictive inference for regression](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf). *Journal of the American Statistical Association*, v. 113, n. 523, p. 1094–1111, 2018. Fundamenta a proposta de intervalos conformais e suas condições de cobertura.
- LUNDBERG, S. M.; LEE, S.-I. [A unified approach to interpreting model predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). In: *Advances in Neural Information Processing Systems 30*, 2017. Fundamenta os valores SHAP (seção 3.3).
- NOVA YORK (Cidade). *Local Law 18 of 2022* — Short-Term Rental Registration Law. Mayor's Office of Special Enforcement, 2022. Fundamenta a mudança regulatória do mercado.
- ROBERTS, D. R. et al. [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/ecog.02881). *Ecography*, v. 40, n. 8, p. 913–929, 2017. Fundamenta a proposta de validação por grupos ou blocos espaciais.
- SCULLEY, D. et al. [Hidden technical debt in machine learning systems](https://proceedings.neurips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html). *Advances in Neural Information Processing Systems*, v. 28, 2015. Fundamenta versionamento e acompanhamento de dependências do sistema.

## Documentação técnica consultada

- [Scikit-learn — OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): agrupamento de categorias e interpretação de `drop=None` em modelos regularizados.
- [Scikit-learn — HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html): modelo e motivo para fixar `early_stopping=False`.
- [Scikit-learn — transformação do alvo em regressão](https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html): uso de `TransformedTargetRegressor` e retorno à escala original.
- [SHAP — TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html): explicações de árvores, dependência entre features e distribuição de referência.

## Bibliotecas

- [Python](https://www.python.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [SciPy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/), [SHAP](https://shap.readthedocs.io/).
- Documentação: [MkDocs](https://www.mkdocs.org/) e [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
