# Setup e Execução

## Requisitos

- Python 3.10 ou superior
- Ambiente virtual recomendado
- Arquivo `AB_NYC_2019.csv` no diretório do projeto (já incluído)

## Instalação

Execute os comandos a partir da pasta `NYCAirbnb-Regression`.

=== "Windows PowerShell"

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```

=== "macOS/Linux"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Dependências de Análise

As bibliotecas principais são:

- `pandas` e `numpy` para manipulação de dados;
- `matplotlib` e `seaborn` para visualização;
- `scikit-learn` para pipelines, validação e modelos;
- `scipy` para distribuições usadas na busca de hiperparâmetros;
- `shap` para interpretação do ensemble.

## Executando o Notebook

1. Abra `nycairbnb.ipynb` no Jupyter ou no VS Code.
2. Confirme que `AB_NYC_2019.csv` está na mesma pasta do notebook.
3. Execute todas as células com **Restart & Run All**.

O notebook define uma semente fixa para os splits, buscas e amostras, de modo que os passos aleatórios possam ser reproduzidos nas mesmas condições.

## Rodando a Documentação Localmente

Instale as dependências de documentação:

```powershell
pip install -r requirements-docs.txt
```

Sirva o site em modo de desenvolvimento:

```powershell
mkdocs serve
```

Abra o endereço indicado no terminal, normalmente `http://127.0.0.1:8000/`.

## Regenerando os Gráficos Descritivos

Os gráficos calculados diretamente do CSV podem ser atualizados sem executar nem modificar o notebook:

```powershell
python scripts/generate_docs_figures.py
```

Os gráficos de validação, resíduos e SHAP foram exportados dos resultados já salvos no notebook. Para atualizá-los, execute novamente as células correspondentes no notebook e exporte os novos resultados.

## Build de Verificação

```powershell
mkdocs build --strict
```

O modo `--strict` ajuda a localizar links quebrados, páginas ausentes e problemas de configuração.

## Publicação no GitHub Pages

O repositório está configurado para publicar a documentação pelo GitHub Actions. Para ativar a publicação automática:

1. Envie os arquivos do projeto para o branch `main` do repositório `carolinaoamorim/NYCAirbnb-Regression`.
2. No GitHub, abra **Settings → Pages** e selecione **GitHub Actions** em *Build and deployment → Source*.
3. Abra a aba **Actions** e aguarde a execução do workflow **Deploy MkDocs to GitHub Pages**.
4. Quando o workflow terminar, acesse <https://carolinaoamorim.github.io/NYCAirbnb-Regression/>.

Depois da configuração inicial, cada atualização enviada ao `main` publica uma nova versão. O endereço público já está definido em `mkdocs.yml` e no README.
