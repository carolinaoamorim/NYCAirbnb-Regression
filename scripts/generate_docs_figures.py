"""Exporta as figuras salvas no notebook; não cria gráficos nem recalcula dados."""

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "nycairbnb.ipynb"
IMAGE_DIR = ROOT / "docs" / "assets" / "images"


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    figures = {}
    for number, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        outputs = cell.get("outputs", [])
        if any(output.get("output_type") == "error" for output in outputs):
            raise ValueError(f"A célula {number} contém erro; execute e salve o notebook.")
        images = [
            output["data"]["image/png"]
            for output in outputs
            if "image/png" in output.get("data", {})
        ]
        names = cell.get("metadata", {}).get("docs_figures", [])
        if len(images) != len(names):
            raise ValueError(
                f"Célula {number}: {len(images)} figuras e {len(names)} nomes em "
                "metadata.docs_figures. Execute e salve o notebook antes de exportar."
            )
        for name, payload in zip(names, images):
            if Path(name).name != name or not name.endswith(".png") or name in figures:
                raise ValueError(f"Nome de figura inválido ou repetido: {name}")
            if isinstance(payload, list):
                payload = "".join(payload)
            figures[name] = base64.b64decode(payload)

    if not figures:
        raise ValueError("Nenhuma figura encontrada nos resultados salvos do notebook.")

    # Só escreve após verificar todas as células; mantém os PNGs exatamente como salvos.
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for name, contents in figures.items():
        (IMAGE_DIR / name).write_bytes(contents)
    print(f"{len(figures)} figuras exportadas de {NOTEBOOK_PATH.name} para {IMAGE_DIR}")


if __name__ == "__main__":
    main()
