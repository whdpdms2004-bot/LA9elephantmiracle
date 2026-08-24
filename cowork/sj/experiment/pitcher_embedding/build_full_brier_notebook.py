from pathlib import Path

import nbformat as nbf


here = Path(__file__).resolve().parent
source_path = here / "pitcher_embedding_brier_submission.ipynb"
output_path = here / "pitcher_embedding_brier_full.ipynb"

nb = nbf.read(source_path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

config = nb.cells[1].source
config = config.replace("QUICK_RUN = True", "QUICK_RUN = False")
config = config.replace("EPOCHS = 3 if QUICK_RUN else 12", "EPOCHS = 4")
nb.cells[1].source = config
nb.cells[0].source = nb.cells[0].source.replace(
    "투수 임베딩 v1 — Brier 최적화와 제출 artifact",
    "투수 임베딩 v1 FULL — Brier 최적화와 제출 artifact",
)

nbf.write(nb, output_path)
print(output_path)
