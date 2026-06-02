import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.predictor import Predictor

CHECKPOINT = "outputs/best_model.pt"
STATIC_DIR = Path(__file__).parent / "static"
PDB_DIR = Path(__file__).parent.parent / "PDB"

app = FastAPI(title="OncovVision")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

predictor: Predictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    if not Path(CHECKPOINT).exists():
        print(f"[WARN] Checkpoint no encontrado: {CHECKPOINT}")
        return
    predictor = Predictor(CHECKPOINT)
    print("[OK] Modelo cargado.")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(503, "Modelo no disponible")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen")

    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "No se pudo leer la imagen")

    result = predictor.predict(pil_img)
    return JSONResponse(result)


# ─── PDB / Visor Molecular ───────────────────────────────────────────────
def _safe_pdb_path(name: str) -> Path:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Nombre de archivo inválido")
    fp = PDB_DIR / name
    if not fp.exists() or fp.suffix.lower() != ".pdb":
        raise HTTPException(404, "PDB no encontrado")
    return fp


@app.get("/api/pdb/list")
def pdb_list():
    if not PDB_DIR.exists():
        return {"files": []}
    files = sorted(p.name for p in PDB_DIR.glob("*.pdb"))
    return {"files": files}


@app.get("/api/pdb/file/{name}")
def pdb_file(name: str):
    fp = _safe_pdb_path(name)
    return FileResponse(str(fp), media_type="chemical/x-pdb", filename=name)


@app.get("/api/pdb/info/{name}")
def pdb_info(name: str):
    fp = _safe_pdb_path(name)
    info = {
        "name": name,
        "size_kb": round(fp.stat().st_size / 1024, 1),
        "models": 0,
        "chains": 0,
        "chain_ids": [],
        "residues": 0,
        "atoms": 0,
        "hetatm": 0,
        "title": "",
    }
    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(name, str(fp))
        models = list(structure)
        info["models"] = len(models)
        chain_ids: list[str] = []
        residues = 0
        atoms = 0
        hetatm = 0
        if models:
            for chain in models[0]:
                chain_ids.append(chain.id)
                for residue in chain:
                    residues += 1
                    if residue.id[0].strip():
                        hetatm += 1
                    for _ in residue:
                        atoms += 1
        info["chains"] = len(chain_ids)
        info["chain_ids"] = chain_ids
        info["residues"] = residues
        info["atoms"] = atoms
        info["hetatm"] = hetatm
        header = getattr(structure, "header", {}) or {}
        title = header.get("name") or header.get("head") or ""
        info["title"] = str(title)[:160]
    except Exception as e:
        info["error"] = f"Biopython no pudo procesar el archivo: {e}"
    return info
