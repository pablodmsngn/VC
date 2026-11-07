#!/usr/bin/env python3
"""Prepara el dataset de matrículas para entrenar con YOLO.

Uso (desde la carpeta dataset/):
    python preparar_dataset.py unir      # junta train/val/test en completo/{images,annotations}
    python preparar_dataset.py separar   # reparte completo/ en train/val/test, genera labels YOLO,
                                         # data.yaml y ground_truth.csv

Ambos pasos MUEVEN archivos (no copian) para no duplicar los ~7 GB de imágenes.

- Reparto 64/16/20 (80% train+val y 20% test, y dentro del 80% un 80/20), como sugiere el enunciado.
- Las imágenes que comparten alguna matrícula legible van siempre al mismo split, para que el
  mismo coche no aparezca a la vez en train y en test (fuga de datos).
- Las anotaciones sin imagen se apartan en descartados/annotations.
- Formato de entrada (JSON, coordenadas normalizadas):
      [{"rectangle": [cx, cy, w, h], "plate": "0000AAA" | "None"}, ...]
- Formato YOLO de salida (labels/<imagen>.txt):  0 cx cy w h
"""
import csv
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
SPLITS = ["train", "val", "test"]
PROPORCIONES = {"train": 0.64, "val": 0.16, "test": 0.20}
SEMILLA = 42
COMPLETO = RAIZ / "completo"
DESCARTADOS = RAIZ / "descartados"
EXT_IMAGEN = {".jpg", ".jpeg", ".png"}
ILEGIBLES = {"", "NONE", "NULL", "NAN"}


def mover(origen: Path, destino_dir: Path):
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / origen.name
    if destino.exists():
        sys.exit(f"ERROR: {destino} ya existe, no se sobrescribe")
    os.replace(origen, destino)


def carpetas_origen():
    """Carpetas images/annotations de los splits existentes."""
    for s in SPLITS:
        for sub in ("images", "annotations"):
            d = RAIZ / s / sub
            if d.is_dir():
                yield sub, d


def unir():
    archivos = [(sub, p) for sub, d in carpetas_origen() for p in d.iterdir() if p.is_file()]
    nombres = Counter((sub, p.name.lower()) for sub, p in archivos)
    repetidos = [n for n, c in nombres.items() if c > 1]
    ya = {(sub, p.name.lower()) for sub in ("images", "annotations")
          for p in (COMPLETO / sub).glob("*")} if COMPLETO.exists() else set()
    choques = repetidos + [n for n in nombres if n in ya]
    if choques:
        sys.exit(f"ERROR: nombres repetidos, no se mueve nada: {choques[:10]}")
    for sub, p in archivos:
        mover(p, COMPLETO / sub)
    # borrar carpetas de split que hayan quedado vacías
    for s in SPLITS:
        for sub in ("images", "annotations", "labels"):
            d = RAIZ / s / sub
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if (RAIZ / s).is_dir() and not any((RAIZ / s).iterdir()):
            (RAIZ / s).rmdir()
    print(f"Unidos {len(archivos)} archivos en {COMPLETO}")


def normalizar_matricula(texto):
    t = re.sub(r"[^A-Z0-9]", "", str(texto).upper())
    return None if t in ILEGIBLES else t


def leer_anotacion(ruta: Path):
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    cajas, invalidas = [], 0
    for obj in datos:
        cx, cy, w, h = obj["rectangle"]
        if w <= 0 or h <= 0:
            invalidas += 1
            continue
        # recortar la caja a los límites de la imagen
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        if x2 <= x1 or y2 <= y1:
            invalidas += 1
            continue
        cajas.append({
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": x2 - x1, "h": y2 - y1,
            "texto_original": obj.get("plate"),
            "matricula": normalizar_matricula(obj.get("plate")),
        })
    return cajas, invalidas


def agrupar(muestras):
    """Union-find: imágenes que comparten una matrícula legible forman un grupo."""
    padre = list(range(len(muestras)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    primera = {}
    for i, m in enumerate(muestras):
        for c in m["cajas"]:
            if c["matricula"]:
                if c["matricula"] in primera:
                    padre[raiz(i)] = raiz(primera[c["matricula"]])
                else:
                    primera[c["matricula"]] = i
    grupos = {}
    for i in range(len(muestras)):
        grupos.setdefault(raiz(i), []).append(i)
    return list(grupos.values())


def repartir(grupos, total):
    rng = random.Random(SEMILLA)
    rng.shuffle(grupos)
    grupos.sort(key=len, reverse=True)  # los grupos grandes primero, el resto en orden aleatorio
    objetivo = {s: PROPORCIONES[s] * total for s in SPLITS}
    actual = {s: 0 for s in SPLITS}
    asignacion = {}
    for g in grupos:
        s = max(SPLITS, key=lambda k: (objetivo[k] - actual[k]) / objetivo[k])
        for i in g:
            asignacion[i] = s
        actual[s] += len(g)
    return asignacion


def separar():
    img_dir, ann_dir = COMPLETO / "images", COMPLETO / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        sys.exit("ERROR: no existe completo/images o completo/annotations. Ejecuta antes 'unir'.")

    imagenes = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in EXT_IMAGEN}
    anotaciones = {p.stem: p for p in ann_dir.glob("*.json")}

    # anotaciones sin imagen -> descartados
    huerfanas = sorted(set(anotaciones) - set(imagenes))
    for stem in huerfanas:
        mover(anotaciones.pop(stem), DESCARTADOS / "annotations")
    sin_anotacion = sorted(set(imagenes) - set(anotaciones))
    for stem in sin_anotacion:
        mover(imagenes.pop(stem), DESCARTADOS / "images")

    muestras, cajas_invalidas = [], 0
    for stem in sorted(imagenes):
        cajas, inv = leer_anotacion(anotaciones[stem])
        cajas_invalidas += inv
        muestras.append({"stem": stem, "img": imagenes[stem], "ann": anotaciones[stem], "cajas": cajas})

    grupos = agrupar(muestras)
    asignacion = repartir(grupos, len(muestras))

    filas_gt = []
    for i, m in enumerate(muestras):
        s = asignacion[i]
        mover(m["img"], RAIZ / s / "images")
        mover(m["ann"], RAIZ / s / "annotations")
        etiquetas = RAIZ / s / "labels"
        etiquetas.mkdir(parents=True, exist_ok=True)
        lineas = [f"0 {c['cx']:.6f} {c['cy']:.6f} {c['w']:.6f} {c['h']:.6f}" for c in m["cajas"]]
        (etiquetas / f"{m['stem']}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
        for k, c in enumerate(m["cajas"]):
            filas_gt.append({
                "imagen": m["img"].name, "split": s, "caja": k,
                "matricula": c["matricula"] or "", "legible": int(bool(c["matricula"])),
                "texto_original": c["texto_original"],
                "cx": f"{c['cx']:.6f}", "cy": f"{c['cy']:.6f}", "w": f"{c['w']:.6f}", "h": f"{c['h']:.6f}",
            })

    for d in (img_dir, ann_dir, COMPLETO):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()

    with open(RAIZ / "ground_truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_gt[0]))
        w.writeheader()
        w.writerows(filas_gt)

    ruta = RAIZ.as_posix()
    (RAIZ / "data.yaml").write_text(
        f"# Dataset de matrículas (generado por preparar_dataset.py)\n"
        f"path: {ruta}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n\n"
        f"nc: 1\n"
        f"names: ['matricula']\n",
        encoding="utf-8",
    )

    # resumen
    por_split = Counter(asignacion.values())
    cajas_split = Counter()
    legibles_split = Counter()
    for f in filas_gt:
        cajas_split[f["split"]] += 1
        legibles_split[f["split"]] += f["legible"]
    print(f"Imágenes válidas: {len(muestras)}  |  grupos (por matrícula compartida): {len(grupos)}")
    print(f"Anotaciones sin imagen -> descartados/annotations: {len(huerfanas)}")
    print(f"Imágenes sin anotación -> descartados/images: {len(sin_anotacion)}")
    print(f"Cajas inválidas ignoradas: {cajas_invalidas}")
    for s in SPLITS:
        print(f"  {s:5s}: {por_split[s]:5d} imágenes ({por_split[s] / len(muestras):.1%}), "
              f"{cajas_split[s]:5d} matrículas, {legibles_split[s]:5d} legibles")
    print("Generados: train/val/test/{images,annotations,labels}, data.yaml, ground_truth.csv")


if __name__ == "__main__":
    pasos = {"unir": unir, "separar": separar}
    if len(sys.argv) != 2 or sys.argv[1] not in pasos:
        sys.exit(__doc__)
    pasos[sys.argv[1]]()
