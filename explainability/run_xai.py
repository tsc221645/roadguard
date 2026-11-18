"""Auditoría XAI reproducible por oclusión para el detector B de RoadGuard."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from ml.operational_metrics import overlap, truth_boxes

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/runs/model_b/weights/best.pt"
IMAGES = ROOT / "data/processed/rdd2022/images/test"
LABELS = ROOT / "data/processed/rdd2022/labels/test"
DEFAULT_OUTPUT = ROOT / "explainability/outputs"
CLASSES = ("D00", "D10", "D20", "D40")


def image_at(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"No se pudo leer {path}")
    return image


def rows(result) -> list[dict]:
    boxes = result.boxes
    if boxes is None:
        return []
    return [{"class_id": int(cls), "class_name": result.names[int(cls)],
             "confidence": float(conf), "xyxy": list(map(float, box))}
            for cls, conf, box in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                                      boxes.xyxy.cpu().tolist())]


def match_truth(predictions: list[dict], truths: list[tuple], limit: float = .5) -> tuple[list[dict], int]:
    """Emparejamiento uno a uno por clase/IoU, ordenado por confianza."""
    used = set()
    matched = []
    for pred in sorted(predictions, key=lambda r: -r["confidence"]):
        candidates = [(overlap(pred["xyxy"], box), idx) for idx, (cls, box) in enumerate(truths)
                      if cls == pred["class_id"] and idx not in used]
        iou, idx = max(candidates, default=(0., -1))
        tp = iou >= limit
        if tp:
            used.add(idx)
        matched.append({**pred, "status": "TP" if tp else "FP", "iou_gt": iou,
                        "ground_truth": list(map(float, truths[idx][1])) if tp else None})
    return matched, len(truths) - len(used)


def target_score(predictions: list[dict], target: dict, limit: float = .5) -> float:
    """S=max(confianza*IoU) entre cajas de igual clase e IoU>=limit; ausente=0."""
    scores = [r["confidence"] * iou for r in predictions
              if r["class_id"] == target["class_id"]
              for iou in [overlap(r["xyxy"], target["xyxy"])] if iou >= limit]
    return max(scores, default=0.)


def grid_boxes(height: int, width: int, grid: int) -> list[tuple[int, int, int, int]]:
    xs = np.linspace(0, width, grid + 1, dtype=int)
    ys = np.linspace(0, height, grid + 1, dtype=int)
    return [(int(xs[c]), int(ys[r]), int(xs[c+1]), int(ys[r+1]))
            for r in range(grid) for c in range(grid)]


def hide(image: np.ndarray, fill: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    copy = image.copy()
    copy[y1:y2, x1:x2] = fill[y1:y2, x1:x2]
    return copy


def sync(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def predict(model: YOLO, source, device: str, batch: int = 1):
    return model.predict(source=source, conf=.10, imgsz=640, iou=.50, device=device,
                         batch=batch, stream=False, verbose=False, save=False)


def prediction_signature(result) -> list[tuple]:
    """Canonical predictions for the before/after XAI consistency check."""
    return sorted((r["class_id"], round(r["confidence"], 5),
                   tuple(round(x, 3) for x in r["xyxy"])) for r in rows(result))


def explain(model: YOLO, image: np.ndarray, target: dict, device: str,
            grid: int = 8, batch: int = 16, fill_method: str = "blur") -> dict:
    boxes = grid_boxes(*image.shape[:2], grid)
    if fill_method == "blur":
        fill = cv2.GaussianBlur(image, (51, 51), 0)
    elif fill_method == "median":
        fill = np.broadcast_to(np.median(image.reshape(-1, 3), axis=0).astype(np.uint8), image.shape).copy()
    else:
        raise ValueError("Reemplazo inválido")
    scores = []
    sync(device); start = time.perf_counter()
    for offset in range(0, len(boxes), batch):
        chunk = boxes[offset:offset+batch]
        results = predict(model, [hide(image, fill, box) for box in chunk], device, len(chunk))
        scores.extend(target_score(rows(result), target) for result in results)
    sync(device)
    baseline = target.get("reference_score", target["confidence"])
    values = (baseline - np.asarray(scores)).reshape(grid, grid)
    if not np.isfinite(values).all():
        raise ValueError("Atribución no finita")
    return {"signed_importance": values.tolist(), "occluded_scores": scores,
            "reference_score": baseline,
            "seconds": time.perf_counter()-start, "inferences": len(boxes),
            "grid": grid, "replacement": fill_method}


def select(inventory: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    used = set()
    local = []
    def take(reason: str, pool: list[dict]) -> None:
        row = next((r for r in pool if r["image"] not in used), None)
        if row is None:
            raise ValueError(f"Sin caso: {reason}")
        local.append({**row, "reason": reason})
        used.add(row["image"])
    take("TP alta", sorted((r for r in inventory if r["status"] == "TP"), key=lambda r: (-r["confidence"], r["image"])))
    take("FP", sorted((r for r in inventory if r["status"] == "FP"), key=lambda r: (-r["confidence"], r["image"])))
    take("Cerca de 0.10", sorted(inventory, key=lambda r: (abs(r["confidence"]-.10), r["image"])))
    take("D40 TP", sorted((r for r in inventory if r["status"] == "TP" and r["class_name"] == "D40"), key=lambda r: (-r["confidence"], r["image"])))
    take("Otra clase", sorted((r for r in inventory if r["status"] == "TP" and r["class_name"] in {"D10", "D20"}), key=lambda r: (r["class_name"], -r["confidence"], r["image"])))
    global_rows = []
    for cls in CLASSES:
        pool = [r for r in inventory if r["class_name"] == cls and r["image"] not in used]
        pool.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['image']}".encode()).hexdigest())
        for row in pool:
            if row["image"] not in used:
                global_rows.append(row); used.add(row["image"])
                if sum(x["class_name"] == cls for x in global_rows) == 5:
                    break
    if len(global_rows) != 20:
        raise ValueError(f"Muestra global incompleta: {len(global_rows)}")
    return local, global_rows


def spatial_stats(values: np.ndarray, target: dict, width: int, height: int) -> dict:
    """Weight each cell by its area overlap with the target box, including small boxes."""
    n = values.shape[0]
    x1,y1,x2,y2 = target["xyxy"]
    weights = np.zeros_like(values, dtype=float)
    for i,(left,top,right,bottom) in enumerate(grid_boxes(height,width,n)):
        intersection = max(0., min(right,x2)-max(left,x1))*max(0., min(bottom,y2)-max(top,y1))
        weights.flat[i] = intersection/((right-left)*(bottom-top))
    pos = np.maximum(values,0)
    inside = float((pos*weights).sum()/weights.sum()) if weights.sum() else 0.
    outside_weights = 1-weights
    outside = float((pos*outside_weights).sum()/outside_weights.sum()) if outside_weights.sum() else 0.
    return {"inside_mean": inside, "outside_mean": outside,
            "inside_outside_ratio": inside/outside if outside else None,
            "inside_fraction": float((pos*weights).sum()/pos.sum()) if pos.sum() else None,
            "box_area_fraction": float((x2-x1)*(y2-y1)/(width*height)),
            "negative_cells": int((values<0).sum()), "max_drop": float(values.max())}


def fidelity(model: YOLO, image: np.ndarray, target: dict, values: np.ndarray,
             device: str, seed: int = 42) -> dict:
    """Independent combined-removal check: top/bottom/random four cells."""
    grid = values.shape[0]
    boxes = grid_boxes(*image.shape[:2], grid)
    fill = cv2.GaussianBlur(image, (51, 51), 0)
    flat = values.ravel()
    order = sorted(range(len(boxes)), key=lambda i: (flat[i], -i))
    chosen = {"top": list(reversed(order[-4:])), "bottom": order[:4],
              "random": random.Random(seed).sample(range(len(boxes)), 4)}
    batch = []
    for indices in chosen.values():
        copy = image.copy()
        for i in indices:
            x1,y1,x2,y2 = boxes[i]
            copy[y1:y2,x1:x2] = fill[y1:y2,x1:x2]
        batch.append(copy)
    result = predict(model, batch, device, 3)
    scores = {key: target_score(rows(pred), target) for key,pred in zip(chosen,result)}
    baseline = target.get("reference_score", target["confidence"])
    return {"cells": chosen, "scores": scores,
            "drops": {key: baseline-score for key,score in scores.items()},
            "top_exceeds_bottom": scores["top"] < scores["bottom"],
            "top_exceeds_random": scores["top"] < scores["random"]}


def inventory(model: YOLO, device: str) -> tuple[list[dict], dict]:
    found=[]; totals={"tp":0,"fp":0,"fn":0}
    results=model.predict(source=str(IMAGES),conf=.10,imgsz=640,iou=.50,device=device,
                          stream=True,verbose=False,save=False)
    count=0
    for count,result in enumerate(results,1):
        path=Path(result.path); h,w=result.orig_shape
        truths=truth_boxes(LABELS/(path.stem+".txt"),w,h)
        predictions=rows(result)
        matched,fn=match_truth(predictions,truths)
        totals["fn"]+=fn
        for row in matched:
            row["reference_score"]=target_score(predictions,row)
            row["image"]=path.name
            row["image_path"]=str(path.relative_to(ROOT)).replace("\\","/")
            row["detection_id"]=hashlib.sha256(f"{path.name}:{row['class_id']}:{row['confidence']:.8f}:{row['xyxy']}".encode()).hexdigest()[:16]
            row["all_ground_truth"]=[{"class_id":cls,"xyxy":list(map(float,box))} for cls,box in truths]
            totals[row["status"].lower()]+=1
            found.append(row)
        if count%250==0: print(f"Inventario {count}",flush=True)
    if count!=1072:
        raise ValueError(f"Test incompleto: {count}")
    return found,totals


def case_panel(ax, item: dict) -> None:
    image=image_at(ROOT/item["image_path"])
    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    values=np.maximum(np.asarray(item["xai"]["signed_importance"],dtype=np.float32),0)
    heat=cv2.resize(values,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    ax.imshow(heat,cmap="magma",alpha=np.clip(heat/max(float(values.max()),1e-9)*.6,0,.6))
    for box,color,style in [(item["xyxy"],"cyan","-")]+[(r["xyxy"],"lime","--") for r in item["all_ground_truth"]]:
        if box:
            x1,y1,x2,y2=box
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linestyle=style,linewidth=2))
    ax.set_title(f"{item['reason']} | {item['class_name']} {item['confidence']:.3f} | {item['status']}\n{item['image']}",fontsize=9)
    ax.axis("off")


def figures(local: list[dict], global_rows: list[dict], out: Path) -> None:
    fig,axes=plt.subplots(2,3,figsize=(16,10),layout="constrained")
    for ax,item in zip(axes.flat,local): case_panel(ax,item)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02,.9,"Cian: predicción\nVerde: GT\nMagma (oscuro → amarillo): caída positiva de S\nLos valores negativos están en JSON.\nEl mapa no prueba causalidad.",fontsize=12,va="top")
    fig.savefig(out/"local_cases.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
    means=[statistics.mean(r["stats"]["max_drop"] for r in global_rows if r["class_name"]==cls) for cls in CLASSES]
    axes[0].bar(CLASSES,means);axes[0].set_ylabel("Caída máxima media de S")
    axes[1].boxplot([[r["stats"][key] for r in global_rows] for key in ("inside_mean","outside_mean")],tick_labels=["Dentro","Fuera"])
    axes[1].set_ylabel("Importancia positiva media por celda")
    fig.savefig(out/"global_summary.png",dpi=170);plt.close(fig)


def pdf(local: list[dict], global_rows: list[dict], bench: dict, out: Path) -> None:
    with PdfPages(out/"laboratorio10_xai.pdf") as pages:
        fig=plt.figure(figsize=(11.7,8.3))
        text=("Laboratorio 10 — Explainable AI | RoadGuard AI\n"
              "Objetivo: auditar qué regiones alteran detecciones concretas de daño vial del modelo B.\n\n"
              "Método                 Grad.  Caja negra  Requisito principal       Límite principal\n"
              "Saliency                 sí        no       objetivo diferenciable    ruido local\n"
              "SmoothGrad          sí        no       múltiples muestras        costo y ruido residual\n"
              "Integrated Grad.    sí        no       línea base y trayectoria  depende de línea base\n"
              "Grad-CAM             sí        no       capa y objetivo escalar  resolución y NMS\n"
              "Oclusión                no        sí       rejilla y reemplazo         fuera de distribución\n"
              "Conceptos/TCAV   sí        no       ejemplos de conceptos     calidad de conceptos\n\n"
              "Herramientas: Captum y pytorch-grad-cam; aquí: Ultralytics + OpenCV.\n\n"
              "La mayoría se formuló para clasificación; YOLO requiere un objetivo por caja y clase.\n"
              "Oclusión usa la salida final: no necesita gradientes ni una capa específica. Grad-CAM sería\n"
              "más rápido una vez adaptado, pero la caja final está desacoplada por NMS; aquí no se validó\n"
              "un objetivo diferenciable correspondiente a esa detección.\n\n"
              "S=max(confianza × IoU con caja original), misma clase e IoU≥0,5; ausente=0.\n"
              "Atribución firmada = S(original) − S(ocluida). 8×8, blur 51×51.\n"
              "Esto mide sensibilidad local a una perturbación: no garantiza causalidad ni completitud.\n"
              "La elección de reemplazo, rejilla, NMS, IoU y escala visual cambia el mapa.\n\n"
              "Un sistema interpretable por diseño (p. ej., reglas de bordes) permite inspeccionar su\n"
              "lógica; no es un detector equivalente sin un experimento comparativo nuevo.\n\n"
              "Fuentes: Zeiler y Fergus (2014); Selvaraju et al. (2017); Sundararajan et al. (2017);\n"
              "Smilkov et al. (2017); Kim et al. (2018). Detalles y enlaces en explainability/README.md.")
        fig.text(.06,.94,text,va="top",fontsize=11,linespacing=1.5)
        pages.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(16.5,11.7));fig.suptitle("Cinco explicaciones locales — test Czech, modelo B YOLO11n",fontsize=16)
        fig.text(.05,.94,"best.pt | conf 0,10 | NMS IoU 0,50 | imagen 640 | match GT IoU 0,50 | semilla 42 | cian: predicción; verde: anotación; magma: caída de S",fontsize=10)
        ax=fig.add_axes((.04,.13,.92,.79));ax.imshow(plt.imread(out/"local_cases.png"));ax.axis("off")
        fig.text(.05,.10,"TP D00 y D10: sensibilidad concentrada cerca de la grieta marcada. FP D00: sensibilidad fuerte en la superficie vial,",fontsize=10)
        fig.text(.05,.075,"pero la anotación no confirma daño. D20 cerca del umbral: respuesta dispersa. D40: caja pequeña; la rejilla limita detalle.",fontsize=10)
        fig.text(.05,.05,"Cada mapa corresponde a una caja específica. 64 oclusiones por caso. Los cambios negativos se conservan en results.json.",fontsize=9)
        pages.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(11.7,8.3));fig.suptitle("Explicación global: 20 imágenes adicionales del test",fontsize=16)
        ax=fig.add_axes((.05,.32,.9,.6));ax.imshow(plt.imread(out/"global_summary.png"));ax.axis("off")
        summary="Clase    n   TP/FP  caída máxima media   fracción positiva dentro\n"
        for cls in CLASSES:
            group=[r for r in global_rows if r["class_name"]==cls]
            tp=sum(r["status"]=="TP" for r in group)
            frac=statistics.mean((r["stats"]["inside_fraction"] or 0) for r in group)
            summary+=f"{cls}      {len(group):2d}    {tp}/{len(group)-tp}             {statistics.mean(r['stats']['max_drop'] for r in group):.3f}                       {frac:.3f}\n"
        fig.text(.1,.27,summary,va="top",fontsize=11,family="monospace")
        fig.text(.1,.105,"Selección: 5 detecciones por clase, hash semilla 42; excluye casos locales. No representa todas las imágenes.",fontsize=9)
        fig.text(.1,.081,"TP D00/D10: señal cercana a anotaciones; FP D00/D20: revisar juntas y GT. No se probó fuga ni sesgo de cámara.",fontsize=9)
        fig.text(.1,.057,"Acción: revisión humana de FP ambiguos; validar cualquier cambio de umbral en val. D40: caja pequeña limita la fracción dentro.",fontsize=9)
        fig.text(.1,.033,"Al operador: imagen, caja, confianza, mapa opcional y confirmación. El mapa no certifica la alerta.",fontsize=9)
        pages.savefig(fig);plt.close(fig)
        fig=plt.figure(figsize=(11.7,8.3))
        fidel_top=sum(r["fidelity"]["top_exceeds_bottom"] and r["fidelity"]["top_exceeds_random"] for r in local)
        text=("Discusión, costo y límites\n\n"
              f"RTX 3050, PyTorch {torch.__version__}, Ultralytics {ultralytics.__version__}; lote 16, 640 px.\n"
              f"Inferencia normal: {bench['normal_median_s']:.3f} s (mediana de 3, tras calentamiento).\n"
              f"Explicación: {bench['local_mean_s']:.2f} s/caso (media de 5; 64 inferencias/caso),\n"
              f"{bench['local_mean_s']/bench['normal_median_s']:.0f} veces la inferencia individual; cinco: {bench['five_s']:.2f} s; veinte: {bench['global_s']:.2f} s.\n"
              f"Pico de memoria GPU asignada: {bench['gpu_peak_bytes']/2**20:.0f} MiB.\n\n"
              f"Fidelidad limitada: retirar juntas las 4 celdas principales redujo S más que retirar 4\n"
              f"celdas inferiores o aleatorias en {fidel_top}/5 casos. Una sola muestra aleatoria/caso.\n"
              "TP D00/D10: sensibilidad cerca del daño anotado, como se esperaba; FP D00 y D20:\n"
              "ambigüedad que mAP, precisión y recall agregados no muestran por caja.\n"
              "La detección D20 marginal fue menos localizada; conf 0,10 introduce saltos al desaparecer\n"
              "una caja. La caja D40 pequeña limita la resolución espacial. En el FP D00 podría haber\n"
              "junta o daño sin anotar: requiere revisión manual.\n\n"
              "Grad-CAM podría abaratar el cálculo, pero aquí no se validó su objetivo por caja.\n"
              "Oclusión sirve para auditorías y lotes selectivos; este costo no respalda explicar\n"
              "cada detección en tiempo real. Repetir con otras rejillas/reemplazos antes de decidir cambios.\n"
              "Priorizar revisión de FP ambiguos y FN; evaluar nuevas condiciones y anotaciones.\n"
              "No usar la salida sola para cerrar calles ni descartar daños peligrosos. Fotos con rostros,\n"
              "placas o GPS exigen control de acceso. Sesgos de procedencia e iluminación siguen abiertos.\n"
              "Si un sistema interpretable iguala rendimiento y seguridad, conviene compararlo en\n"
              "un experimento nuevo; el mapa de calor no reemplaza inspección profesional.\n\n"
              "Referencias: arxiv.org/abs/1703.01365; 1706.03825; 1711.11279; 1610.02391.\n"
              "Repositorio: https://github.com/tsc221645/roadguard")
        fig.text(.06,.94,text,va="top",fontsize=11,linespacing=1.6)
        pages.savefig(fig);plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="0")
    parser.add_argument("--grid",type=int,default=8)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if not 2<=args.grid<=16 or args.batch<1: parser.error("grid 2..16; batch >=1")
    if not MODEL.is_file() or not IMAGES.is_dir() or not LABELS.is_dir(): parser.error("Faltan checkpoint o datos test")
    model=YOLO(str(MODEL))
    expected=yaml.safe_load((ROOT/"configs/classes.yaml").read_text(encoding="utf-8"))["names"]
    if model.names!=expected: raise ValueError("Clases de modelo/dataset distintas")
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    args.output.mkdir(parents=True,exist_ok=True)
    warm=image_at(next(IMAGES.glob("*.jpg")))
    original_signature=prediction_signature(predict(model,warm,args.device)[0])
    items,totals=inventory(model,args.device)
    local,global_sample=select(items)
    if args.smoke: global_sample=global_sample[:1]
    normal=[]
    for _ in range(3):
        sync(args.device);start=time.perf_counter();predict(model,warm,args.device);sync(args.device)
        normal.append(time.perf_counter()-start)
    completed=[];global_results=[]
    for name,selection,destination in (("local",local,completed),("global",global_sample,global_results)):
        for idx,item in enumerate(selection,1):
            image=image_at(ROOT/item["image_path"])
            xai=explain(model,image,item,args.device,args.grid,args.batch)
            stats=spatial_stats(np.asarray(xai["signed_importance"]),item,image.shape[1],image.shape[0])
            check=fidelity(model,image,item,np.asarray(xai["signed_importance"]),args.device) if name=="local" else None
            destination.append({**item,"xai":xai,"stats":stats,"fidelity":check})
            print(f"{name} {idx}/{len(selection)} {item['image']} {item['class_name']} {item['status']} {xai['seconds']:.2f}s",flush=True)
    after_signature=prediction_signature(predict(model,warm,args.device)[0])
    if original_signature!=after_signature:
        raise ValueError("La inferencia normal cambió tras XAI")
    bench={"normal_times_s":normal,"normal_median_s":statistics.median(normal),
           "five_s":sum(r["xai"]["seconds"] for r in completed),
           "global_s":sum(r["xai"]["seconds"] for r in global_results),
           "local_mean_s":statistics.mean(r["xai"]["seconds"] for r in completed),
           "gpu_peak_bytes":torch.cuda.max_memory_allocated() if args.device!="cpu" and torch.cuda.is_available() else None}
    report={"run_utc":datetime.now(timezone.utc).isoformat(),"seed":42,
            "checkpoint":str(MODEL.relative_to(ROOT)).replace("\\","/"),
            "checkpoint_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            "dataset":"RDD2022 Czech test", "inventory_counts":totals,
            "config":{"conf":.10,"imgsz":640,"nms_iou":.50,"matching_iou":.5,
                      "grid":args.grid,"replacement":"Gaussian blur 51x51","batch":args.batch,"device":args.device},
            "versions":{"python":platform.python_version(),"torch":torch.__version__,"ultralytics":ultralytics.__version__},
            "hardware":{"platform":platform.platform(),"gpu":torch.cuda.get_device_name(0) if args.device!="cpu" and torch.cuda.is_available() else None},
            "predictions_unchanged":True,
            "local":completed,"global":global_results,"benchmark":bench}
    (args.output/"results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not args.smoke:
        figures(completed,global_results,args.output)
        pdf(completed,global_results,bench,args.output)
    print(f"Resultados: {args.output}",flush=True)


if __name__=="__main__": main()
