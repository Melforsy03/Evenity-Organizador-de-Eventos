import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from modules.embedding import EventEmbedder

# ======================= MÉTRICAS =======================
def coincide(predicho: str, esperados: list) -> bool:
    pred_norm = predicho.strip().lower()
    return any(ev.strip().lower() in pred_norm or pred_norm in ev.strip().lower() for ev in esperados)

def dcg(scores):
    return sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(scores))

def ndcg(predicted_ids, ideal_ids, k=10):
    ideal_relevance = {eid: 3 for eid in ideal_ids}
    predicted_scores = []
    for ev in predicted_ids[:k]:
        match = next((eid for eid in ideal_ids if coincide(ev, [eid])), None)
        predicted_scores.append(ideal_relevance.get(match, 0) if match else 0)
    ideal_sorted = sorted(ideal_relevance.values(), reverse=True)[:k]
    return dcg(predicted_scores) / dcg(ideal_sorted) if ideal_sorted else 0.0

def precision_recall(predicted_ids, ideal_ids, k=10):
    pred_set = set(predicted_ids[:k])
    ideal_set = set(ideal_ids)
    matches = sum(1 for p in pred_set if coincide(p, ideal_ids))
    precision = matches / k
    recall = matches / len(ideal_set) if ideal_set else 0
    return precision, recall

# ======================= EVALUACIÓN =======================
def evaluar_modelo(model_name, queries, eventos, tipo, k=10):
    print(f"\nEvaluando modelo: {model_name} con queries tipo: {tipo}")
    embedder = EventEmbedder(model_name)

    print(f"[DEBUG] Eventos cargados: {len(eventos)}")
    embeddings = embedder.generate_embeddings(eventos)

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("❌ No se generaron embeddings. Revisa si hay eventos cargados.")

    embedder.build_index(embeddings)

    resultados_modelo = []
    for query in queries:
        resultados, _ = embedder.filtered_search(
            query["query"],
            city=query.get("ciudad"),
            max_km=None,
            user_coords=None,
            k=k
        )
        predicted_ids = [ev.get("basic_info", {}).get("title", "") for ev in resultados]

        ndcg_score = ndcg(predicted_ids, query["esperados"], k)
        precision, recall = precision_recall(predicted_ids, query["esperados"], k)

        resultados_modelo.append({
            "query": query["query"],
            "tipo": tipo,
            "ndcg": ndcg_score,
            "precision": precision,
            "recall": recall,
            "predicted": predicted_ids,
            "esperados": query["esperados"]
        })

    return resultados_modelo

# ======================= MAIN =======================
def ejecutar_evaluacion():
    query_files = {
        "estandar": "queries_sinteticas_formato_extendido.json",
        "naturales": "queries_llm_naturales.json",
        "con_fecha": "queries_con_fecha_llm.json"
    }

    with open("eventos_sinteticos_formato_extendido.json", "r", encoding="utf-8") as f:
        eventos_raw = json.load(f)

    eventos = []
    for ev in eventos_raw:
        eventos.append({
            "basic_info": {"title": ev["basic_info"]["title"]},
            "spatial_info": {"area": {"city": ev["spatial_info"]["area"].get("city", "")}},
            "classification": {"primary_category": ev["classification"]["primary_category"]},
            "temporal_info": {"start_time": ev["temporal_info"]["start"]},
            "participants": {
                "performers": ev.get("participants", {}).get("performers", [])
            }
        })

    modelos = [
        
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "paraphrase-MiniLM-L3-v2",
        "all-mpnet-base-v2",
        "distiluse-base-multilingual-cased-v1",
        "sentence-transformers/LaBSE",
        "sentence-transformers/msmarco-distilbert-base-v3",
        "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        "paraphrase-multilingual-mpnet-base-v2",
    ]

    resumen = {}

    for tipo, archivo in query_files.items():
        with open(archivo, "r", encoding="utf-8") as f:
            queries = json.load(f)

        for modelo in modelos:
            clave = f"{modelo}::{tipo}"
            resultados = evaluar_modelo(modelo, queries, eventos, tipo, k=10)
            resumen[clave] = resultados

    with open("evaluacion_completa.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("\n✅ Evaluación completa. Resultados en evaluacion_completa.json")

def graficos_tablas():
    with open("evaluacion_completa.json", "r", encoding="utf-8") as f:  
        data = json.load(f)

    rows = []
    for modelo_tipo, resultados in data.items():
        modelo, tipo = modelo_tipo.split("::")
        for r in resultados:
            rows.append({
                "Modelo": modelo,
                "TipoQuery": tipo,
                "Query": r["query"],
                "NDCG@10": r["ndcg"],
                "Precision@10": r["precision"],
                "Recall@10": r["recall"]
            })

    df = pd.DataFrame(rows)
    df_grouped = df.groupby(["Modelo", "TipoQuery"])[["NDCG@10", "Precision@10", "Recall@10"]].mean().reset_index()
    print("\n📊 Métricas Promedio por Modelo y Tipo de Query:\n", df_grouped)

    os.makedirs("graficos", exist_ok=True)

    for metric in ["NDCG@10", "Precision@10", "Recall@10"]:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df_grouped, x="TipoQuery", y=metric, hue="Modelo")
        plt.title(f"Comparación de {metric} por Tipo de Query y Modelo", fontsize=14)
        plt.ylabel(f"{metric} Promedio")
        plt.xlabel("Tipo de Query")
        plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.tight_layout()
        plt.savefig(f"graficos/{metric}_comparacion_por_tipo_query.png")
        plt.close()

if __name__ == "__main__":
    ejecutar_evaluacion()
    graficos_tablas()
