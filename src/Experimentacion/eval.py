import json
import csv
from sklearn.metrics import precision_score, recall_score
import numpy as np
import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.embedding import EventEmbedder

def evaluar_query(embedder, query_data, k=5):
    query = query_data["query"]
    ciudad = query_data.get("ciudad")
    esperados = query_data["esperados"]

    resultados, _ = embedder.filtered_search(query=query, city=ciudad, k=k)

    predichos = [ev["basic_info"]["title"] for ev in resultados]
    y_true = [1 if title in esperados else 0 for title in predichos]
    y_pred = [1] * len(predichos)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "query": query,
        "ciudad": ciudad if ciudad else "-",
        "precision@k": round(precision, 3),
        "recall@k": round(recall, 3),
        "esperados": esperados,
        "devueltos": predichos
    }

def evaluar_caso(nombre, json_file, csv_file):
    print(f"\n=== Evaluando: {nombre} ===")
    embedder = EventEmbedder.load("embedding_data")

    with open(json_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    resultados = []
    for q in queries:
        res = evaluar_query(embedder, q, k=5)
        resultados.append(res)

    # Mostrar resumen en consola
    for r in resultados:
        print(f"🔍 Query: {r['query']}")
        print(f"   🎯 Precision@5: {r['precision@k']} | Recall@5: {r['recall@k']}")
        print(f"   ✅ Esperados: {r['esperados']}")
        print(f"   📦 Devueltos: {r['devueltos']}")
        print("---")

    avg_precision = np.mean([r["precision@k"] for r in resultados])
    avg_recall = np.mean([r["recall@k"] for r in resultados])
    print(f"📊 Promedio Precision@5: {avg_precision:.3f}")
    print(f"📊 Promedio Recall@5: {avg_recall:.3f}")

    # Guardar CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "ciudad", "precision@k", "recall@k", "esperados", "devueltos"])
        writer.writeheader()
        for r in resultados:
            writer.writerow({
                "query": r["query"],
                "ciudad": r["ciudad"],
                "precision@k": r["precision@k"],
                "recall@k": r["recall@k"],
                "esperados": "; ".join(r["esperados"]),
                "devueltos": "; ".join(r["devueltos"])
            })

    print(f"✅ Resultados guardados en {csv_file}")

def main():
    escenarios = [
        ("Estructurado", "queries_gold.json", "resultados_estructurado.csv"),
        ("Lenguaje Natural", "queries_gold_natural.json", "resultados_natural.csv"),
        ("Mixto", "queries_gold_mixtas.json", "resultados_mixto.csv")
    ]

    for nombre, json_path, csv_path in escenarios:
        evaluar_caso(nombre, json_path, csv_path)

if __name__ == "__main__":
    main()
