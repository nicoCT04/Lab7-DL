# Laboratorio #7 — NLP end-to-end y Embeddings

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Nicolás Concuá**

Pipeline completo **texto crudo → tokens → IDs → vectores → modelo → salida**:

1. **Preprocesamiento** de WikiText-103: normalización consistente con GloVe, tokenizador propio vs. NLTK, ley de Zipf,
   vocabulario con `min_count`, submuestreo y pares skip-gram.
2. **Skip-gram con negative sampling (SGNS) desde cero en PyTorch** (dos tablas `nn.Embedding`), 11 iteraciones de
   hiperparámetros sobre un subconjunto de 25 M tokens, y **Word2Vec de gensim** con la mejor configuración como referencia.
3. **Aritmética vectorial**: analogías propias (3CosAdd), benchmark `questions-words` por categoría (3CosAdd y 3CosMul)
   sobre un vocabulario compartido, paralelismo de vectores diferencia y t-SNE.
4. **Clasificación de AG News**: TF-IDF + regresión logística vs. `nn.EmbeddingBag` + MLP con embeddings aleatorios,
   SGNS, gensim y GloVe (congelados y con fine-tuning), y experimento con fracciones de datos (1 %–100 %).
5. Tabla comparativa SGNS / gensim / GloVe y discusión.

## Resultados principales

| | SGNS propio | Word2Vec gensim | GloVe-100 |
|---|---|---|---|
| Tokens de entrenamiento | 25 M | 25 M | 6,000 M |
| Analogías (3CosAdd, vocab. compartido) | 43.3 % | 43.0 % | 66.0 % |
| WordSim-353 / SimLex-999 (ρ) | 0.654 / 0.302 | 0.648 / 0.306 | 0.533 / 0.298 |
| F1 test AG News (fine-tuning) | 91.5 % | 91.5 % | 92.0 % |

Baseline TF-IDF + regresión logística: F1 test 92.2 %.

## Estructura

```
Lab7-DL/
├── notebook/
│   └── lab7.ipynb        # notebook completo y ejecutado (secciones 0–8 del enunciado)
├── reports/              # figuras (.png) y tablas (.csv) generadas por el notebook; informe en PDF
├── vectores/
│   └── sgns_mejor.kv     # vectores del mejor SGNS (gensim KeyedVectors, 90 k palabras × 100 d)
└── data/                 # cachés de entrenamiento (no se versiona)
```

Cargar los vectores:

```python
from gensim.models import KeyedVectors
kv = KeyedVectors.load('vectores/sgns_mejor.kv')
kv.most_similar(positive=['king', 'woman'], negative=['man'])
```

## Cómo reproducir

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install torch gensim datasets nltk scikit-learn scipy pandas matplotlib jupyter
cd notebook
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 lab7.ipynb
```

Los datos (WikiText-103, AG News, GloVe) se descargan automáticamente. Los entrenamientos de SGNS y gensim se guardan
en `data/` y se reutilizan al re-ejecutar; `LAB7_RETRAIN=1` fuerza a reentrenar y `LAB7_QUICK=1` corre una prueba
rápida con un corpus pequeño. El notebook detecta CUDA, MPS (Apple Silicon) o CPU. Hardware usado: Apple M4 Pro (MPS).
