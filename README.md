# 🧬 BioHackathon — Cancer Image Classification with Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.x-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AlphaFold](https://img.shields.io/badge/AlphaFold-Protein_Modeling-blue?style=flat)](https://github.com/google-deepmind/alphafold)

Este proyecto fue desarrollado en el marco de una **BioHackathon**. El desafío principal consistió en diseñar e implementar un modelo de *Deep Learning* para el reconocimiento y clasificación de imágenes histopatológicas, integrando además el análisis estructural de proteínas y consideraciones bioéticas.

---

## 📌 Contexto del Proyecto

El sistema está diseñado para realizar un análisis multidimensional a partir de una sola muestra médica, resolviendo cuatro tareas clave:
1. **Detección:** Determinar la presencia o ausencia de células cancerígenas.
2. **Nivel de Magnificación:** Identificar el zoom óptico de la muestra.
3. **Clasificación del Tipo:** Clasificar el tumor en Benigno o Maligno.
4. **Identificación de Tejido:** Determinar el tipo de tejido de la muestra.

Además del modelado de Visión por Computadora (CV), el equipo profundizó en el contexto biológico detrás del análisis histopatológico y abordó las implicaciones bioéticas del manejo de datos clínicos.

---

## 📊 Dataset

El modelo fue entrenado utilizando el **BreakHis (Breast Cancer Histopathological Database)**.
* **Origen:** Disponible en [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis).
* **Características:** Imágenes de tumores de mama en cuatro niveles de magnificación (**40x, 100x, 200x, 400x**).

### Estructura de Clases:
| Condición | Tipos Histológicos Incluidos |
| :--- | :--- |
| **Benigno** | Adenosis, Fibroadenoma, Tumor Phyllodes, Adenoma Tubular |
| **Maligno** | Carcinoma Ductal, Carcinoma Lobulillar, Carcinoma Mucinoso, Carcinoma Papilar |

---

## 🧬 Modelado de Estructura de Proteínas (AlphaFold)

Para enriquecer el contexto biológico del proyecto, la carpeta `PDB/` incluye archivos `.pdb` generados mediante **AlphaFold** en Google Colab:
* **Estructura Nativa:** La forma tridimensional normal de la proteína de interés.
* **Variante Mutada:** La variante asociada al fenotipo cancerígeno analizado.

> 💡 **Nota de Visualización:** Estos archivos se pueden abrir y analizar con herramientas especializadas como **UCSF ChimeraX** o **iCn3D**.

---

## 📂 Estructura del Repositorio

```text
Hackathon/
├── app/
│   ├── main.py           # Aplicación principal en FastAPI
│   ├── predictor.py      # Lógica de inferencia del modelo de DL
│   └── static/           # Archivos estáticos del Frontend (HTML/CSS/JS)
├── PDB/                  # Estructuras de proteínas generadas con AlphaFold
├── outputs/              # Checkpoints y pesos del modelo entrenado
├── train.ipynb           # Notebook de Jupyter con el proceso de entrenamiento
└── requirements.txt      # Dependencias del proyecto
