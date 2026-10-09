# Traveler Insights UPTC 2026 — Team BETO

Proyecto académico de clasificación de sentimiento de reseñas turísticas en tres clases: `negativo`, `neutral` y `positivo`. El desarrollo sigue CRISP-ML(Q) y compara un baseline TF-IDF + regresión logística con BETO y XLM-R ajustados por el equipo.

## Archivos principales

- `team-beto.ipynb`: cuaderno completo para ejecutar y entregar en Kaggle.
- `Data/train.csv`: conjunto de desarrollo etiquetado (818 filas).
- `Data/test.csv`: conjunto local etiquetado reservado para evaluación final (351 filas).
- `Data/submission.csv`: conjunto sin etiqueta para generar el envío (129 filas).
- `proyecto_final.ipynb`: plantilla inicial conservada como referencia.

## Ejecución en Kaggle

1. Abrir `team-beto.ipynb` en Kaggle.
2. Adjuntar los datos de la competición `traveler-insights-uptc-2026`.
3. Activar un acelerador GPU T4.
4. Activar Internet para descargar los checkpoints de Hugging Face. Si Internet no está permitido durante la ejecución evaluada, descargar previamente los modelos como recursos de Kaggle y adaptar sus rutas locales.
5. Ejecutar todo desde un kernel limpio.
6. Completar la categorización manual de al menos 20 errores y los aportes individuales.
7. Revisar `/kaggle/working/submission_kaggle.csv` y guardar una versión del notebook.

El cuaderno genera además el registro de experimentos, la bitácora de IA, el análisis de errores, el modelo final y un archivo alternativo con etiquetas textuales.

## Hallazgos de calidad importantes

- La clase neutral representa cerca del 78 % de `train.csv`; por eso se reportan F1 macro y recall de la clase negativa además de accuracy.
- Hay reseñas repetidas entre los archivos. El notebook las excluye del entrenamiento cuando reaparecen en test o submission y agrupa duplicados internos para evitar fuga entre train y validación.
- `Sitio` junto con `Valoración_num` determina la etiqueta en todos los datos etiquetados. Estas variables se auditan, pero se excluyen del modelo principal para evitar un atajo conceptual y evaluar realmente el texto.
- La documentación de Kaggle es contradictoria sobre el rol de `test.csv` y el formato de la etiqueta. El cuaderno usa el contenido real de los archivos y produce tanto el envío numérico como una alternativa textual.

## Reproducibilidad

La semilla global es 42. Las versiones efectivamente usadas, el hardware, la configuración y las métricas de cada corrida se guardan como artefactos durante la ejecución.
