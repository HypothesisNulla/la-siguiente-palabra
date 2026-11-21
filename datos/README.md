# Datos

Lo que entra en los programas está en `entrada/`; lo que sale de ellos, en `salidas/`. Las rutas, en este fichero, son las de `entrada/`.

- `entrada/quijote.txt` — Don Quijote, Proyecto Gutenberg (dominio público).
- `entrada/candidatos_es.txt` — lista de los libros en español usados en el capítulo 5: número de
  Gutenberg, título y autor, uno por línea. Los textos NO se incluyen aquí (138 MB); el guion
  `codigo/descargar_corpus.py` los baja de nuevo, a `entrada/corpus_es/`, a partir de esta lista.
- `entrada/arc_25ff71a9.json` — un ejercicio de la prueba ARC de François Chollet (tarea `25ff71a9` del
  conjunto público de entrenamiento, https://github.com/fchollet/ARC-AGI, licencia Apache 2.0).
  Lo dibuja `codigo/figura_prueba_de_chollet.py` para el capítulo 14.
- `entrada/declaracion_universal/` — la Declaración Universal de Derechos Humanos entera en nueve idiomas
  (`spa`, `eng`, `fra`, `cat`, `deu`, `eus`, `rus`, `ell_monotonic`, `hin`), sin tocar. Copia del
  corpus `udhr2` de NLTK, que recoge las traducciones del proyecto «UDHR in Unicode»; bajado el
  10 de octubre de 2026 de
  https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr2.zip
  (sha256 del zip `0796c314b09a930c989c6f9d93d226af9af13feccd88496e196c743dd266c7f3`). Las
  cuenta `codigo/idiomas_en_trozos.py` para el capítulo 7.
- `entrada/oasst1/` — OpenAssistant, `oasst1`: conversaciones escritas a mano por voluntarios en 2023, con varias respuestas a la misma
  petición ordenadas por personas (https://huggingface.co/datasets/OpenAssistant/oasst1, licencia Apache 2.0). El fichero es
  `2023-04-12_oasst_ready.trees.jsonl.gz` (md5 `bd119c17938362c3ea25bc1523e99551`, bajado el 15 de noviembre de 2026). No se publica
  (34 MB de textos de terceros): `codigo/ejemplos_de_adiestramiento.py` lo baja de nuevo y comprueba el md5.
- `entrada/qwen25-32b-instruct-model.safetensors.index.json` — el índice oficial (`model.safetensors.index.json`) de `Qwen/Qwen2.5-32B-Instruct` en Hugging Face, guardado el 21 nov 2026 (Orden 187): su `metadata.total_size` (65.527.752.704 bytes) es lo que ocupa esa copia sin comprimir. Lo lee `codigo/comprimir_cinco_numeros.py` (la copia entera no está descargada).
