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
