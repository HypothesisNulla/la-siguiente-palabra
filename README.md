# Código compañero

Repositorio público con el código y los datos del libro. El libro cuenta qué salió;
aquí está cómo se obtuvo, para que cualquiera pueda repetirlo.

Máquina de referencia: MacBook Pro M4 Max, 36 GB de memoria unificada, macOS.

## El mapa

| carpeta | qué hay |
|:--|:--|
| `codigo/` | un fichero por experimento, sin rutas absolutas, con semilla fija; `medir_en_mac.sh`, que lanza las mediciones en el Mac; y `historico/`, los programas que ya no se usan |
| `datos/entrada/` | lo que entra en los programas: `quijote.txt`, `candidatos_es.txt`, `arc_25ff71a9.json`, `declaracion_universal/` y, si se descarga, `corpus_es/` (los trescientos libros, que no se publican) |
| `datos/salidas/` | lo que sale de ellos: las tablas guardadas (`.txt`), los `.csv`, `resultados_mac.txt`, y las carpetas de muestras y de pruebas. Cada tabla del libro se copia literalmente de una de estas salidas |
| `datos/README.md` | de dónde viene cada dato de entrada |
| `figuras/` | carpeta vacía al clonar: los programas de `codigo/` dibujan aquí las figuras al ejecutarse. Las figuras no se publican (el `.gitignore` las deja fuera) |
| `archivo/` | lo viejo que ya no se usa; nada se borra |
| `requirements.txt`, `requirements-mac-exacto.txt`, `LICENSE`, `.gitignore` | lo que exigen las herramientas |

**Rutas que cambiaron el 12 de noviembre de 2026 (Orden 158).** Las salidas guardadas y los informes anteriores a esa fecha
pueden citar las de antes.

| antes | ahora |
|:--|:--|
| `datos/quijote.txt`, `candidatos_es.txt`, `arc_25ff71a9.json`, `corpus_es/`, `declaracion_universal/` | `datos/entrada/` |
| los `.csv` sueltos en `codigo/` y en la raíz de `datos/`, y `codigo/resultados_mac.txt` | `datos/salidas/` |
| las copias viejas de seis `.csv` que estaban a la vez en `datos/` y en `codigo/` (`maquina_entera`, `perceptron`, `reparto_atencion`, `reparto_una_a_una`, `respuesta_modelo`, `retropropagacion`) | `archivo/datos-sueltos-viejos-2026-11-12/` |

## Dónde se ejecuta cada cosa

Las mediciones se dividen en dos clases y no se ejecutan en el mismo sitio.

**Independientes de la máquina.** Aciertos, recuentos, proporciones, vecinas más próximas,
comprobaciones de gradiente. Con la misma semilla dan el mismo resultado en cualquier ordenador,
así que da igual dónde se corran.

**Dependientes de la máquina.** Cualquier cifra que lleve un tiempo dentro: cuánto entrena un
modelo en X minutos, cuántos pasos da, cuánto tarda algo. Estas **solo** valen si se ejecutan en
el portátil del que habla el libro, un MacBook Pro M4 Max con 36 GB, y usando su GPU.

Para eso está `codigo/medir_en_mac.sh`, que hay que lanzar desde el Terminal de macOS —no desde
la máquina virtual Linux de la aplicación de escritorio, que no alcanza la GPU del Mac—:

    git clone https://github.com/HypothesisNulla/la-siguiente-palabra
    cd la-siguiente-palabra/codigo
    python3 -m pip install -r ../requirements.txt
    bash medir_en_mac.sh

Deja los resultados en `datos/salidas/resultados_mac.txt`, con la máquina y la versión de las bibliotecas
anotadas en la cabecera.

`requirements-mac-exacto.txt` es el `pip freeze` entero del entorno de las mediciones (un Mac con
Apple Silicon y Python 3.12.14): sirve para repetirlas con las mismas versiones de todo, no para el uso
normal, que es `requirements.txt`.

## Licencia

El código (`codigo/`), los resultados que produce (`datos/salidas/`) van bajo licencia MIT (`LICENSE`, texto estándar en
inglés). La licencia no cubre el texto del libro, cuyos derechos se reserva el autor y que no está
en este repositorio, ni los datos de terceros, que conservan sus propias condiciones:

- `datos/entrada/quijote.txt`: Don Quijote, del Proyecto Gutenberg, de dominio público.
- `datos/entrada/arc_25ff71a9.json`: una tarea del conjunto público de ARC-AGI
  (https://github.com/fchollet/ARC-AGI), bajo licencia Apache 2.0.
- Los trescientos libros de los capítulos 5 y siguientes no se incluyen: `codigo/descargar_corpus.py`
  los descarga del Proyecto Gutenberg.
