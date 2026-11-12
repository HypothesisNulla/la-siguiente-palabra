#!/bin/bash
# Ejecuta las mediciones del libro EN macOS, usando la GPU del Mac (Metal/MPS).
#
# Por qué existe este guion: las mediciones se prepararon en un contenedor Linux sin GPU, y
# el shell que la aplicación de escritorio expone también es una máquina virtual Linux, que
# no alcanza el Metal del Mac. Cualquier cifra del libro que lleve un TIEMPO dentro tiene que
# salir de aquí, ejecutado por Carlos en su propio Terminal de macOS.
#
# Las cifras sin tiempo (aciertos, recuentos, proporciones, vecinas más próximas) tampoco son
# siempre iguales en otra máquina: en los capítulos 5 y 7 cambian puestos y decimales entre un
# procesador x86 y uno ARM. Por eso, desde el 9 de octubre de 2026, todas las salidas del libro
# se generan en el Mac del autor (Apple Silicon); en otra máquina pueden salir distintas en la
# última cifra.
#
# Uso, desde Terminal de macOS (no desde la aplicación):
#     cd la-siguiente-palabra/codigo     (la carpeta donde hayas clonado el repositorio)
#     bash medir_en_mac.sh
#
# Deja la salida en ../datos/salidas/resultados_mac.txt, que es lo que hay que pasarme.

set -e
cd "$(dirname "$0")"

ENTORNO="$HOME/.venvs/libro-ia"
if [ ! -d "$ENTORNO" ]; then
  echo "Creando entorno en $ENTORNO ..."
  python3 -m venv "$ENTORNO"
fi
# shellcheck disable=SC1091
source "$ENTORNO/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch numpy scikit-learn gensim

SALIDA="../datos/salidas/resultados_mac.txt"
{
  echo "=============================================="
  echo "Mediciones del libro, ejecutadas en macOS"
  echo "fecha: $(date '+%Y-%m-%d %H:%M')"
  echo "máquina: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
  echo "memoria: $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
  python - <<'PY'
import torch, platform
print("torch:", torch.__version__, "| python:", platform.python_version())
print("GPU del Mac (Metal) disponible:", torch.backends.mps.is_available())
PY
  echo "=============================================="
  echo
  echo "########## capítulo 6: memoria recurrente ##########"
  MPS=1 python memoria_recurrente.py

  echo
  echo "########## banco de copia (cap 1, 6, 13): ¿lo había visto ya? ##########"
  MPS=1 python lo_habia_visto_ya.py --selftest
  MPS=1 python escribir_muestras.py --semilla 20260914 --tirada 1
  MPS=1 python escribir_muestras.py --semilla 20260914 --tirada 2
  MPS=1 python escribir_muestras.py --semilla 20260915 --tirada 1
  MPS=1 python escribir_muestras.py --semilla 20260915 --tirada 2
  MPS=1 python escribir_muestras.py --semilla 20260916 --tirada 1
  MPS=1 python escribir_muestras.py --semilla 20260916 --tirada 2
  MPS=1 python lo_habia_visto_ya.py > ../datos/salidas/lo_habia_visto_ya.txt
} 2>&1 | tee "$SALIDA"

echo
echo "Listo. Pásame el fichero: $(pwd)/$SALIDA"
