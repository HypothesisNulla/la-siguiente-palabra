#!/usr/bin/env python3
"""Comprueba que el catálogo de fallos de AUDITORIA.md está al día.

    verificar_catalogo.py ../../libro-ia-libro
    verificar_catalogo.py --selftest

Cada auditor encuentra fallos nuevos y los escribe en SU informe. El catálogo del apartado 4 de
`notas/control/AUDITORIA.md` es lo que lee el auditor siguiente, y es lo único que lee: si el fallo no
está ahí, el que venga detrás no lo busca.

Se ha quedado atrás **tres veces** en una semana. Las tres se arreglaron a mano, y las tres
volvieron. Un fallo de proceso que se arregla a mano vuelve; por eso esto existe.

Comprueba además que el número de verificadores que el procedimiento manda pasar coincide con los
que hay en el directorio, porque eso también se quedó atrás dos veces."""
import os, re, sys

# Los informes viven aquí, uno por capítulo, y ahí es donde nacen los fallos nuevos.
CARPETA_INFORMES = os.path.join('notas', 'control', 'auditoria')
NO_SON_INFORMES = {'README.md', 'PENDIENTES.md'}
CABECERA = re.compile(r'^#{2,3}\s*(4\.\d+)\b', re.M)


def numeros(texto):
    """Los «4.N» que ese texto declara como entradas de catálogo, con encabezado propio."""
    return {m.group(1) for m in CABECERA.finditer(texto)}


def revisar(catalogo, informes):
    """catalogo: el texto de AUDITORIA.md. informes: {nombre: texto}.
    Devuelve la lista de (fallo, dónde se describió) que el catálogo no recoge."""
    tiene = numeros(catalogo)
    faltan = []
    for nombre, texto in sorted(informes.items()):
        for n in sorted(numeros(texto), key=lambda x: int(x.split('.')[1])):
            if n not in tiene:
                faltan.append((n, nombre))
    return faltan


def selftest():
    """Tres pruebas sobre lo único que decide: si un fallo descrito en un informe está también
    en el catálogo."""
    fallos = []
    catalogo = "## 4. Catálogo\n### 4.1 Uno\n### 4.2 Otro\n"

    # 1. TEST NULO — un informe que no describe ningún fallo nuevo, y otro que describe uno que
    #    ya está. No puede faltar nada. Si aquí saltara algo, el verificador estaría pidiendo
    #    añadir al catálogo cosas que ya están, y acabaríamos desconectándolo.
    sale = revisar(catalogo, {"cap01.md": "texto sin fallos nuevos, cita el 4.1 de pasada",
                              "cap02.md": "### 4.2 Otro\ncomo ya está, no falta"})
    print(f"[1] test nulo         informes sin nada nuevo: {len(sale)} faltan"
          + (f" ({sale})" if sale else ""))
    if sale:
        fallos.append(f"test nulo: pide añadir algo que ya está: {sale}")

    # 2. SEÑAL IMPLANTADA — un informe describe el 4.3 y el catálogo no lo tiene. Tiene que
    #    cazarlo, y decir en qué informe está para poder copiarlo de ahí.
    sale = revisar(catalogo, {"cap06.md": "### 4.3 La muestra que el capítulo eligió\nlargo"})
    print(f"[2] señal implantada  un fallo nuevo sin catalogar: {sale}")
    if sale != [("4.3", "cap06.md")]:
        fallos.append(f"señal implantada: esperaba [('4.3', 'cap06.md')] y salió {sale}")

    # 3. INVARIANTE DEL DOMINIO — una MENCIÓN no es una descripción. Los informes citan fallos
    #    del catálogo todo el rato («esto es el 4.18 otra vez»); solo cuenta lo que lleva
    #    encabezado propio, que es como se escribe una entrada nueva. Si contara las menciones,
    #    el verificador suspendería siempre y no serviría de nada.
    sale = revisar(catalogo, {"cap04.md": "es el 4.9 otra vez, y roza el 4.17 y el 4.20"})
    print(f"[3] invariante        menciones sueltas de fallos no catalogados: {len(sale)} "
          f"(tienen que ser 0: mencionar no es describir)")
    if sale:
        fallos.append(f"invariante: cuenta menciones como si fueran entradas: {sale}")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


if '--selftest' in sys.argv:
    sys.exit(selftest())

raiz = sys.argv[1]
catalogo = open(os.path.join(raiz, 'notas', 'control', 'AUDITORIA.md'), encoding='utf-8').read()
carpeta = os.path.join(raiz, CARPETA_INFORMES)
informes = {f: open(os.path.join(carpeta, f), encoding='utf-8').read()
            for f in sorted(os.listdir(carpeta))
            if f.endswith('.md') and f not in NO_SON_INFORMES}

faltan = revisar(catalogo, informes)

# Y el recuento de verificadores, que también se quedó atrás dos veces.
aqui = len([f for f in os.listdir(os.path.dirname(os.path.abspath(__file__)))
            if f.startswith('verificar_') and f.endswith('.py')])
LETRAS = {7: 'siete', 8: 'ocho', 9: 'nueve', 10: 'diez', 11: 'once', 12: 'doce', 13: 'trece',
          14: 'catorce', 15: 'quince'}
dice = re.search(r'Pasar los (\w+) verificadores', catalogo)
descuadre = dice and LETRAS.get(aqui) != dice.group(1)

for n, donde in faltan:
    print(f"  {n:<6} descrito en notas/control/auditoria/{donde} y NO está en el catálogo")
if descuadre:
    print(f"  el procedimiento manda pasar «{dice.group(1)}» verificadores y hay {aqui} "
          f"({LETRAS.get(aqui, aqui)})")

if faltan or descuadre:
    print(f"\nFALLA: el catálogo de AUDITORIA.md no está al día.")
    print("El auditor siguiente solo lee el catálogo: lo que no esté ahí, no lo va a buscar.")
    sys.exit(1)
print(f"PASA: los {len(numeros(catalogo))} fallos del catálogo cubren los {len(informes)} "
      f"informes, y el procedimiento cuenta bien los {aqui} verificadores.")
