#!/usr/bin/env python3
"""Comprueba la regla 2: cada capítulo tiene su nota de trabajo, y la nota no está vacía.

    verificar_notas.py ../../libro-ia-libro
    verificar_notas.py --selftest

La regla 2 dice que las cuentas y las comprobaciones de un capítulo se hacen ANTES de escribirlo,
en `notas/NOTAS-MATE-CAPnn.md`, y que sin nota no hay capítulo. Esa nota es la autoridad: es lo
que se verificó con las fuentes delante.

Existía la regla y no la comprobaba nadie. Faltaban dos notas —la del 3 y la del 8— y el libro
llevaba así desde el 16 de septiembre sin que nada lo dijera. Y sin nota, el fallo 4.18 (el
capítulo que desmiente su propia nota) no se puede auditar: no hay contra qué cotejar."""
import os, re, sys

# El prólogo, el epílogo y el apéndice no son capítulos con cuentas propias: el prólogo y el epílogo
# no tienen ni una cifra que no esté escrita con letra, y el apéndice describe las máquinas. No se
# les exige nota (el epílogo, desde el 26 de octubre de 2026).
SIN_NOTA = {'00-prologo.md', '98-epilogo.md', '99-apendice.md'}
MINIMO_UTIL = 400   # caracteres: una nota más corta que esto es un fichero vacío con cabecera


def numero_de_capitulo(fichero):
    """De «cap03-muchos.md» saca «03». Devuelve None si el fichero no es un capítulo numerado."""
    m = re.match(r'cap(\d+)', fichero)
    return m.group(1) if m else None


def revisar(capitulos, notas_existentes):
    """capitulos: lista de nombres de fichero, en el orden de Book.txt.
    notas_existentes: dict {'03': tamaño en caracteres}. Devuelve la lista de incidencias."""
    fallos = []
    for c in capitulos:
        if c in SIN_NOTA:
            continue
        n = numero_de_capitulo(c)
        if n is None:
            fallos.append((c, 'nombre', 'no se puede saber qué nota le toca'))
            continue
        if n not in notas_existentes:
            fallos.append((c, 'falta', f'NOTAS-MATE-CAP{n}.md no existe'))
        elif notas_existentes[n] < MINIMO_UTIL:
            fallos.append((c, 'vacía', f'NOTAS-MATE-CAP{n}.md tiene {notas_existentes[n]} caracteres'))
    return fallos


def selftest():
    """Tres pruebas sobre lo único que decide: si a un capítulo le falta su nota."""
    fallos = []

    # 1. TEST NULO — todo en su sitio: no puede saltar nada. Y el prólogo, el epílogo y el apéndice,
    #    que no llevan nota por diseño, tampoco pueden saltar: si saltaran, el verificador daría un
    #    FALLA permanente y acabaríamos desconectándolo.
    completo = revisar(['00-prologo.md', 'cap01-a.md', 'cap02-b.md', '98-epilogo.md', '99-apendice.md'],
                       {'01': 5000, '02': 5000})
    print(f"[1] test nulo         todo en su sitio: {len(completo)} incidencias; "
          f"el prólogo, el epílogo y el apéndice no piden nota")
    if completo:
        fallos.append(f"test nulo: salta con todo en su sitio: {completo}")

    # 2. SEÑAL IMPLANTADA — se quita una nota y se vacía otra. Tiene que cazar las dos, y cada
    #    una por su nombre, porque un fichero vacío engaña más que uno que no está.
    tocado = revisar(['cap01-a.md', 'cap02-b.md', 'cap03-c.md'], {'01': 5000, '02': 12})
    clases = sorted(f[1] for f in tocado)
    print(f"[2] señal implantada  una nota borrada y otra vacía: caza {clases}")
    if clases != ['falta', 'vacía']:
        fallos.append(f"señal implantada: tenía que cazar ['falta', 'vacía'] y cazó {clases}")

    # 3. INVARIANTE DEL DOMINIO — el verificador mira el número del capítulo, no su título ni su
    #    posición: el libro ya se renumeró una vez y los ficheros conservaron su nombre. Con los
    #    mismos capítulos en otro orden, y con el título cambiado, tiene que decir lo mismo.
    a = revisar(['cap01-a.md', 'cap03-c.md'], {'01': 5000})
    b = revisar(['cap03-titulo-nuevo.md', 'cap01-otro.md'], {'01': 5000})
    igual = sorted(f[1:] for f in a) == sorted(f[1:] for f in b)
    print(f"[3] invariante        el orden y el título no cambian el veredicto: "
          f"{'sí' if igual else 'NO'}")
    if not igual:
        fallos.append("invariante: el veredicto depende del orden o del título")

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
capitulos = [l.strip() for l in open(os.path.join(raiz, 'manuscript', 'Book.txt'),
                                     encoding='utf-8') if l.strip()]
notas = {}
for f in os.listdir(os.path.join(raiz, 'notas')):
    m = re.match(r'NOTAS-MATE-CAP(\d+)\.md$', f)
    if m:
        notas[m.group(1)] = len(open(os.path.join(raiz, 'notas', f), encoding='utf-8').read())

fallos = revisar(capitulos, notas)
if fallos:
    for c, clase, detalle in fallos:
        print(f"  {c:<38} {clase:<8} {detalle}")
    print(f"\nFALLA: {len(fallos)} capítulo(s) sin nota de trabajo utilizable (regla 2)")
    sys.exit(1)
print(f"PASA: los {len(capitulos) - len(SIN_NOTA & set(capitulos))} capítulos tienen su nota de trabajo.")
