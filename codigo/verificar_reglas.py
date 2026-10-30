#!/usr/bin/env python3
"""Vigila la regla 1 del libro: ni notación, ni jerga sin traducir, ni código en el cuerpo.

    verificar_reglas.py capitulo.md
    verificar_reglas.py --selftest

Es el verificador que sostiene la promesa del prólogo —«en este libro no hay ni una sola
fórmula»—, así que se comprueba a sí mismo como cualquier otra medición del libro."""
import os, re, sys, unicodedata

# Jerga que no entra en el cuerpo por ningún motivo. Los términos que el libro SÍ se
# apropia —red neuronal, neurona, capa, pesos— no se vigilan aquí: se declaran en
# notas/VOCABULARIO.md y los vigila verificar_vocabulario.py, que comprueba algo más
# exigente que su ausencia: que se bauticen antes de usarse (reglas 5 bis y 5 ter).
PROHIBIDAS = ["softmax","vector","matriz","matrices","gradiente","token","embedding",
    "query","key","value","parametro","hiperparametro","logaritm","dimension",
    "producto escalar","funcion de perdida","backprop","tensor","capa oculta",
    "n-grama","bigrama","trigrama"]
GRIEGO = re.compile(r'[Ͱ-Ͽ]')
MATE   = re.compile(r'[=∑∏√∫±×·⋅≈≤≥^]|\b\d+\s*[*/]\s*\d+')
CODIGO = re.compile(r'^\s*(```|import |def |>>> )')
# Envoltorio de imprenta para los trozos que la fuente del libro no dibuja (chino, tailandés,
# árabe). Es formato, no notación: se quita antes de buscar.
MARCADO = re.compile(r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}')


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s.lower())
                   if unicodedata.category(c) != 'Mn')


# Lo que escribe un modelo, letra por letra, dentro de un bloque «::: muestra» (los renglones con
# sangría de cuatro espacios) no es notación del libro: es salida de máquina, que la regla 6
# prohíbe editar y que verificar_muestras coteja con su salida. Desde el capítulo 14 (10 de
# octubre de 2026) esas muestras llevan lo que el modelo escribe de verdad —una orden con llaves
# y un asterisco, una cuenta con su «=»— y eso es justo lo que el capítulo enseña; la prosa lo
# explica símbolo a símbolo. Se exentan solo esos renglones: el título y las notas de la muestra,
# y toda la prosa, se siguen vigilando igual.
SANGRIA_MUESTRA = '    '

# El puente (regla 5 septies, 30 de octubre de 2026): «token» y «parámetro» siguen prohibidos, pero
# el libro los dice UNA vez, en la línea de su puente («a estos trozos, en la prensa y en
# inglés, se les llama *tokens*»), y pueden aparecer dentro de un bloque «::: tabla» o
# «::: muestra» (salida de máquina o texto que el programa imprime). En cualquier otro sitio
# del cuerpo siguen suspendiendo. El apartado «Puentes» de notas/VOCABULARIO.md dice cuál es
# la línea de cada puente (la frase ancla); verificar_vocabulario.py vigila además que
# ninguna palabra de un puente vaya antes que él.
LIBRES_EN_UN_PUENTE = ('token', 'parametro')
APARTADO_PUENTES = '## Puentes'


def puentes_de(ruta):
    """Las frases ancla de los puentes declarados para este fichero, o [] si no hay
    notas/VOCABULARIO.md por encima de él (se sube desde su carpeta hasta encontrarlo)."""
    d = os.path.dirname(os.path.abspath(ruta))
    while not os.path.exists(os.path.join(d, 'notas', 'VOCABULARIO.md')):
        if os.path.dirname(d) == d:
            return []
        d = os.path.dirname(d)
    nombre = os.path.basename(ruta)
    anclas, dentro = [], False
    for linea in open(os.path.join(d, 'notas', 'VOCABULARIO.md'), encoding='utf-8'):
        if linea.startswith('## '):
            dentro = linea.strip() == APARTADO_PUENTES
            continue
        if not dentro or linea.count('|') != 4 or linea[:1] in ('#', ' ', '\t'):
            continue
        fichero, ancla = [x.strip() for x in linea.split('|')][3:5]
        if fichero == nombre:
            anclas.append(ancla)
    return anclas


def revisar(lineas, es_el_mapa=False, anclas=()):
    """Devuelve la lista de incidencias: (número de línea, clase, término, trozo).
    `anclas`: las frases de los puentes de este fichero (puentes_de)."""
    fallos = []
    en_muestra = False
    en_bloque = False        # dentro de un «::: tabla» o un «::: muestra», título y notas incluidos
    for i, l in enumerate(lineas, 1):
        if l.strip() in ('::: tabla', '::: muestra'):
            en_bloque = True
        elif en_bloque and l.strip() == ':::':
            en_bloque = False
        if l.strip() == '::: muestra':
            en_muestra = True
            continue
        if en_muestra and l.strip() == ':::':
            en_muestra = False
            continue
        if en_muestra and l.startswith(SANGRIA_MUESTRA):
            continue
        if l.strip().startswith('*[') or l.strip().startswith('[NO EJECUTADO'):
            continue
        l = MARCADO.sub(r'\1', l)
        n = norm(l)
        if not es_el_mapa:
            for p in PROHIBIDAS:
                if p in n:
                    if p in LIBRES_EN_UN_PUENTE and (en_bloque or any(a in l for a in anclas)):
                        continue
                    fallos.append((i, 'jerga', p, l.strip()[:70]))
        if GRIEGO.search(l): fallos.append((i, 'griego', '', l.strip()[:70]))
        if MATE.search(l):   fallos.append((i, 'notacion', '', l.strip()[:70]))
        if CODIGO.match(l):  fallos.append((i, 'codigo', '', l.strip()[:70]))
    return fallos


def clases(fallos):
    return sorted({f[1] for f in fallos})


def selftest():
    """Tres pruebas sobre lo único que este verificador decide: si una línea del libro lleva
    notación, jerga o código."""
    fallos = []

    # 1. TEST NULO — prosa limpia del libro, con cifras y con rayas dentro. Si aquí saltara
    #    algo, el verificador estaría marcando texto bueno y acabaríamos desactivándolo.
    limpio = ["En 1948, un ingeniero de la compañía telefónica Bell publicó un artículo.",
              "Aquel ZX81 tenía un kilobyte: mil veinticuatro bytes, ni uno más.",
              "Donde el suelo quede cerca —son cuatro sitios— te lo diré.",
              "La máquina acierta el 90,4 % de las veces, frente al 78,1 % de antes.",
              "    python ngrama.py --selftest"]   # el NOMBRE del programa no es jerga
    sale = revisar(limpio)
    print(f"[1] test nulo         prosa limpia: {len(sale)} incidencias"
          + (f" ({clases(sale)})" if sale else ""))
    if sale:
        fallos.append(f"test nulo: marca prosa buena: {sale[0]}")

    # 2. SEÑAL IMPLANTADA — una línea mala de cada clase, de una en una. Tiene que cazar las
    #    cuatro, y cada una por su nombre, no cuatro veces la misma.
    implantadas = {'jerga':    "El modelo guarda cada palabra como un vector de números.",
                   'griego':   "Se ajusta con un paso de tamaño α en cada vuelta.",
                   'notacion': "El resultado es 3 × 4 ≈ 12, redondeando.",
                   'codigo':   "import numpy as np"}
    implantadas['jerga2'] = "Eso es lo que hace un modelo de n-gramas, ni más ni menos."
    implantadas['jerga'], implantadas['jerga2'] = implantadas['jerga'], implantadas['jerga2']
    cazadas = {}
    for clase, linea in implantadas.items():
        esperada = 'jerga' if clase.startswith('jerga') else clase
        cazadas[clase] = esperada in clases(revisar([linea]))
    print("[2] señal implantada  " + "; ".join(
        f"{c}: {'la caza' if v else 'SE LE ESCAPA'}" for c, v in cazadas.items()))
    if not all(cazadas.values()):
        fallos.append("señal implantada: se le escapa "
                      + ", ".join(c for c, v in cazadas.items() if not v))

    # 3. INVARIANTE DEL DOMINIO — dos cosas que tienen que seguir siendo verdad pase lo que
    #    pase. Primera: en el mapa de los sótanos se levanta la prohibición de jerga Y NADA
    #    MÁS; una fórmula ahí sigue siendo una fórmula. Segunda: el envoltorio de imprenta
    #    que dibuja el chino es formato, no notación, y no puede contar como incidencia.
    en_el_mapa = revisar([implantadas['jerga'], implantadas['notacion']], es_el_mapa=True)
    solo_notacion = clases(en_el_mapa) == ['notacion']
    envuelto = revisar(["La máquina escribió `\\textcjk{下}`{=latex} y siguió."])
    # Tercera: lo que escribe la máquina dentro de una muestra no cuenta, pero lo mismo en la
    # prosa, o en el título de la muestra, sí.
    muestra = ["::: muestra", "Lo que escribe la máquina", "", "    6396 * 7576 = 48406576", ":::"]
    fuera = revisar(["La cuenta es 6396 * 7576 = 48406576."])
    titulo = revisar(["::: muestra", "Lo que escribe: 3 × 4", "", "    texto", ":::"])
    exenta = not revisar(muestra) and clases(fuera) == ['notacion'] and clases(titulo) == ['notacion']
    # Cuarta (regla 5 septies): «token» y «parámetro» valen en la línea de su puente y dentro de un
    # bloque de tabla o de muestra; en la prosa de cualquier otro sitio, no. Y el puente de un
    # token no deja pasar un vector en la misma línea.
    ancla = "se les llama *tokens*"
    puente = "A estos trozos, en la prensa y en inglés, se les llama *tokens*; aquí, trozos."
    suelta = "Cada token es un número."
    en_tabla = ["::: tabla", "| trozo | token |", "Cada token es un número.", ":::"]
    en_su_puente = not revisar([puente], anclas=[ancla])
    sin_puente = clases(revisar([puente])) == ['jerga']
    en_prosa = clases(revisar([suelta], anclas=[ancla])) == ['jerga']
    en_bloque = not revisar(en_tabla) and not revisar(["::: muestra", "    token", ":::"])
    mezclado = clases(revisar([puente + " Y un vector."], anclas=[ancla])) == ['jerga']
    puentes_ok = en_su_puente and sin_puente and en_prosa and en_bloque and mezclado
    print(f"[3] invariante        en el mapa queda {clases(en_el_mapa)} "
          f"({'bien: solo la notación' if solo_notacion else 'MAL'}); "
          f"envoltorio de imprenta: {len(envuelto)} incidencias; texto de máquina en una muestra "
          f"exento, y fuera de ella o en su título, no: {exenta}; «token» libre en su puente y en "
          f"una tabla o muestra, y no en la prosa: {puentes_ok}")
    if not solo_notacion or envuelto or not exenta or not puentes_ok:
        fallos.append("invariante: el mapa levanta solo la jerga, el envoltorio no es notación, "
                      "solo el texto de máquina de una muestra está exento y «token» solo se "
                      "libra en su puente y en tablas y muestras")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las tres pruebas pasan.")
    return 0


if '--selftest' in sys.argv:
    sys.exit(selftest())

path = sys.argv[1]
lineas = open(path, encoding='utf-8').read().split('\n')

# El mapa de los sótanos (hoy el capítulo 15; fue el 14 hasta el 19 de septiembre de 2026)
# existe precisamente para imprimir los nombres técnicos que el resto del libro evita (regla
# 5 bis). Se le levanta la prohibición de jerga —y solo ésa—, en voz alta para que la excepción
# no pase inadvertida. Las demás siguen: ni fórmulas, ni griego, ni código.
# Se reconoce por el NOMBRE, no por el número: al renumerar, «cap14» pasó a ser otro capítulo
# y durante unas horas la exención se aplicó al capítulo equivocado (medido: 18 incidencias en
# el mapa y ninguna en el nuevo 14, que es justo al revés de lo que había que vigilar).
ES_EL_MAPA = 'mapa-de-los-sotanos' in path
if ES_EL_MAPA:
    print("NOTA: es el mapa de los sótanos; se permite la jerga y NO se permite nada más.")

fallos = revisar(lineas, ES_EL_MAPA, puentes_de(path))
if fallos:
    for f in fallos:
        print(f"  linea {f[0]:>4}  {f[1]:<9} {f[2]:<16} {f[3]}")
    print(f"\nFALLA: {len(fallos)} incidencia(s)")
    sys.exit(1)
print("PASA: cero notacion, cero jerga sin traducir")
