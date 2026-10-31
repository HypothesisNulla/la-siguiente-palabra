#!/usr/bin/env python3
"""Comprueba que todas las citas en bloque de un capítulo son literales: que aparecen tal cual
en la salida del script que las generó.

    verificar_muestras.py capitulo.md salida.txt [salida2.txt ...]

En este libro una cita en bloque es, por defecto, salida de máquina sin retocar. Las únicas
excepciones son bloques escritos por el autor (frases de ejemplo, definiciones); para que el
verificador las acepte hay que declararlas a mano, una por una, en el fichero de excepciones.
Declararlas es un acto consciente: si no está declarada y no es literal, el capítulo falla.

Los bloques se localizan sobre el texto CRUDO, con sus saltos de línea intactos; la normalización
(espacios y coma decimal) se aplica solo al comparar."""
import re, sys, io, os

MINIMO = 10_000   # el bloque entero: ver el fallo 4.21 del catálogo
EXCEPCIONES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', '..', 'libro-ia-libro', 'notas', 'CITAS-DE-AUTOR.md')

def quitar_marcado(t, comillas=True):
    """Quita el marcado de imprenta que envuelve trozos de la salida, dejando el trozo.

    A temperatura alta el modelo escribe en chino, tailandés o árabe. La fuente del libro no
    dibuja esos signos, y xelatex los tira EN SILENCIO: sin envolverlos, el libro enseñaría una
    versión limpiada de lo que la máquina escribió. El envoltorio es formato, no dato.

    También quita las comillas simples inversas, que en el libro ponen un trozo en monoespaciado
    y no forman parte de lo que imprimió el programa. Salvo con `comillas=False`, que es lo que
    se pide dentro de un bloque `::: muestra` (Orden 118): allí las líneas sangradas son la
    salida de máquina tal cual, y una comilla invertida es un carácter que la máquina escribió
    (el asistente de `una_tarea_con_asistente.py` escribe `fichero.csv` con comillas
    invertidas, como en markdown), no marcado de imprenta."""
    t = re.sub(r'`\\[a-záéíóúñ]+\{(.*?)\}`\{=latex\}', r'\1', t)
    return t.replace('`', '') if comillas else t


def normalizar(t):
    """Compara ignorando dos convenciones de imprenta que no son datos: los espacios y el
    separador decimal (el libro escribe 0,018 y algún script viejo imprime 0.018). El espacio
    antes del signo de porcentaje NO se ignora desde el 20 sep 2026: lo pone el programa
    (formato.py, regla 9) y el libro lo copia tal cual; tolerarlo tapaba que el libro lo
    quitara (capítulo 13, M7 de su auditoría)."""
    t = re.sub(r'(?<=\d),(?=\d)', '.', t)
    return re.sub(r'\s+', ' ', t)

def texto_del_bloque(b, literal=False):
    lineas = [re.sub(r'^(> ?| {4})', '', l) for l in b.splitlines()]
    lineas = [l for l in lineas if l.strip() != '```']   # las vallas son formato, no dato
    return normalizar(quitar_marcado(' '.join(lineas), not literal)).strip()

def fila_esta_en_alguna_linea(celdas, lineas=None):
    """¿Hay una línea de la salida que contenga todas estas celdas, en este orden?

    En orden y sin cambiar nada, pero permitiendo que entre medias haya cosas: el libro
    puede QUITAR una columna que no cabe o que es notación (regla 6), y entonces su fila
    es un trozo de la del programa, no la fila entera. Lo que no puede es cambiar un
    número, ni cambiarlos de orden, ni traerse uno de otra fila."""
    patron = re.compile(r'(?<![\w,.])' + r'.*?'.join(re.escape(c) for c in celdas))
    return any(patron.search(l) for l in (lineas_salida if lineas is None else lineas))


def lineas_crudas_del_bloque(b, literal=False):
    """Las líneas del bloque sin el prefijo de cita ni las vallas, pero SIN aplastar los espacios."""
    out = []
    for l in b.splitlines():
        l = re.sub(r'^(> ?| {4})', '', l)
        if l.strip() == '```':
            continue
        l = quitar_marcado(l, not literal).rstrip()
        if l.strip():
            out.append(l)
    return out


def casa_con_los_espacios(linea):
    """¿Esta línea del libro está en la salida con sus rachas de espacios intactas?

    El libro SÍ puede repartir un renglón largo en varias líneas para que quepa en la página, y
    eso convierte un espacio suelto en un salto de línea. Lo que no puede es cambiar una racha de
    tres espacios por uno: ahí el espacio es un dato. Así que el espacio suelto se compara con
    manga ancha —vale cualquier cosa en blanco— y la racha de dos o más, al milímetro."""
    partes = re.split(r'( {2,})', linea)
    patron = ''.join(re.escape(p) if p.startswith('  ') else re.escape(p).replace(r'\ ', r'\s+')
                     for p in partes)
    return re.search(patron, salida_con_espacios) is not None


def espacios_comidos(b, literal=False):
    """¿El libro ha cambiado una racha de espacios por uno solo?

    Un bloque de máquina puede QUITAR una columna entera que no cabe (regla 6), y un renglón de
    220 caracteres tiene que partirse porque la página mide seis pulgadas. Lo que no puede es
    cambiar tres espacios por uno: en la máquina de contar letras del capítulo 1 el espacio es
    uno de los 42 símbolos que se eligen, con su frecuencia real, y una racha de tres es lo que
    la máquina escribió. Antes esto pasaba desapercibido porque la comparación aplasta los
    espacios de los dos lados y luego anuncia «0 recortadas».

    Solo se avisa de la línea que SÍ está en la salida aplastando espacios y NO está sin
    aplastarlos: ésa es exactamente la que el libro ha retocado. Una línea que falta del todo, o
    a la que se le ha quitado una columna, la caza la comprobación de siempre."""
    avisos = 0
    for l in lineas_crudas_del_bloque(b, literal):
        if normalizar(l).strip() in salida and not casa_con_los_espacios(l.strip()):
            print(f"AVISO, espacios distintos: «{l.strip()[:50]}…» está en la salida con otras\n  rachas de espacios. Si son ancho de columna, da igual; si el espacio es un dato del\n  bloque —capítulo 1—, no da igual.")
            avisos += 1
    return avisos


def parece_salida_de_maquina(lineas):
    """¿Este capítulo enseña ALGO con pinta de salida de máquina?

    Se mira a propósito con otros ojos que el analizador de más abajo: vallas de código,
    sangría de cuatro espacios, citas en bloque, barras verticales. Si los dos miraran igual,
    un cambio de formato que despistara al analizador despistaría también a este aviso, y el
    verificador pasaría a dar por bueno un capítulo entero sin mirar nada."""
    for l in lineas:
        if l.startswith('```') or l.startswith('~~~'):
            return True
        if (l.startswith('    ') or l.startswith('>') or '|' in l) and re.search(r'\d', l):
            return True
    return False


def titulos_y_notas(lineas, salida_lineas):
    """Los renglones de título y de nota de los bloques «::: tabla» y «::: muestra» que no
    están, tal cual, en la salida (formato.py, tabla_editorial y muestra_editorial).

    Sus filas de tabla y su texto sangrado ya se comprueban como cualquier otra tabla o
    bloque. El título y la nota también los escribe el programa, y un título retocado a mano
    cambia lo que el lector cree que está mirando."""
    perdidos, dentro, n = [], False, 0
    conjunto = set(salida_lineas)
    for l in lineas:
        if re.match(r'^:::\s*(tabla|muestra)\s*$', l):
            dentro, n = True, n + 1
            continue
        if dentro and l.strip() == ':::':
            dentro = False
            continue
        if dentro and l.strip() and not l.startswith('|') and not l.startswith('    '):
            t = normalizar(quitar_marcado(l)).strip()
            if t not in conjunto:
                perdidos.append(t)
    return n, perdidos


def lineas_de_datos(b, literal=False):
    """Las líneas del bloque, sin el prefijo de cita ni las vallas de código."""
    out = []
    for l in b.splitlines():
        l = re.sub(r'^(> ?| {4})', '', l)
        if l.strip() == '```':
            continue
        l = normalizar(quitar_marcado(l, not literal)).strip()
        if l:
            out.append(l)
    return out


def comprobar_bloque(b, salida_texto, literal=False):
    """Un bloque contra la salida (ya normalizada): ('literal', None) si entero está en ella;
    ('recompuesto', primera línea) si cada línea, una por una, lo está; ('no literal', primera
    línea que falta) si no. `literal` es True dentro de un `::: muestra` (ver quitar_marcado)."""
    t = texto_del_bloque(b, literal)
    if t[:MINIMO] in salida_texto:
        return 'literal', None
    sueltas = lineas_de_datos(b, literal)
    perdidas = [l for l in sueltas if l not in salida_texto]
    if perdidas:
        return 'no literal', perdidas[0]
    return 'recompuesto', sueltas[0]


def selftest():
    """Tres pruebas sobre lo único que este verificador decide de verdad: si una fila de
    tabla del libro está o no en la salida del programa."""
    fallos = []
    lineas = [normalizar("        20         0.933 (0.867-1.000)       1.000 (1.000-1.000)").strip(),
              normalizar("       160         0.114 (0.109-0.119)       0.121 (0.114-0.128)").strip()]

    # 1. TEST NULO — las celdas existen, pero repartidas entre DOS líneas distintas. No
    #    puede dar por buena una fila cuyos números vengan de sitios diferentes: sería
    #    exactamente la manera de colar un dato inventado sin que nadie lo note.
    mezclada = fila_esta_en_alguna_linea(["20", "0.933", "0.121"], lineas)
    print(f"[1] test nulo         celdas de dos líneas distintas: ¿las acepta? "
          f"{'SÍ' if mezclada else 'no'}")
    if mezclada:
        fallos.append("test nulo: acepta una fila cuyas celdas vienen de dos líneas distintas")

    # 2. SEÑAL IMPLANTADA — se cambia una sola cifra de una fila que sí está. Tiene que
    #    dejar de encontrarla.
    buena = fila_esta_en_alguna_linea(["20", "0.933", "1.000"], lineas)
    tocada = fila_esta_en_alguna_linea(["20", "0.934", "1.000"], lineas)
    print(f"[2] señal implantada  la fila buena: {'la encuentra' if buena else 'NO la encuentra'}; "
          f"con una cifra cambiada: {'la encuentra' if tocada else 'no la encuentra'}")
    if not buena or tocada:
        fallos.append("señal implantada: tiene que encontrar la fila buena y no la tocada")

    # 3. INVARIANTE DEL DOMINIO — el libro puede QUITAR una columna (regla 6), así que una
    #    fila más corta y en orden vale; en desorden, no, porque el orden es un dato.
    corta = fila_esta_en_alguna_linea(["20", "1.000"], lineas)
    revuelta = fila_esta_en_alguna_linea(["1.000", "0.933", "20"], lineas)
    print(f"[3] invariante        quitando una columna: {'vale' if corta else 'NO vale'}; "
          f"en otro orden: {'vale' if revuelta else 'no vale'}")
    if not corta or revuelta:
        fallos.append("invariante: quitar una columna vale; cambiar el orden no")

    # 4. LA GUARDIA — un capítulo sin datos medidos (el prólogo) tiene que pasar en silencio;
    #    un capítulo que SÍ trae bloques o tablas y del que no se reconoce nada tiene que
    #    reventar, porque eso significa que el formato ha cambiado y que desde ese momento
    #    este verificador estaría aprobando sin mirar.
    sin_datos = parece_salida_de_maquina(["Prosa corriente, con el año 1948 dentro.",
                                          "Otra línea sin nada medido."])
    con_valla = parece_salida_de_maquina(["Texto.", "```", "  0,933", "```"])
    con_sangria = parece_salida_de_maquina(["Texto.", "    aciertos       0,933"])
    con_tabla = parece_salida_de_maquina(["Texto.", "| capa | 90,4 % |"])
    print(f"[4] la guardia        prosa sin datos: {'la avisa' if sin_datos else 'calla'}; "
          f"valla: {'avisa' if con_valla else 'CALLA'}; "
          f"sangría: {'avisa' if con_sangria else 'CALLA'}; "
          f"tabla: {'avisa' if con_tabla else 'CALLA'}")
    if sin_datos or not (con_valla and con_sangria and con_tabla):
        fallos.append("la guardia: tiene que callar con prosa y avisar con valla, sangría o tabla")

    # 5. LOS ESPACIOS — un bloque del libro que cuadre solo aplastando rachas de espacios no es
    #    literal. Se comprueba sobre la comparación cruda, que es la que decide.
    maquina = "hay tres   espacios seguidos aqui y dos  aqui"
    igual = "hay tres   espacios seguidos aqui y dos  aqui"
    comido = "hay tres espacios seguidos aqui y dos aqui"
    print(f"[5] los espacios      copia exacta: {'pasa' if igual in maquina else 'NO PASA'}; "
          f"con los espacios comidos: {'PASA (mal)' if comido in maquina else 'no pasa'}")
    if igual not in maquina or comido in maquina:
        fallos.append("los espacios: la copia exacta pasa; la de espacios comidos, no")

    # 6. LOS RÓTULOS — la fila de cabecera se compara igual que una de datos: se puede
    #    quitar una columna, no rebautizarla. Se prueba sobre la cabecera de verdad del
    #    programa del capítulo 9, que es donde apareció el fallo.
    cab = [normalizar("  longitud   en serie (s)   a la vez (s)   ventaja").strip()]
    literal = fila_esta_en_alguna_linea(["longitud", "en serie (s)", "a la vez (s)",
                                         "ventaja"], cab)
    recortada = fila_esta_en_alguna_linea(["longitud", "ventaja"], cab)
    inventada = fila_esta_en_alguna_linea(["longitud del texto, en palabras",
                                           "leyendo en orden, segundos"], cab)
    print(f"[6] los rótulos       los del programa: {'pasan' if literal else 'NO PASAN'}; "
          f"quitando una columna: {'pasan' if recortada else 'NO PASAN'}; "
          f"rebautizados: {'PASAN (mal)' if inventada else 'no pasan'}")
    if not literal or not recortada or inventada:
        fallos.append("los rótulos: los del programa pasan, quitar una columna vale, "
                      "rebautizarlos no")

    # 7. LA CLAVE DEBAJO DE LA TABLA — el sitio donde el programa SÍ puede decir largo lo
    #    que significa una columna. El rótulo tiene que caber en la caja, así que es corto;
    #    la clave va debajo, en línea entera, y el libro la copia como copia cualquier otra
    #    línea del bloque. Esta prueba comprueba que la clave se compara igual de estricto
    #    que una fila: literal o nada.
    #
    #    Antes aquí había una prueba de cabecera APILADA en dos renglones, que es lo que el
    #    informe del capítulo 9 proponía. Se quitó al comprobar que no se puede verificar: al
    #    apilar por columnas, los dos renglones leídos seguidos entrelazan las palabras de
    #    columnas distintas —«longitud del | pasos en fila, | texto, en trozos | leyendo en
    #    orden»— y la celda del libro no aparece entera en ningún sitio. Un verificador que
    #    no puede comprobar algo no debe fingir que lo comprueba.
    clave = [normalizar("«ventaja»: cuántas veces más rápida es la de «a la vez»;").strip()]
    copiada = fila_esta_en_alguna_linea(["«ventaja»: cuántas veces más rápida es la de "
                                         "«a la vez»;"], clave)
    retocada = fila_esta_en_alguna_linea(["«ventaja»: cuántas veces más rápida es la "
                                          "moderna;"], clave)
    print(f"[7] la clave          copiada tal cual: {'pasa' if copiada else 'NO PASA'}; "
          f"reescrita por el libro: {'PASA (mal)' if retocada else 'no pasa'}")
    if not copiada or retocada:
        fallos.append("la clave: la línea que explica una columna se copia literal o no vale")

    # 8. EL TÍTULO Y LA NOTA DE UNA TABLA EDITORIAL — se copian literales, como las filas.
    sal = [normalizar(x).strip() for x in ("Lo que puede venir detrás de «La capital»",
                                            "Modelo de 7.000 millones, en crudo.")]
    bien = ["::: tabla", "Lo que puede venir detrás de «La capital»", "", "| a | b |",
            "|:--|--:|", "| x | 1 |", "", "Modelo de 7.000 millones, en crudo.", ":::"]
    mal = list(bien)
    mal[7] = "Modelo de 7.000 millones, adiestrado."
    nb, pb = titulos_y_notas(bien, sal)
    nm, pm = titulos_y_notas(mal, sal)
    print(f"[8] tabla editorial   título y nota copiados: {'pasan' if not pb else 'NO PASAN'}; "
          f"nota retocada: {'PASA (mal)' if not pm else 'no pasa'}")
    if pb or not pm or nb != 1:
        fallos.append("tabla editorial: el título y la nota se copian literales o no valen")

    # 9. LAS COMILLAS INVERTIDAS DENTRO DE UNA MUESTRA (Orden 118) — en un bloque `::: muestra` las
    #    líneas sangradas son salida de máquina tal cual, y una comilla invertida es un carácter
    #    que escribió la máquina (el asistente de una_tarea_con_asistente.py escribe `fichero.csv`
    #    como en markdown). Con `literal` se comparan tal cual y el bloque pasa; fuera de una
    #    muestra se siguen quitando; y la prueba del revés: una palabra cambiada sigue suspendiendo.
    sal = normalizar("Sale del fichero A (`cuantos_hilos-A.csv`), con 14 hilos y\n"
                     "longitud 512, y la columna `ventaja` lo confirma.\n")
    bloque = ("    Sale del fichero A (`cuantos_hilos-A.csv`), con 14 hilos y\n"
              "    longitud 512, y la columna `ventaja` lo confirma.\n")
    cambiado = bloque.replace("longitud", "anchura")
    en_muestra = comprobar_bloque(bloque, sal, literal=True)[0]
    fuera = comprobar_bloque(bloque, sal, literal=False)[0]
    revuelta = comprobar_bloque(cambiado, sal, literal=True)[0]
    print(f"[9] comillas invertidas  en una muestra: {en_muestra}; fuera de una muestra se "
          f"quitan: {fuera}; con una palabra cambiada: {revuelta}")
    if en_muestra != 'literal' or fuera == 'literal' or revuelta != 'no literal':
        fallos.append("comillas invertidas: dentro de una muestra se comparan tal cual y el bloque "
                      "pasa; fuera se quitan; una palabra cambiada suspende")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las nueve pruebas pasan.")
    return 0


lineas_salida = []
if '--selftest' in sys.argv:
    sys.exit(selftest())

capitulo = sys.argv[1]
crudo = io.open(capitulo, encoding='utf-8').read()
salida = ' '.join(normalizar(io.open(f, encoding='utf-8').read()) for f in sys.argv[2:])
# Segunda copia de la salida SIN aplastar los espacios. normalizar() convierte cualquier racha
# de espacios en uno solo, que es lo correcto para comparar un bloque partido en líneas, pero
# deja ciego al verificador ante un bloque donde el espacio ES UN DATO: en la máquina de contar
# letras del capítulo 1, el espacio es uno de los 42 símbolos que se eligen, con su frecuencia
# real, y una racha de tres espacios seguidos es lo que la máquina escribió.
salida_con_espacios = ' '.join(
    io.open(f, encoding='utf-8').read().replace('\n', ' ') for f in sys.argv[2:])
assert salida.strip(), "no me has dado ninguna salida contra la que comparar"

# Las mismas salidas, pero línea a línea. Una fila de tabla del libro sale de UNA línea de
# la salida del programa; comprobarla contra el montón entero dejaría pasar una fila cuyos
# números vinieran de tres sitios distintos.
for f in sys.argv[2:]:
    for l in io.open(f, encoding='utf-8'):
        # Sin las comillas de monoespaciado (formato.py, trozo): en la fila del libro tampoco
        # cuentan, y una celda «`_Par` + `ís`» se compara como «_Par + ís».
        l = normalizar(quitar_marcado(l)).strip()
        if l:
            lineas_salida.append(l)


declaradas = []
if os.path.exists(EXCEPCIONES):
    for l in io.open(EXCEPCIONES, encoding='utf-8'):
        m = re.match(r'^-\s+`([^`]+)`', l)
        if m:
            declaradas.append(normalizar(m.group(1)).strip())

# Un bloque es cualquier cosa sangrada: con «>» (cita) o con cuatro espacios (salida de
# programa que necesita monoespaciado, como una tabla con columnas). La regla 6 dice
# «todo lo que va sangrado en bloque», no «todo lo que lleva >»: durante un tiempo el
# verificador solo miraba las citas, y una tabla sangrada con cuatro espacios pasaba sin
# que nadie la comprobara.
muestras = [(m.start(), m.end()) for m in re.finditer(r'^::: muestra\s*\n.*?^:::\s*$', crudo,
                                                       re.M | re.S)]
bloques = [(m.group(1), any(a <= m.start() < z for a, z in muestras))
           for m in sorted(list(re.finditer(r'((?:^>.*\n)+)', crudo, re.M)) +
                           list(re.finditer(r'((?:^ {4}.*\n|^\n(?= {4}))+)', crudo, re.M)),
                           key=lambda m: m.start())]

fallos = comprobados = de_autor = recompuestos = 0
for b, literal in bloques:
    t = texto_del_bloque(b, literal)
    if not t:
        continue
    if any(t.startswith(d) for d in declaradas):
        de_autor += 1
        continue
    comprobados += 1
    # Si el bloque entero no cuadra, segunda oportunidad: que cada línea, una por una, sea
    # literal. Eso permite quitar del libro lo que es adorno de consola —unas comillas, una
    # columna de barras que no cabe en la página— y sigue prohibiendo cambiar un solo dato.
    estado, linea = comprobar_bloque(b, salida, literal)
    if estado == 'literal':
        espacios_comidos(b, literal)
    elif estado == 'no literal':
        print(f"NO LITERAL: «{linea[:60]}…»")
        fallos += 1
    else:
        recompuestos += 1
        print(f"recompuesto (líneas literales, bloque recortado): «{linea[:50]}…»")

# Las tablas con números también son datos medidos: cada fila tiene que aparecer, con sus
# números y en el mismo orden, en la salida del programa.
#
# Antes esto solo miraba las filas que llevaban un signo de porcentaje, y las tablas de
# tiempos del capítulo de la atención —segundos, sin porcentajes— no las comprobaba nadie.
# Ahora mira cualquier fila con una cifra. Las tablas que NO son medición (una tabla que
# enseña una idea, como «64 palabras, 64 pasos en fila») se declaran a mano en el mismo
# fichero de excepciones que los bloques, por su primer trozo de texto.
def celdas_de(l):
    """Las celdas de una fila de tabla del libro, sin el marcado de imprenta.

    El asterisco doble es negrita del libro, no un dato: fuera de las celdas antes de
    compararlas, o la celda «**1,000**» no se encuentra nunca."""
    celdas = [normalizar(quitar_marcado(c)).replace('**', '').strip()
              for c in l.strip('|').split('|')]
    return [c for c in celdas if c and not set(c) <= set('-: ')]


def tablas_del_capitulo(lineas):
    """Las tablas del capítulo, cada una como (cabecera, [filas de datos]).

    Una tabla empieza en la fila de rótulos, que se reconoce porque la línea siguiente es
    la de guiones, y termina en la primera línea que ya no empieza por barra."""
    out = []
    k = 0
    while k < len(lineas) - 1:
        siguiente = lineas[k + 1]
        if (lineas[k].startswith('|') and siguiente.startswith('|')
                and set(siguiente) <= set('|-: ')):
            datos, j = [], k + 2
            while j < len(lineas) and lineas[j].startswith('|'):
                datos.append(lineas[j])
                j += 1
            out.append((lineas[k], datos))
            k = j
        else:
            k += 1
    return out


filas = cabeceras = 0
lineas_crudas = crudo.splitlines()
for cabecera, datos in tablas_del_capitulo(lineas_crudas):
    medida = False
    for l in datos:
        if not re.search(r'\d', l):
            continue
        celdas = celdas_de(l)
        fila = ' '.join(celdas)
        if any(fila.startswith(d) for d in declaradas):
            de_autor += 1
            continue
        filas += 1
        medida = True
        if fila in salida or fila_esta_en_alguna_linea(celdas):
            continue
        print(f"TABLA NO LITERAL: «{fila[:60]}…»")
        fallos += 1

    # La fila de rótulos SÍ es un dato, y esto no se miraba (fallo 4.32). En el capítulo 9
    # el libro reescribió a mano las doce columnas de sus tres tablas y, de paso, le puso
    # al eje una unidad que el programa no mide: «longitud del texto, EN PALABRAS», cuando
    # lo que el programa cronometra es ruido, y la longitud son posiciones. Un rótulo
    # inventado es un dato inventado, y encima es el que dice qué significan los demás.
    #
    # Solo se comprueban los rótulos de las tablas de las que ya se ha comprobado alguna
    # fila de datos: una tabla ilustrativa, declarada a mano, no la toca esto.
    if not medida:
        continue
    celdas = celdas_de(cabecera)
    rotulo = ' '.join(celdas)
    if not celdas or any(rotulo.startswith(d) for d in declaradas):
        de_autor += 1
        continue
    cabeceras += 1
    if rotulo in salida or fila_esta_en_alguna_linea(celdas):
        continue
    print(f"ROTULOS NO LITERALES: «{rotulo[:60]}…»\n  Los rótulos de las columnas los "
          "escribe el programa, no el libro. El libro puede QUITAR una columna; no puede\n"
          "  rebautizarla, y mucho menos ponerle una unidad.")
    fallos += 1

divs, perdidos = titulos_y_notas(lineas_crudas, lineas_salida)
for t in perdidos:
    print(f"TÍTULO O NOTA NO LITERAL: «{t[:60]}…»")
    fallos += 1

# Un capítulo puede no enseñar ni un dato medido —el prólogo no enseña ninguno— y eso no es un
# fallo. El fallo es que el capítulo SÍ traiga bloques o tablas y no se reconozca ninguno: ahí el
# formato ha cambiado y este verificador habría pasado a aprobar sin mirar. Antes los dos casos
# daban el mismo error, y el aviso que importa quedaba escondido detrás del que no importa.
if not (comprobados or filas or cabeceras or de_autor or divs):
    assert not parece_salida_de_maquina(crudo.splitlines()), (
        f"{capitulo} trae bloques o filas con cifras y no he reconocido ninguno: el formato ha "
        "cambiado y desde ahora este verificador estaría aprobando el capítulo sin mirar nada")
    print(f"PASA: {capitulo} no enseña ningún dato medido, así que no hay nada que cotejar")
    sys.exit(0)

print(f"{comprobados} citas de máquina comprobadas ({recompuestos} recortadas), "
      f"{de_autor} bloques declarados del autor, {filas} filas de tabla con números, "
      f"{cabeceras} filas de rótulos, {divs} tablas o muestras editoriales")
print("FALLA" if fallos else "PASA: todas las citas de máquina son literales")
sys.exit(1 if fallos else 0)
