#!/usr/bin/env python3
"""Comprueba que las salidas de los programas escriben los números en castellano.

    verificar_cifras.py ../datos/salidas
    verificar_cifras.py --selftest

La regla 9 dice coma decimal y punto de millar: 2.095.402 y 90,4 %. Los ficheros de
`datos/salidas/` los cita el libro **literalmente** (regla 6), así que si un programa escribe
`2,095,402`, el libro tiene dos salidas y las dos son malas: copiarlo tal cual, y que la página
enseñe un número a la inglesa; o arreglarlo a mano al copiar, que es exactamente lo que la regla 6
prohíbe. Eso último ya pasó dos veces, en `retropropagacion.py` y en `ngrama.py`, y lo cazó un
auditor leyendo, no un verificador.

`formato.py` existe para esto: `miles()` y `coma()`. Lo que faltaba era alguien que lo comprobara.

Lo que NO se marca, porque no son cantidades y marcarlas convertiría esto en un verificador que
nadie mira: versiones (`Linux 6.18.44`), fechas ISO (`2026-09-18`), notación científica
(`3.7e-12`), nombres de fichero y rutas, e identificadores de modelo (`Qwen/Qwen2.5-0.5B`).

Ese último se añadió el 18 de septiembre, en la sesión del capítulo 7, y conviene saber por qué:
el `0.5` de `Qwen/Qwen2.5-0.5B` no es una cantidad, es parte del nombre con el que se descarga
el modelo. Castellanizarlo —«Qwen/Qwen2,5-0,5B»— no arregla nada: rompe el programa. El
verificador ya declaraba arriba que no marca nombres ni rutas, pero su guardia de ruta solo
miraba a la DERECHA del número, y en `Qwen/Qwen2.5-0.5B` la barra está a la izquierda. La regla
es estrecha a propósito —hace falta una letra o una barra a la izquierda, dentro de la misma
palabra— y está MEDIDA sobre los 44 ficheros de `datos/salidas/`: silencia los seis
`Qwen2.5-0.5B` y `Qwen2.5-1.5B` y ningún otro. `TAMAÑO 0.5B` (que sí es una cantidad) y
`1.98x` (una ventaja) siguen marcándose, y la prueba 5 del selftest está para que sigan.

Lo que tampoco se mira, y es lo único que se declara a mano (Orden 118, 31 oct 2026): la salida de
un programa AJENO que el libro cita tal cual. `una_tarea_con_asistente.py` enseña lo que escribe
Claude Code y lo que le devuelven sus herramientas —un `awk` suyo imprime `recalc max 6.69887`—, y
eso no es nuestro ni se puede retocar (regla 6). El programa que imprime esa salida la rodea con dos
líneas de marca, `[salida ajena: empieza …]` y `[salida ajena: termina]`, y las líneas que van
entre las dos no se miran. Solo esas: el resto del fichero y los demás ficheros, igual. Una marca
que empieza y no termina (o al revés) es un fallo, para que un descuido no deje sin mirar el resto
de un fichero."""
import os, re, sys

# Una cantidad con separador de millar inglés. Aquí pasa lo mismo que con el punto, en espejo:
# «0,018» es cero coma cero dieciocho en castellano, y «7,192» puede ser siete coma ciento noventa
# y dos. Con UNA sola coma no hay manera de saberlo. Con DOS o más (2,095,402) sí: en castellano
# eso no es nada.
MILES_INGLES = re.compile(r'(?<![\d.,])\d{1,3}(?:,\d{3}){2,}(?![\d.,])')
# Una sola coma seguida de tres cifras: ambiguo. Se avisa, no se suspende, salvo que la parte
# entera tenga más de tres cifras (12345,678 no es un decimal castellano con sentido).
MILES_DUDOSO = re.compile(r'(?<![\d.,])(\d{1,3}),(\d{3})(?![\d.,])')
# Un decimal con punto: 90.4, 0.933. Se excluye lo que tenga otro punto cerca (versiones, rutas),
# lo que lleve una e detrás (notación científica) y lo que vaya pegado a una letra.
#
# Y hay un caso que NO se puede decidir mirando el número: «1.024» es mil veinticuatro en
# castellano y uno coma cero veinticuatro en inglés, y las dos lecturas son legítimas. Un
# verificador que adivina acaba marcando texto bueno, y un verificador que marca texto bueno acaba
# desconectado. Así que se marca solo lo que es inequívoco:
#   - la parte decimal NO tiene exactamente tres cifras  (90.4, 0.9331) — nunca es millar
#   - o la parte entera es un cero solo                  (0.933)       — «cero mil» no existe
# «1.024», «15.048» y compañía se dejan pasar, y esta decisión está escrita aquí a propósito para
# que quien se encuentre uno sepa que no es un descuido: es que no se puede saber.
DECIMAL_CANDIDATO = re.compile(r'(?<![\w.,])(\d+)\.(\d+)(?![\d.]|\s*[eE][-+]?\d)(?![^\s]*[/\\])')


def es_decimal_ingles(entera, decimal):
    return len(decimal) != 3 or entera == "0"


# Una letra o una barra a la IZQUIERDA del número, dentro de la misma palabra: eso no es una
# cantidad, es un nombre. `Qwen2.5-0.5B`, `v1.2`, `datos/x2.5`. Se mira a la izquierda porque a
# la derecha ya mira DECIMAL_CANDIDATO. Estrecha a propósito: `0.5B` y `1.98x` NO llevan letra
# delante y se siguen marcando, porque ahí el número sí es la cantidad.
NOMBRE = re.compile(r'[A-Za-zÁÉÍÓÚÜÑáéíóúüñ/\\]')


def es_parte_de_un_nombre(linea, inicio):
    """¿El número cae dentro de una palabra que ya traía letras antes?"""
    corte = linea.rfind(" ", 0, inicio) + 1
    return bool(NOMBRE.search(linea[corte:inicio]))


# Y un segundo caso de número que es un nombre: la referencia a una entrada del catálogo de
# fallos o a una regla del libro. «fallo 4.30» y «regla 6.2» no son cantidades; son la
# numeración de un documento, y castellanizarlas —«fallo 4,30»— sería absurdo. Apareció el 18 de
# septiembre, en cuanto una cabecera de salida empezó a citar el catálogo por su número.
#
# La regla es estrecha: hace falta la palabra delante, pegada. Un «4.30» suelto se sigue
# marcando, que es lo que se quiere.
REFERENCIA = re.compile(
    r'\b(?:fallos?|reglas?|apartados?|propuestas?)\s+(?:el\s+|la\s+|los\s+|las\s+)?$',
    re.IGNORECASE)


def es_referencia_del_catalogo(linea, inicio):
    """¿El número va detrás de «fallo», «regla», «apartado» o «propuesta»?"""
    return bool(REFERENCIA.search(linea[:inicio]))


# Las dos líneas de marca de la salida ajena. Lo que sigue a «empieza» o «termina» es libre (el
# programa que las imprime explica ahí qué es y por qué no se toca).
MARCA_AJENA = re.compile(r'^\[salida ajena: (empieza|termina)\b')


def revisar_texto(texto):
    """Devuelve [(número de línea, clase, trozo)] de lo que está escrito a la inglesa."""
    fallos = []
    ajena = 0                                     # la línea donde empezó la salida ajena, o 0
    for i, l in enumerate(texto.splitlines(), 1):
        marca = MARCA_AJENA.match(l)
        if marca:
            if marca.group(1) == 'empieza' and ajena:
                fallos.append((i, 'marca', 'empieza otra salida ajena sin cerrar la de la línea %d' % ajena))
            elif marca.group(1) == 'termina' and not ajena:
                fallos.append((i, 'marca', 'termina una salida ajena que no había empezado'))
            ajena = i if marca.group(1) == 'empieza' else 0
            continue
        if ajena:
            continue
        # Una línea de traceback o de ruta no es una tabla de datos del libro.
        if l.lstrip().startswith(('File "', 'Traceback', '  File')):
            continue
        for m in MILES_INGLES.finditer(l):
            fallos.append((i, 'miles', m.group(0)))
        for m in MILES_DUDOSO.finditer(l):
            # «0,018» no es millar en ningún idioma: la parte entera de un millar nunca es cero.
            if m.group(1) != "0":
                fallos.append((i, 'dudoso', m.group(0)))
        for m in DECIMAL_CANDIDATO.finditer(l):
            if (es_decimal_ingles(m.group(1), m.group(2))
                    and not es_parte_de_un_nombre(l, m.start())
                    and not es_referencia_del_catalogo(l, m.start())):
                fallos.append((i, 'decimal', m.group(0)))
    if ajena:
        fallos.append((ajena, 'marca', 'empieza una salida ajena y no termina'))
    return fallos


def selftest():
    """Tres pruebas sobre lo único que decide: si un número está escrito a la inglesa."""
    fallos = []

    # 1. TEST NULO — todo lo que es correcto o no es una cantidad. Si algo de esto saltara, el
    #    verificador marcaría salidas buenas y acabaríamos desconectándolo, que es como se
    #    pierden los verificadores.
    bueno = "\n".join([
        "  aciertos          90,4 %",
        "  corpus         2.095.402 palabras",
        "Máquina: x86_64, Linux 6.18.44-fc-v33.",      # una versión no es una cantidad
        "Medido el 2026-09-18. Semilla: 20260914.",    # una fecha tampoco
        "diferencia máxima 3.7e-12",                   # notación científica
        "Escrito datos/salidas/que_mira.csv",          # una ruta
        "de 1.024 bytes justos",            # mil veinticuatro: ambiguo, no se marca
        "el kilobyte son 15.048 y pico",    # ídem
        "  salidas: 0,018  0,984  0,980",   # decimales castellanos, no millares
        "modelo: Qwen/Qwen2.5-0.5B en float32",   # un identificador no es una cantidad
    ])
    sale = revisar_texto(bueno)
    print(f"[1] test nulo         salida correcta: {len(sale)} incidencias"
          + (f" ({sale})" if sale else ""))
    if sale:
        fallos.append(f"test nulo: marca salida buena: {sale}")

    # 2. SEÑAL IMPLANTADA — los dos casos reales que ya ocurrieron en este libro, uno de cada
    #    clase, y cada uno tiene que cazarse por su nombre.
    implantadas = {'miles':   "Corpus: 2,095,402 caracteres de texto.",
                   'decimal': "un solo tribunal, sin capa     90.4 %"}
    cazadas = {c: c in {f[1] for f in revisar_texto(l)} for c, l in implantadas.items()}
    print("[2] señal implantada  " + "; ".join(
        f"{c}: {'la caza' if v else 'SE LE ESCAPA'}" for c, v in cazadas.items()))
    if not all(cazadas.values()):
        fallos.append("señal implantada: se le escapa "
                      + ", ".join(c for c, v in cazadas.items() if not v))

    # 3. INVARIANTE DEL DOMINIO — el mismo número escrito bien y mal en la misma línea: tiene que
    #    marcar uno y solo uno. Si marcara los dos, no está distinguiendo nada; si no marcara
    #    ninguno, tampoco.
    mezcla = revisar_texto("antes 2.095.402 y ahora 2,095,402")
    print(f"[3] invariante        el mismo número bien y mal en una línea: marca {len(mezcla)} "
          f"({[f[2] for f in mezcla]})")
    if len(mezcla) != 1 or mezcla[0][2] != "2,095,402":
        fallos.append(f"invariante: tenía que marcar solo el malo y marcó {mezcla}")

    # 4. LO QUE NO SE PUEDE DECIDIR — «1.024» es mil veinticuatro o uno coma cero veinticuatro, y
    #    no hay manera de saberlo mirando el número. Se deja pasar A PROPÓSITO, y esta prueba está
    #    aquí para que si alguien «mejora» el verificador y empieza a marcarlo, se entere.
    ambiguo = revisar_texto("tenía 1.024 bytes")
    claro = revisar_texto("acierta 90.4 % y 0.933")
    print(f"[4] lo ambiguo        «1.024»: {len(ambiguo)} incidencias (tiene que ser 0); "
          f"«90.4» y «0.933»: {len(claro)} (tienen que ser 2)")
    if ambiguo or len(claro) != 2:
        fallos.append("lo ambiguo: 1.024 se deja pasar; 90.4 y 0.933 no")

    # 5. LO QUE EL NOMBRE NO PUEDE TAPAR — la excusa del identificador es estrecha, y esta
    #    prueba es su valla: en cuanto el número deja de llevar letras delante vuelve a ser una
    #    cantidad, y hay que marcarlo. Si alguien ensancha la regla y empieza a perdonar
    #    «0.5B» o «1.98x», se entera aquí y no dentro de seis meses en una página impresa.
    nombre = revisar_texto("modelo: Qwen/Qwen2.5-0.5B")
    cantidad = revisar_texto("TAMAÑO 0.5B     ventaja 1.98x")
    print(f"[5] el nombre y la cifra  «Qwen/Qwen2.5-0.5B»: {len(nombre)} incidencias (tiene que "
          f"ser 0); «0.5B» y «1.98x»: {len(cantidad)} (tienen que ser 2)")
    if nombre or len(cantidad) != 2:
        fallos.append(f"el nombre y la cifra: el identificador se deja pasar; «0.5B» y «1.98x» "
                      f"no ({len(nombre)} y {len(cantidad)})")

    # 6. LA REFERENCIA Y LA CANTIDAD — misma valla para la otra excusa. «fallo 4.30» es la
    #    numeración del catálogo; «4.30» a secas, en una tabla, es una cantidad y se marca.
    #    Sin esta prueba, bastaría escribir la palabra «fallo» en cualquier sitio de la línea
    #    para silenciar todos los números que vinieran detrás.
    referencia = revisar_texto("la anterior no valía (fallo 4.30), y el fallo 4.31 encima")
    suelto = revisar_texto("   ventaja     4.30     media   0.933")
    lejos = revisar_texto("el fallo se ve en la columna de la derecha:   4.30")
    print(f"[6] la referencia         «fallo 4.30» y «el fallo 4.31»: {len(referencia)} "
          f"incidencias "
          f"(tiene que ser 0); sueltos: {len(suelto)} (2); con la palabra lejos: {len(lejos)} (1)")
    if referencia or len(suelto) != 2 or len(lejos) != 1:
        fallos.append(f"la referencia: «fallo 4.30» se deja pasar, un «4.30» suelto no, y la "
                      f"palabra «fallo» al principio de la línea no tapa nada "
                      f"({len(referencia)}, {len(suelto)}, {len(lejos)})")

    # 7. LA SALIDA AJENA (Orden 118) — un número a la inglesa DENTRO de las dos marcas pasa; el
    #    mismo, fuera, suspende; y una marca mal cerrada es un fallo (si no, un «empieza» olvidado
    #    dejaría sin mirar el resto del fichero).
    dentro = revisar_texto("antes 90,4 %\n[salida ajena: empieza — un awk\n"
                           "recalc max 6.69887 y 5.6118\n[salida ajena: termina]\ndespués 90,4 %")
    fuera = revisar_texto("antes 6.69887\n[salida ajena: empieza]\nrecalc max 6.69887\n"
                          "[salida ajena: termina]\ndespués 5.6118")
    sin_cerrar = revisar_texto("[salida ajena: empieza]\nrecalc max 6.69887\nluego 5.6118")
    sin_abrir = revisar_texto("luego 90,4 %\n[salida ajena: termina]")
    print(f"[7] la salida ajena   dentro de las marcas: {len(dentro)} incidencias (tiene que ser 0); "
          f"las mismas cifras fuera: {len(fuera)} (2); marca sin cerrar: "
          f"{[f[1] for f in sin_cerrar]} (['marca']); marca sin abrir: "
          f"{[f[1] for f in sin_abrir]} (['marca'])")
    if (dentro or [f[0] for f in fuera] != [1, 5] or [f[1] for f in sin_cerrar] != ['marca']
            or [f[1] for f in sin_abrir] != ['marca']):
        fallos.append(f"la salida ajena: dentro de las marcas pasa, fuera suspende, y una marca mal "
                      f"cerrada falla ({dentro}, {fuera}, {sin_cerrar}, {sin_abrir})")

    print()
    if fallos:
        for f in fallos:
            print("FALLA:", f)
        return 1
    print("SELFTEST: las siete pruebas pasan.")
    return 0


if '--selftest' in sys.argv:
    sys.exit(selftest())

carpeta = sys.argv[1]
assert os.path.isdir(carpeta), (
    f"Se esperaba la carpeta de las salidas; se encontró {carpeta!r}, que no es una carpeta")
total = malos = avisos = 0
for nombre in sorted(os.listdir(carpeta)):
    if not nombre.endswith('.txt'):
        continue
    total += 1
    todos = revisar_texto(open(os.path.join(carpeta, nombre), encoding='utf-8').read())
    ciertos = [f for f in todos if f[1] != 'dudoso']
    dudosos = [f for f in todos if f[1] == 'dudoso']
    if ciertos:
        malos += 1
        muestra = ", ".join(f"«{f[2]}» (l. {f[0]})" for f in ciertos[:3])
        print(f"  {nombre:<34} {len(ciertos):>3} a la inglesa: {muestra}")
    if dudosos:
        avisos += len(dudosos)
        muestra = ", ".join(f"«{f[2]}»" for f in dudosos[:2])
        print(f"  {nombre:<34} {'':>3} AVISO: {muestra} puede ser millar inglés o decimal nuestro")

if avisos:
    print(f"\n{avisos} caso(s) que no se pueden decidir mirando el número: una coma sola seguida")
    print("de tres cifras es millar en inglés y decimal en castellano. Avisa y no suspende.")
if malos:
    print(f"\nFALLA: {malos} de {total} ficheros de salida escriben números a la inglesa.")
    print("Se arregla en el programa que los imprime, con miles() y coma() de formato.py,")
    print("nunca a mano en el libro al copiarlos (regla 6).")
    sys.exit(1)
# Un cero aquí no es un aprobado: es que no se ha mirado nada. Apuntado con la carpeta
# equivocada, este verificador imprimía «PASA: los 0 ficheros» y devolvía 0, que es el mismo
# agujero que el test nulo del capítulo 9 (fallo 4.31): decir «pasa» sin haber comprobado.
assert total, (
    f"Se esperaban ficheros .txt de salida en {carpeta}; no hay ninguno. Un verificador que "
    "no encuentra nada que mirar no aprueba: avisa de que le has dado la carpeta que no era")
print(f"PASA: los {total} ficheros de salida escriben los números en castellano.")
