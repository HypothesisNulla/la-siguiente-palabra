# De dónde sale cada figura

Tabla calculada, no escrita a mano, por `codigo/de_donde_sale_cada_figura.py`.  
Fecha: 2026-11-15 10:50 UTC.  
Libro: con los borradores de en-curso/; numeración cotejada con La-siguiente-palabra-lectura.pdf.  
56 figuras. La tabla no depende de la máquina: sale de ficheros de texto.

Cómo se lee:

- **fig.** es el número de la figura, en el orden en que salen en el libro.
- **cap.** es el capítulo en que está. Con una «A» delante (por ejemplo A8), está en el anexo de ese capítulo.
- **programa** es el programa que la dibuja. En `codigo/` el fichero es `figura_` + el nombre + `.py`, salvo cuando el nombre ya empieza por `figuras_` (por ejemplo `figuras_el_comite_por_dentro`): entonces es el nombre entero. «(foto)» quiere decir que ningún programa la dibuja: es una fotografía.
- **salidas que lee** son los ficheros de `datos/salidas/` de los que el programa saca sus números, con el nombre sin `.txt` (si acaban de otra manera, va escrito). Con asterisco, un fichero de `datos/entrada/`. «ninguna» quiere decir que el programa no lee ningún fichero: dibuja un esquema, calcula él mismo los números o los trae escritos como ejemplo.

| fig. | cap. | programa | salidas que lee |
|--:|--:|:--|:--|
| 1 | 1 | paginas_libreta | traza_escalera, casillas_vacias, *quijote.txt |
| 2 | 1 | pagina_el_caballero | pagina_el_caballero, *quijote.txt |
| 3 | 1 | casillas_vacias | casillas_vacias |
| 4 | 2 | el_comite | ninguna |
| 5 | 2 | que_es_aprender | ninguna |
| 6 | 2 | (foto) | ninguna |
| 7 | 2 | el_colegio | ninguna |
| 8 | 3 | figuras_el_comite_por_dentro | el_comite_por_dentro |
| 9 | 3 | figuras_el_comite_por_dentro | el_comite_por_dentro |
| 10 | 3 | figuras_el_comite_por_dentro | el_comite_por_dentro |
| 11 | 3 | cuadrado_del_colegio | ninguna |
| 12 | 3 | figuras_el_comite_por_dentro | el_comite_por_dentro |
| 13 | 3 | una_capa | ninguna |
| 14 | 4 | la_red_del_colegio | culpa_en_el_colegio_datos.csv |
| 15 | 5 | mapa_de_palabras | palabras_de_cerca |
| 16 | 5 | misma_direccion | palabras_de_cerca |
| 17 | 6 | copias_muestra | copias_de_una_muestra |
| 18 | 6 | red_desenrollada | una_prueba_de_memoria |
| 19 | 7 | linea_circular | maquina_entera |
| 20 | 7 | trocear_a_mano | trocear_a_mano.csv, trocear_a_mano |
| 21 | 7 | encuesta | maquina_entera, banco_y_paris |
| 22 | 8 | cuadricula | cuadricula_y_fila |
| 23 | 8 | papelera_cuerpo | ninguna |
| 24 | 9 | en_fila | cuadricula_y_fila |
| 25 | 10 | cuatro_retratos | cuatro_retratos, en_que_orden_aprende |
| 26 | 10 | lo_que_encuentra_en_un_paso | cuatro_retratos |
| 27 | 11 | seis_maneras | crecer.csv |
| 28 | 11 | todo_o_nada | todo_o_nada.csv |
| 29 | 13 | cien_papeletas | temperatura_a_mano.csv |
| 30 | 14 | cifra_a_cifra | ordenes_a_la_vista |
| 31 | 14 | con_manos | ninguna |
| 32 | 15 | la_salamandra | ninguna |
| 33 | 15 | el_empujon | cuentas_a_mano.csv |
| 34 | 15 | prueba_de_chollet | *arc_25ff71a9.json |
| 35 | 15 | tablero | ninguna |
| 36 | 16 | dos_valles | dos_valles.csv |
| 37 | A1 | lo_poco_que_hay | casillas_vacias |
| 38 | A3 | diez_digitos | ninguna |
| 39 | A3 | saltan_la_regla | reloj_a_mano |
| 40 | A3 | siete_segmentos | ninguna |
| 41 | A3 | sorteo_del_arranque | el_sorteo_del_arranque |
| 42 | A3 | los_dos_que_mas_se_parecen | que_mira_cada_una |
| 43 | A4 | la_red_del_colegio | culpa_en_el_colegio_datos.csv |
| 44 | A4 | cuadrado_de_la_red | ninguna |
| 45 | A4 | de_la_regla_al_cuadrado | culpa_en_el_colegio_datos.csv |
| 46 | A4 | la_rampa_del_colegio | culpa_en_el_colegio_datos.csv |
| 47 | A4 | culpa_hacia_atras_colegio | culpa_en_el_colegio_datos.csv, culpa_en_el_colegio |
| 48 | A4 | el_terreno_del_colegio | culpa_en_el_colegio_datos.csv, culpa_en_el_colegio |
| 49 | A4 | la_rampa_y_el_codo | culpa_hacia_atras |
| 50 | A5 | misma_direccion | palabras_de_cerca |
| 51 | A8 | maquina_del_quijote | la_e_de_acordarme |
| 52 | A8 | mirada_a_mano | mirada_a_mano |
| 53 | A8 | el_reparto | reparto_una_a_una.csv |
| 54 | A10 | cuanto_ha_leido | cuatro_retratos |
| 55 | A10 | lo_mal_que_lo_hace | en_que_orden_aprende_curva.csv, en_que_orden_aprende |
| 56 | A10 | seguir_entrenando | seguir_entrenando.csv |
