# Guía del alumno

## Resultado esperado

Al terminar debes entregar:

- Una base limpia con 168 partidos: 48 de 2018, 48 de 2022 y 72 de 2026.
- Una bitácora de problemas detectados y decisiones de limpieza.
- Una comparación de los tres torneos.
- Un modelo de clasificación sencillo y una reflexión sobre sus limitaciones.

## Parte A. Perfilado en Excel

### Paso 1. Abrir y reconocer los archivos

1. Abre `Kit_Excel_Alumnos.xlsx`.
2. Revisa las hojas de datos sucios.
3. No edites todavía los valores.
4. Registra en `06_Control_Calidad`:
   - cantidad de filas;
   - columnas disponibles;
   - valores vacíos;
   - posibles duplicados;
   - formatos extraños;
   - nombres de equipo que parecen representar la misma selección.

### Paso 2. Convertir cada rango en tabla

1. Selecciona los datos de una hoja.
2. Usa **Insertar > Tabla**.
3. Activa “La tabla tiene encabezados”.
4. Asigna nombres claros: `Sucio2018`, `Sucio2022` y `Sucio2026`.

Las tablas permiten filtrar, ordenar y revisar valores únicos con mayor facilidad.

### Paso 3. Detectar duplicados

1. Ordena por identificador de partido.
2. Aplica formato condicional para valores duplicados.
3. No elimines inmediatamente: confirma que las filas representan el mismo partido.
4. Registra cuántos duplicados encontraste.

### Paso 4. Normalizar textos

Aplica funciones como:

- `ESPACIOS()` para quitar espacios sobrantes.
- `LIMPIAR()` para eliminar caracteres no imprimibles.
- `MAYUSC()` o `MINUSC()` para crear claves de comparación.
- `SUSTITUIR()` para reemplazar separadores.

Ejemplo:

```text
=ESPACIOS(LIMPIAR(A2))
```

Usa `02_Catalogo_Equipos` junto con `BUSCARX()` para convertir variantes como `USA`, `Korea Republic` o `Cabo Verde` en un nombre canónico.

### Paso 5. Separar el marcador

Los marcadores utilizan `-`, `–`, `—`, `:`, o `x`.

Puedes usar:

- **Datos > Texto en columnas**, después de normalizar el separador.
- `SUSTITUIR()` para convertir todos los símbolos a `-`.
- `TEXTOANTES()` y `TEXTODESPUES()` en versiones recientes de Excel.

Ejemplo conceptual:

```text
Marcador original: 2 x 1
Marcador normalizado: 2-1
Goles local: 2
Goles visitante: 1
```

### Paso 6. Corregir tipos de datos

Verifica que:

- `mundial`, `jornada`, `goles_local` y `goles_visitante` sean números;
- `fecha` sea una fecha real;
- los indicadores de anfitrión sean valores lógicos;
- los nombres de equipos sean texto limpio.

Un valor con apariencia numérica puede seguir almacenado como texto.

### Paso 7. Resolver conflictos

La fuente didáctica incluye goles vacíos, números negativos y marcadores con distintos formatos.

Regla propuesta:

1. Si el marcador puede interpretarse, úsalo para reparar los goles.
2. Si el marcador está vacío, utiliza los goles separados.
3. Si ambos existen y no coinciden, documenta el conflicto antes de elegir.
4. Ningún partido válido puede tener goles negativos.

### Paso 8. Crear la base integrada

Copia los campos limpios en `07_Base_Limpia`.

La tabla final debe contener una fila por partido y conservar:

- identificador;
- mundial;
- grupo;
- jornada;
- fecha;
- equipo local y visitante;
- goles;
- condición de anfitrión;
- fuente.

Las columnas derivadas del libro calcularán marcador, resultado, goles totales, diferencia y puntos.

## Parte B. Comparación en Excel

Completa la hoja `08_Comparacion`.

Analiza:

- partidos por torneo;
- goles totales;
- goles por partido;
- porcentaje de empates;
- porcentaje de victorias del anfitrión;
- equipo con mejor diferencia de goles;
- proporción de partidos con más de 2.5 goles.

Pregunta central:

> ¿Cambian las conclusiones cuando se comparan totales en lugar de tasas?
Sí. Comparando totales, 2026 parece muy superior en goles y empates. Comparando tasas (goles/partido, % empates), las diferencias entre los tres mundiales son moderadas. La conclusión correcta es que 2026 tiene ligeramente más goles por partido y más empates proporcionalmente, no una diferencia dramática como sugieren los totales crudos.

Crea al menos dos gráficos:

- goles totales y goles por partido;
- distribución de resultados por mundial.

## Parte C. Reproducir la limpieza en Python

Abre `notebooks/01_limpieza_analisis_alumnos.ipynb`.

Ejecuta cada sección en orden. No copies directamente la solución.

Debes poder explicar:

- qué problema resuelve cada transformación;
- por qué se utiliza un catálogo;
- cómo se detectan duplicados;
- por qué la fecha serial de Excel requiere un origen;
- qué validaciones aseguran que la base quedó limpia.

## Parte D. Entrada al análisis supervisado

El objetivo será predecir el resultado del equipo local:

- `Gana`
- `Empata`
- `Pierde`

Solo pueden utilizarse variables conocidas antes del partido, por ejemplo:

- puntos promedio previos;
- diferencia de goles promedio previa;
- goles promedio previos;
- jornada;
- condición de anfitrión.

No deben utilizarse los goles finales del mismo partido como entrada. Eso produciría fuga de información.

Entrena con 2018 y 2022 y prueba con 2026.

La meta no es obtener una precisión perfecta. La meta es comprender:

- la diferencia entre variables predictoras y variable objetivo;
- la separación entrenamiento/prueba;
- la línea base;
- la matriz de confusión;
- las limitaciones de un modelo pequeño.
