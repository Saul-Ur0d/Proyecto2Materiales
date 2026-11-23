# Análisis de aceros al carbono y aleados SAE

Miembros del equipo:

    Meza Prieto Adriel
    
    Uriostegui Rogríguez Adolfo Saúl
    
    Vega Hurtado Mildred Dessire

Pipeline reproducible en **Python** para limpiar, auditar y analizar una base de **190 registros y 27 columnas** con composición química y propiedades mecánicas de aceros SAE. Estudia la relación entre **%C**, resistencia última (UTS), límite de fluencia (YS), dureza Brinell (HB), elongación, condiciones de procesamiento y familias SAE. Los resultados se presentan mediante figuras, tablas y un **dashboard HTML interactivo** publicable en GitHub Pages.

## Estructura

```text
Proyecto2/
├── data/
│   ├── raw/
│   │   └── Steel- Property and Composition (Carbon Steel).csv
│   └── processed/
│       ├── aceros_limpio.csv
│       ├── auditoria_inconsistencias.csv
│       └── tabla_02_*.csv
├── src/
│   ├── 01_limpieza_datos.py
│   ├── 02_analisis_carbono_y_procesos.py
│   ├── 03_recocido_normalizado_y_anova.py
│   ├── 04_carbono_vs_aleados.py
│   └── 05_generar_dashboard_html.py
├── outputs/
│   ├── figuras/                    # PNG de los scripts 02–04
│   ├── tablas/                     # CSV de los scripts 03–04
│   ├── resumen_02_carbono_procesos.txt
│   ├── resumen_03_recocido_normalizado_anova.txt
│   ├── resumen_04_carbono_vs_aleados.txt
│   └── index.html                 # Dashboard generado
├── docs/                          # Copia HTML para GitHub Pages (opcional)
│   └── index.html
├── requirements.txt
└── README.md
```

Las carpetas y archivos de resultados se crean al ejecutar los scripts. `docs/` se genera únicamente cuando se solicita la copia publicable.

## Instalación

Se recomienda **Python 3.10 o posterior**. Desde la raíz `Proyecto2/`, abre la terminal de VS Code y ejecuta en Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las dependencias son `pandas`, `numpy`, `scipy`, `matplotlib` y `seaborn`, sin versiones fijas. El dashboard utiliza **Plotly.js mediante CDN HTTPS**: no requiere instalar Plotly con `pip`, pero sí conexión a internet para cargar las gráficas.

## Orden de ejecución

Coloca el CSV original en `data/raw/` y ejecuta **del 01 al 05**, desde la raíz del repositorio:

```powershell
python src/01_limpieza_datos.py
python src/02_analisis_carbono_y_procesos.py
python src/03_recocido_normalizado_y_anova.py
python src/04_carbono_vs_aleados.py
python src/05_generar_dashboard_html.py
```

| Script | Propósito y salida principal |
| --- | --- |
| **01** | Limpieza y auditoría sin eliminar filas → `data/processed/aceros_limpio.csv` y `auditoria_inconsistencias.csv`. |
| **02** | %C frente a propiedades, comparación de condiciones y pares laminado/estirado → `data/processed/tabla_02_*.csv`, figuras 02 y resumen 02. |
| **03** | Recocido, normalizado, análisis pareado y ANOVA con supuestos → `outputs/tablas/tabla_03_*.csv`, figuras 03 y resumen 03. |
| **04** | Contraste entre 10xx, 11xx–12xx, 13xx–15xx y aleados 3xxx–9xxx → `outputs/tablas/tabla_04_*.csv`, figuras 04 y resumen 04. |
| **05** | Integra datos, tablas y resúmenes disponibles en cinco pestañas con filtros → `outputs/index.html`. |

Los scripts analíticos (02–04) usan **UTS/YS corregidos** por defecto y permiten `--valores originales`. El parámetro `--anomalias` admite `resaltar` (predeterminado), `excluir` e `incluir`. El script 05 también permite configurar esos valores iniciales, sin impedir que el visitante cambie los filtros en el navegador.

## Publicación en GitHub Pages

El HTML en `outputs/index.html` se puede abrir directamente en el navegador. Para publicarlo desde la carpeta `docs/` de GitHub Pages:

1. Genera la copia:

   ```powershell
   python src/05_generar_dashboard_html.py --copiar docs
   ```

2. Sube el proyecto al repositorio de GitHub y **asegúrate de incluir `docs/index.html`**. Si ya tienes el repositorio configurado:

   ```powershell
   git add README.md requirements.txt src/ docs/
   git commit -m "Agregar documentación y dashboard SAE"
   git push origin main
   ```

3. En GitHub, entra a **Settings → Pages → Build and deployment**. Selecciona **Deploy from a branch**, rama **`main`**, carpeta **`/docs`** y guarda los cambios.
También puedes generar `index.html` en la raíz con `--copiar raiz` y elegir `/(root)` en Pages. `outputs/` no es una opción directa de publicación al desplegar desde una rama. Para actualizar el sitio, vuelve a ejecutar el script 05 y sube el nuevo HTML.

El dashboard **incrusta los datos en el propio HTML** y no requiere servidor Python. Antes de publicarlo, verifica que puedes compartir el dataset. Guía oficial: [Configuración de GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Limitaciones metodológicas

- Los registros son **valores nominales de tablas SAE**, no ensayos independientes con réplicas. Los contrastes estadísticos son exploratorios.
- `C_pct` se estima a partir de los rangos Min/Max; **no es una medición de la composición real** de cada colada.
- Hay inconsistencias de captura, escalas y unidades. `flag_anomalia` y `auditoria_inconsistencias.csv` conservan su trazabilidad; solo se compara dureza en la escala **HB** cuando es válida.
- Los tamaños de muestra y las condiciones **no están equilibrados entre familias**. Las diferencias 10xx–aleados pueden confundirse con el tratamiento térmico y el %C; una correlación no prueba causalidad.
- El ANOVA requiere evaluar **independencia, normalidad y homogeneidad de varianzas**. Se prefieren comparaciones del **mismo grado SAE** cuando existen pares; algunos grupos tienen muestras demasiado pequeñas.
- Los filtros del dashboard actualizan gráficos y descriptivos interactivos, pero **no recalculan los ANOVA ni las tablas estadísticas precalculadas** en los scripts 02–04.

Metalúrgicamente, en los 10xx predominan ferrita y perlita según el %C y el enfriamiento; elementos como Cr, Ni, Mo, Mn y Si pueden modificar templabilidad y mecanismos de endurecimiento en los aleados. La microestructura final también depende del procesamiento.
