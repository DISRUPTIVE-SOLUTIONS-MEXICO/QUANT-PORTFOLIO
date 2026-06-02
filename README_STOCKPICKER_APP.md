# Quant Portfolio-Kaizen

Streamlit app para ejecutar un pipeline causal de stock picking:

1. Regimen macro: tasas, curva 10Y-2Y, credit premium, real yield proxy.
2. Panel fundamental con `Availability_Date <= signal_date`, usando SEC EDGAR `companyfacts` y `acceptedDateTime` cuando hay cobertura.
3. Normalizacion sectorial robusta y distancia de Mahalanobis.
4. Filtro fundamental por calidad, valor, solvencia y cobertura de intereses.
5. Optimizacion exhaustiva de chunks por Sortino.
6. Backtest walk-forward con embargo y costes de transaccion.
7. Cache parquet local con TTL para precios, volumen, macro, universos y fundamentales.
8. Universos publicos: Wikipedia S&P 500 as-of aproximado, NasdaqTrader, SEC company tickers y CSV locales.
9. Curvas soberanas modulares: FRED/OECD proxy, ECB Data Portal, BCB SGS, Bank of Canada Valet y Banxico SIE con token opcional.
10. Opciones costo 0: snapshot de Yahoo con IV ATM, skew, DTE, put/call open interest y bid/ask.
11. Validacion gratis: moving block bootstrap, stationary bootstrap, rolling IC, ICIR y rank stability.
12. Datos alternativos: FRED macro/liquidez/credito/volatilidad y GDELT geopolitico/news opcional.
13. XBRL PIT mas fino: `SEC_FY`, `SEC_FP`, `SEC_Frame`, `SEC_Period_Type`, `SEC_Accepted_At`.
14. Deflated Sortino y CPCV/PBO con matriz IS/OOS por trial para penalizar multiple testing.
15. Robust optimization con penalizacion por incertidumbre de alpha y covarianza.
16. Seleccion robusta IS/OOS: el chunk puede penalizarse o bonificarse por persistencia historica OOS del trial.
17. Model confidence throttle: reduce alpha efectivo y sube aversion al riesgo cuando el ranking IS deja de convertir OOS.
18. Regimen latente causal: Gaussian mixture expanding-window con probabilidades, entropia y matriz de transicion para condicionar el decaimiento del alpha.
19. Modelo factorial estructural: `Sigma = B Sigma_f B^T + D`, blend configurable con Ledoit-Wolf, contribucion marginal al riesgo por activo y descomposicion factor/especifica.
20. Posterior bayesiano del alpha: media, desviacion, probabilidad de alpha positivo e intervalo creible para evitar rankings con precision estadistica pobre.
21. Shrinkage bayesiano jerarquico por sector: el alpha posterior se contrae hacia priors sectoriales contemporaneos, no hacia un cero global uniforme.
22. Decision dinamica: exposicion causal ajustada por confianza del modelo, probabilidad/entropia de regimen y cash weight explicito.
23. Attribution OOS: descomposicion periodo a periodo en factor, seleccion especifica y costes de transaccion.
24. Stress testing factorial y performance condicionada por regimen para validar en que estados vive o muere el alpha.
25. Penalizacion CVaR dentro de la construccion de pesos para controlar cola izquierda, no sólo downside deviation.
26. White Reality Check / Hansen SPA sobre trials OOS para atacar data mining y multiple testing.
27. Ledger event-driven con signal/order/fill, capital, cash drag, costes y drawdown path.
28. Restricciones de participacion ADV/Almgren-Chriss proxy dependientes del notional configurado.
29. Factores estilo cross-sectional: value, quality, momentum, low-vol, size y liquidity.
30. Cadena de Markov causal sobre estados latentes: persistencia, probabilidad forward de stress/risk-on y throttle de exposicion por riesgo de transicion.
31. Suitability Engine: horizonte, capital, aportaciones, liquidez, drawdown tolerado, aversion y objetivo se traducen en limites de volatilidad, CVaR, numero de activos, concentracion, sector cap, ADV cap y bloqueos por incoherencia.
32. Benchmark Governance: valida coherencia de benchmark contra mandato, pais, sector, universo y objetivo; sugiere SPY/QQQ/ACWI/VT/EWW/XLK/XLV/XLU/etc. y advierte cuando IR/Treynor no son interpretables.
33. Model Registry reproducible: cada corrida genera `run_hash`, `code_version`, `config_hash`, `universe_hash`, `data_hash`, timestamps de datos, config completa, restricciones, warnings y métricas; se persiste localmente en parquet/json y se envía a Supabase con fallback.
34. Tests de causalidad/no-leakage: suite sintética para `Availability_Date`, contaminación futura, purging, embargo, OOS y estabilidad ante precios posteriores al `asof`.
35. Black-Litterman bayesiano: alpha posterior como views, prior de equilibrio, matriz de incertidumbre Omega por Bayesian std/CRLB/cobertura y posterior `BL_Posterior_Alpha`.
36. HRP: objetivo `hrp` con clustering jerárquico de correlaciones, distancia `sqrt((1-rho)/2)`, quasi-diagonalización y recursive bisection como arranque robusto.
37. SEC NLP 10-K/10-Q: descarga filings públicos, extrae términos de riesgo/litigio/impairment/going concern/covenants/AI, similitud contra filing previo y `TextRisk_Score` causal para penalizar alpha.
38. Kaizen Contextual Bandit: meta-política LinUCB-lite para recomendar hiperparámetros por régimen/perfil/reward OOS; no elige tickers ni pesos directamente y usa promotion gate DSR/PBO/SPA/drawdown.
39. Side Boom Portfolio: sleeve paralelo optimizado por Sortino contra benchmark, con aliases de nombres corporativos (`CEREBRAS -> CBRS`, `MICROSOFT -> MSFT`, `LENOVO -> LNVGY`, `TSMC -> TSM`) y restricción manual de peso fijo para escenarios.

## Ejecutar

```powershell
python -m streamlit run stockpicker_app.py --server.port 8501
```

## Nota PIT

La app evita look-ahead contable usando `Availability_Date`. Para SEC EDGAR, la fecha se toma de `SEC_Accepted_At` cuando existe; para Yahoo Finance se mantiene un lag configurable sobre el cierre fiscal.

## Variables opcionales

```powershell
BANXICO_TOKEN=tu_token_sie
```

Banxico SIE requiere token para su API oficial. Sin token, Mexico usa proxy FRED/OECD; no se hace scraping HTML frágil por defecto.

## Supabase

La app guarda en Supabase las tablas existentes (`runs`, `portfolio_weights`, `backtest_perf`, `risk_diagnostics`, `variance_model_selection`). Las cadenas completas de opciones permanecen en parquet/CSV para evitar inflar Postgres; a Supabase se mandan metricas agregadas como IV ATM, skew, put/call OI, validacion, datos alternativos y resumen del regimen latente.
