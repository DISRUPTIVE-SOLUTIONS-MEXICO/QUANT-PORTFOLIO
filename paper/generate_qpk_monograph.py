from __future__ import annotations

import re
import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"
PDF_PATH = OUT_DIR / "Quant_Portfolio_Kaizen_Monografia_Formal.pdf"
TEX_PATH = OUT_DIR / "Quant_Portfolio_Kaizen_Monografia_Formal.tex"


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    bold_candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf"),
    ]
    mono_candidates = [
        Path(r"C:\Windows\Fonts\consola.ttf"),
        Path(r"C:\Windows\Fonts\cour.ttf"),
    ]
    regular = next((p for p in candidates if p.exists()), None)
    bold = next((p for p in bold_candidates if p.exists()), None)
    mono = next((p for p in mono_candidates if p.exists()), None)
    if regular:
        pdfmetrics.registerFont(TTFont("QPK-Regular", str(regular)))
    if bold:
        pdfmetrics.registerFont(TTFont("QPK-Bold", str(bold)))
    if mono:
        pdfmetrics.registerFont(TTFont("QPK-Mono", str(mono)))
    return (
        "QPK-Regular" if regular else "Helvetica",
        "QPK-Bold" if bold else "Helvetica-Bold",
        "QPK-Mono" if mono else "Courier",
    )


REGULAR, BOLD, MONO = register_fonts()


class HR(Flowable):
    def __init__(self, width: float = 6.4 * inch, color=colors.HexColor("#334155")):
        super().__init__()
        self.width = width
        self.height = 0.08 * inch
        self.color = color

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(0.75)
        self.canv.line(0, self.height / 2, self.width, self.height / 2)


def styles():
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "QPKTitle",
            parent=base["Title"],
            fontName=BOLD,
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=16,
        ),
        "Subtitle": ParagraphStyle(
            "QPKSubtitle",
            parent=base["Normal"],
            fontName=REGULAR,
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#334155"),
            spaceAfter=10,
        ),
        "H1": ParagraphStyle(
            "QPKH1",
            parent=base["Heading1"],
            fontName=BOLD,
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=8,
        ),
        "H2": ParagraphStyle(
            "QPKH2",
            parent=base["Heading2"],
            fontName=BOLD,
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=9,
            spaceAfter=6,
        ),
        "Body": ParagraphStyle(
            "QPKBody",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=9.4,
            leading=13.2,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "QPKSmall",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=8.3,
            leading=11,
            textColor=colors.HexColor("#334155"),
            spaceAfter=4,
        ),
        "Formula": ParagraphStyle(
            "QPKFormula",
            parent=base["BodyText"],
            fontName=MONO,
            fontSize=8.6,
            leading=11.4,
            textColor=colors.HexColor("#0f172a"),
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=0.35,
            borderPadding=5,
            leftIndent=8,
            rightIndent=8,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "Box": ParagraphStyle(
            "QPKBox",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor("#0f172a"),
            backColor=colors.HexColor("#eef6ff"),
            borderColor=colors.HexColor("#7dd3fc"),
            borderWidth=0.45,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "Proof": ParagraphStyle(
            "QPKProof",
            parent=base["BodyText"],
            fontName=REGULAR,
            fontSize=8.9,
            leading=12.5,
            textColor=colors.HexColor("#1f2937"),
            leftIndent=12,
            rightIndent=8,
            spaceAfter=6,
        ),
        "Code": ParagraphStyle(
            "QPKCode",
            parent=base["Code"],
            fontName=MONO,
            fontSize=7.8,
            leading=10,
            textColor=colors.HexColor("#0f172a"),
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.35,
            borderPadding=5,
        ),
    }


S = styles()


def p(text: str):
    return Paragraph(text, S["Body"])


def small(text: str):
    return Paragraph(text, S["Small"])


def h1(text: str):
    return Paragraph(text, S["H1"])


def h2(text: str):
    return Paragraph(text, S["H2"])


def formula(text: str):
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Preformatted(safe, S["Formula"])


def box(text: str):
    return Paragraph(text, S["Box"])


def proof(text: str):
    return Paragraph("<b>Demostración.</b> " + html.escape(text), S["Proof"])


def bullet(items: list[str]):
    return ListFlowable(
        [ListItem(Paragraph(x, S["Body"]), leftIndent=12) for x in items],
        bulletType="bullet",
        leftIndent=18,
    )


def table(data: list[list[str]], widths: list[float] | None = None):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), BOLD),
                ("FONTNAME", (0, 1), (-1, -1), REGULAR),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("LEADING", (0, 0), (-1, -1), 9.2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def section(title: str, body: list):
    return [PageBreak(), h1(title), HR(), Spacer(1, 4)] + body


def theorem_block(name: str, statement: str, proof_text: str):
    return KeepTogether([box(f"<b>{html.escape(name)}.</b> {html.escape(statement)}"), proof(proof_text)])


def content_sections() -> list[tuple[str, list]]:
    common_objective = (
        "El framework no asume que el alpha sea una constante observable; lo trata como una variable latente "
        "estimada bajo una filtración. La implementación debe separar estrictamente lo conocido en la fecha de "
        "señal de lo observado después de ejecutar el portafolio. Esta separación es la condición mínima para que "
        "una métrica de backtest tenga interpretación económica y no sea sólo una estadística contaminada."
    )
    sections: list[tuple[str, list]] = []

    sections.append(
        (
            "1. Resumen ejecutivo y tesis central",
            [
                p(
                    "Quant Portfolio-Kaizen es un framework de stock picking y asset allocation long-only, costo-cero, "
                    "diseñado para convertir datos públicos imperfectos en decisiones de portafolio auditables. Su tesis "
                    "central es que el edge no proviene de maximizar agresivamente una señal, sino de reducir incertidumbre "
                    "antes de asignar capital. La arquitectura combina filtros fundamentales sectoriales, inferencia de régimen, "
                    "benchmark governance, control robusto de downside, validación purgada y persistencia cloud por artifacts."
                ),
                formula(
                    "PIT Data -> Signal Reliability -> UncertaintyState -> Robust Control\n"
                    "-> Nested Walk-Forward -> Promotion Gate -> Dashboard Artifact"
                ),
                p(
                    "El sistema no promete dominancia universal sobre todo benchmark. La meta matemáticamente defendible es "
                    "elegir un benchmark óptimo de mandato, denotado xi, y buscar active return positivo con menor captura de "
                    "downside. El conjunto Omega de benchmarks se usa como stress set, no como un único oponente lineal."
                ),
                formula(
                    "E[R_p - R_xi] > 0,   UC_p > 1,   DC_p < 1,\n"
                    "DD_p <= DD_xi,       CVaR_p <= CVaR_xi,       D_-(p) <= D_-(xi)."
                ),
                box(
                    "Dictamen metodológico: la app debe comunicar tres estados, no sólo retornos: APPROVED, RESEARCH-ONLY "
                    "y BLOCKED. Un portafolio puede tener buen retorno y aun así quedar bloqueado si falla suitability, "
                    "WRC, SPA, PBO o preservación de downside."
                ),
            ],
        )
    )

    sections.append(
        (
            "2. Notación, filtración y espacio de decisión",
            [
                p(common_objective),
                formula(
                    "U_t = universe as-of approximation\n"
                    "P_{i,t} = adjusted price, V_{i,t} = volume\n"
                    "r_{i,t} = P_{i,t}/P_{i,t-1} - 1\n"
                    "F_t = sigma(P_{i,s},V_{i,s},M_s,FUND_{i,tau},O_{i,s}^{snap},N_s : s<=t, tau<=t)"
                ),
                p(
                    "La política de portafolio es una función medible respecto a la filtración. En términos de ingeniería "
                    "financiera, esto significa que cada feature usada por el optimizador debe tener una fecha de disponibilidad "
                    "menor o igual a la fecha de señal."
                ),
                formula(
                    "pi_t : F_t -> Delta_N,\n"
                    "Delta_N = { w in R^N : w_i >= 0, 1'w = 1 },\n"
                    "R_{p,t+1} = w_t' r_{t+1} - TC_t."
                ),
                theorem_block(
                    "Proposición 2.1 (Causalidad de la política)",
                    "Si todos los inputs de pi_t son F_t-medibles, entonces w_t es F_t-medible y el retorno OOS R_{p,t+1} no contamina la decisión.",
                    "Por definición, una composición finita de funciones medibles es medible. Si cada feature pertenece a F_t y el solver es determinístico dado el estado, entonces w_t=pi_t(F_t) pertenece a F_t. Como r_{t+1} no pertenece a F_t, sólo afecta la evaluación posterior, no la selección. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "3. Data generating process y sentido físico-estocástico",
            [
                p(
                    "El proceso de precios se representa como semimartingala con difusión, volatilidad condicional y jumps. "
                    "La app no estima un modelo estructural completo de equilibrio general; usa esta formulación como mapa "
                    "conceptual para ubicar drift, varianza, discontinuidades y régimen."
                ),
                formula(
                    "dS_{i,t}/S_{i,t-} = mu_{i,t} dt + sigma_{i,t} dW_{i,t} + J_{i,t} dN_{i,t},\n"
                    "z_t in {risk-on, inflation shock, credit stress, liquidity crisis, recovery}."
                ),
                p(
                    "La presencia de variación cuadrática finita justifica modelar varianza realizada y volatilidad condicional. "
                    "Los jumps y cambios de régimen justifican PELT, EVT y penalizaciones por cola. El drift mu_t es de baja señal-ruido; por eso se aplica shrinkage bayesiano y CRLB."
                ),
                formula(
                    "[log S]_t = integral_0^t sigma_s^2 ds + sum_{0<s<=t} (Delta log S_s)^2."
                ),
                theorem_block(
                    "Lema 3.1 (Separación drift-volatilidad)",
                    "En horizontes cortos, el error de estimación de mu domina al error de estimación de sigma bajo baja SNR.",
                    "Si r_t ~ N(mu, sigma^2), Var(mu_hat)=sigma^2/T. Para acciones diarias, |mu| es pequeño frente a sigma. Por tanto, el ratio |mu_hat|/sqrt(Var(mu_hat)) crece lentamente con T. La varianza condicional, aunque ruidosa, exhibe clustering y puede estimarse con más estabilidad relativa. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "4. Fuentes costo-cero y aproximación point-in-time",
            [
                p(
                    "La restricción costo-cero obliga a usar yfinance, SEC EDGAR, FRED, Banxico, ECB, BCB, BoC, ForexFactory/FairEconomy, "
                    "GDELT, Google News RSS y snapshots de opciones de Yahoo. Ninguna de estas fuentes equivale a Bloomberg/FactSet PIT institucional."
                ),
                table(
                    [
                        ["Fuente", "Uso", "Riesgo", "Mitigación"],
                        ["yfinance", "Precios, volumen, fundamentals, options snapshot", "Campos cambiantes y no PIT estricto", "Cache, validación de nulos, PIT confidence"],
                        ["SEC EDGAR", "Companyfacts, filing dates, accepted timestamps", "Cobertura compleja por taxonomía", "Availability_Date y accounting lag"],
                        ["FRED/Banxico/ECB/BCB/BoC", "Curvas, tasas, macro", "Frecuencia heterogénea", "Sampling discreto por fecha de observación"],
                        ["GDELT/RSS", "Atención geopolítica", "Query bias y comparabilidad cruda", "Z robusto within-topic y fallback cualitativo"],
                        ["Yahoo options", "IV ATM, skew, bid/ask, OI", "Sin histórico causal de opciones", "Snapshot sólo diagnóstico contemporáneo"],
                    ],
                    widths=[1.15 * inch, 1.4 * inch, 1.55 * inch, 1.75 * inch],
                ),
                formula(
                    "PITConfidence_i = SourceConfidence_i * exp(-StalenessDays_i/540) * Coverage_i."
                ),
                p(
                    "Todo ratio derivado de Yahoo debe etiquetarse como PIT approximation. SEC EDGAR mejora causalidad porque la fecha de filing y accepted timestamp permiten bloquear datos que aún no estaban disponibles."
                ),
            ],
        )
    )

    ratios = [
        ("ROIC", "NOPAT / Invested Capital", "Calidad de asignación de capital."),
        ("EV/EBITDA", "(MarketCap + Debt - Cash)/EBITDA", "Valoración neutral a estructura de capital."),
        ("FCF Yield", "FreeCashFlow / MarketCap", "Caja real por unidad de valor de mercado."),
        ("Net Debt / EBITDA", "(Debt - Cash)/EBITDA", "Apalancamiento operativo-crediticio."),
        ("Piotroski F-Score", "sum_{k=1}^9 1_{criterion_k}", "Salud financiera discreta."),
        ("Asset Turnover", "Revenue / AvgAssets", "Eficiencia operativa."),
        ("Altman Z", "1.2X1+1.4X2+3.3X3+0.6X4+1.0X5", "Distress probability proxy."),
        ("Interest Coverage", "EBIT / InterestExpense", "Resiliencia a tasas altas."),
        ("Retention Ratio", "1 - Dividends/NetIncome", "Motor endógeno de crecimiento."),
        ("Earnings Yield", "E/P", "Comparación contra tasa libre de riesgo."),
        ("P/B", "Price / BookValuePerShare", "Value sectorial."),
        ("ROE", "NetIncome / Equity", "Rentabilidad patrimonial con riesgo de leverage bias."),
    ]
    sections.append(
        (
            "5. Filtro fundamental sectorial",
            [
                p(
                    "La regla central es no comparar peras con manzanas: un P/E de software no tiene la misma distribución "
                    "que un P/E de utilities. Por tanto, cada ratio se normaliza dentro de sector y, cuando la cobertura es baja, se aplica winsorization y score de confianza."
                ),
                table([["Ratio", "Definición", "Interpretación"]] + [list(x) for x in ratios], widths=[1.2 * inch, 2.0 * inch, 2.4 * inch]),
                formula(
                    "Z_{i,k}^{sector} = (x_{i,k} - median_{j:g(j)=g(i)} x_{j,k}) /\n"
                    "                  (1.4826 * median_j |x_{j,k} - median(x_{g,k})| + eps)."
                ),
                theorem_block(
                    "Proposición 5.1 (Validez del z-score robusto sectorial)",
                    "Si la distribución del ratio dentro de sector es unimodal con contaminación acotada, el z-score basado en mediana y MAD tiene breakdown point superior al z-score clásico.",
                    "La media y desviación estándar tienen breakdown point cero: un outlier arbitrario puede desplazar ambos estimadores. La mediana y MAD toleran hasta 50% de contaminación antes de divergir. El factor 1.4826 calibra el MAD a sigma bajo normalidad. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "6. Mahalanobis sectorial y anomalías multivariadas",
            [
                p(
                    "El vector fundamental de una empresa no debe evaluarse ratio por ratio de forma independiente. La distancia de Mahalanobis mide qué tan lejos está una compañía del centroide robusto de su sector considerando correlaciones entre ratios."
                ),
                formula(
                    "x_i = (Z_{i,1},...,Z_{i,K})',\n"
                    "D_M(i)^2 = (x_i - m_g)' S_g^{-1} (x_i - m_g)."
                ),
                p(
                    "Una distancia alta puede ser oportunidad o value trap. La decisión depende del signo del alpha posterior, cobertura fundamental, liquidez y riesgo textual SEC."
                ),
                theorem_block(
                    "Proposición 6.1 (Invariancia afín de Mahalanobis)",
                    "Para una transformación no singular y=Ax+b, la distancia de Mahalanobis es invariante si la covarianza se transforma como ASA'.",
                    "Sustituyendo y_i-m_y=A(x_i-m_x) y S_y^{-1}=(A')^{-1}S_x^{-1}A^{-1}, se obtiene (y_i-m_y)'S_y^{-1}(y_i-m_y)=(x_i-m_x)'S_x^{-1}(x_i-m_x). QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "7. Régimen macro, curvas de tasas y clasificación hawkish/dovish",
            [
                p(
                    "El régimen se estima usando curvas soberanas, inflación, crédito, liquidez, momentum de benchmark, pendiente 10Y-2Y y variables alternativas. Las tasas son procesos discretos de observación, no trayectorias continuas: la visualización debe respetar fechas reales y frecuencias heterogéneas."
                ),
                formula(
                    "CurveSlope_t = y_{10Y,t} - y_{2Y,t},\n"
                    "PolicyImpulse_t = Delta PolicyRate_t + Delta Inflation_t,\n"
                    "CreditStress_t = z(BAA_t - AAA_t)."
                ),
                p(
                    "La etiqueta hawkish/dovish se deriva de presión de política monetaria; bullish/bearish de price action, amplitud, volatilidad y crédito. La clasificación no debe confundirse con predicción perfecta de retornos."
                ),
                table(
                    [
                        ["Estado", "Condición típica", "Efecto esperado"],
                        ["Dovish Bull", "curva relajando + momentum positivo", "growth y duración reciben soporte"],
                        ["Hawkish Bull", "momentum positivo + tasas altas", "quality/cash-flow domina speculative growth"],
                        ["Dovish Bear", "relajación por estrés", "riesgo de recesión; defensive sleeve sube"],
                        ["Hawkish Bear", "tasas restrictivas + drawdown", "reducir beta, CVaR y concentración"],
                    ],
                    widths=[1.2 * inch, 2.2 * inch, 2.4 * inch],
                ),
            ],
        )
    )

    sections.append(
        (
            "8. Construcción del benchmark óptimo xi",
            [
                p(
                    "El benchmark xi no se elige por conveniencia. Se estima como el benchmark que mejor representa el mandato del portafolio. Omega es el conjunto de candidatos: SPY, QQQ, ACWI, VT, VTI, IWM, USMV, SPLV, MTUM, QUAL, VLUE y ETFs sectoriales."
                ),
                formula(
                    "Omega = {omega_1,...,omega_K},\n"
                    "xi* = argmax_{xi in Omega} Fit(p,xi)."
                ),
                formula(
                    "Fit(p,xi) = a1 Corr(R_p,R_xi) - a2 |Beta_{p,xi}-1|\n"
                    "            - a3 TrackingError(p,xi) + a4 SectorOverlap(p,xi)\n"
                    "            + a5 FactorOverlap(p,xi) + a6 MandateMatch(p,xi)."
                ),
                p(
                    "Para un mandato absoluto defensivo, xi puede ser USMV/SPLV; para tecnología growth puede ser QQQ/XLK; para global puede ser ACWI/VT. Comparar todo contra QQQ sin mandato growth tecnológico es benchmark mismatch."
                ),
                theorem_block(
                    "Proposición 8.1 (Gobernanza de benchmark)",
                    "El Information Ratio y Treynor son métricas interpretables sólo si xi representa el riesgo sistemático del mandato.",
                    "IR divide active return por tracking error respecto a xi. Si xi no comparte universo, país, sector o factor dominante, el tracking error mide desalineación estructural, no skill. Treynor usa beta contra xi; un beta mal definido invalida la razón. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "9. Alpha bayesiano, Fisher information y CRLB",
            [
                p(
                    "El alpha score se modela como estimador con incertidumbre. No basta ordenar activos por Composite_Score; debe penalizarse el error estándar posterior, la baja cobertura fundamental y el bajo tamaño muestral efectivo."
                ),
                formula(
                    "alpha_i | F_t ~ N(alpha_hat_i, tau_i^2),\n"
                    "I(mu_i) = T_i / sigma_i^2,\n"
                    "CRLB(mu_hat_i) >= sigma_i^2 / T_i."
                ),
                formula(
                    "mu_i^{robust} = alpha_hat_i * |alpha_hat_i| / (|alpha_hat_i| + sqrt(CRLB_i) + eps)."
                ),
                theorem_block(
                    "Proposición 9.1 (Shrinkage por CRLB)",
                    "Si CRLB aumenta manteniendo alpha_hat fijo, el alpha robusto disminuye monótonamente en magnitud.",
                    "Sea f(c)=a|a|/(|a|+sqrt(c)). Para c>0, df/dc tiene signo opuesto a a y magnitud positiva en el denominador, por tanto |f(c)| decrece. QED.",
                ),
                p(
                    "La información de Fisher es el puente entre estadística y física: cuantifica curvatura local de la log-verosimilitud. Baja curvatura implica parámetros poco identificables y, por tanto, menor capital asignable."
                ),
            ],
        )
    )

    sections.append(
        (
            "10. Entropía de Shannon y gobierno de concentración",
            [
                p(
                    "La entropía controla tanto la concentración de pesos como la discriminación del ranking. Una señal con softmax plano tiene alta incertidumbre transversal; una cartera con un solo peso dominante tiene fragilidad idiosincrática."
                ),
                formula(
                    "H(w) = - sum_i w_i log(w_i),\n"
                    "H_N(w) = H(w) / log(N),\n"
                    "p_i = exp(s_i/tau) / sum_j exp(s_j/tau)."
                ),
                p(
                    "El sistema usa entropía como governor, no como fin absoluto: se evita concentración espuria, pero no se obliga a igual ponderación si existe evidencia causal y robusta."
                ),
                theorem_block(
                    "Lema 10.1 (Cotas de entropía normalizada)",
                    "Para w en Delta_N, H_N(w) pertenece a [0,1].",
                    "Por concavidad de -x log x, la entropía mínima ocurre en vértice del simplex y vale 0; la máxima ocurre en w_i=1/N y vale log N. Dividir por log N da la cota. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "11. Covarianza, Ledoit-Wolf y Random Matrix Theory",
            [
                p(
                    "La matriz de covarianza muestral es inestable cuando N/T no es despreciable. El framework usa shrinkage y limpieza espectral inspirada en Marchenko-Pastur para separar modos informativos de ruido."
                ),
                formula(
                    "lambda_pm = sigma^2 (1 +- sqrt(q))^2,   q=N/T.\n"
                    "Sigma_RMT = V diag(lambda_clean) V',\n"
                    "lambda_clean_j = lambda_j if lambda_j > lambda_+ else mean(lambda_noise)."
                ),
                theorem_block(
                    "Proposición 11.1 (PSD de limpieza espectral)",
                    "Si todos los eigenvalores limpiados son no negativos, Sigma_RMT es semidefinida positiva.",
                    "Para cualquier x, x'V diag(lambda_clean)V'x = y'diag(lambda_clean)y = sum_j lambda_j^clean y_j^2 >= 0. QED.",
                ),
                p(
                    "El RMT crowding penalty se interpreta físicamente: cuando pocos modos explican demasiada varianza, el mercado está crowded y la diversificación aparente puede colapsar."
                ),
                formula(
                    "EffectiveRank(Sigma) = exp(- sum_j p_j log p_j),   p_j=lambda_j/sum_k lambda_k."
                ),
            ],
        )
    )

    sections.append(
        (
            "12. Arquitecturas de varianza: EWMA, ARCH, GARCH, EGARCH y Volterra",
            [
                p(
                    "La app selecciona arquitectura de varianza por AIC, BIC y log-likelihood, pero la promoción exige mejora OOS en QLIKE. AIC/BIC in-sample no bastan."
                ),
                formula(
                    "EWMA:     sigma_t^2 = lambda sigma_{t-1}^2 + (1-lambda) eps_{t-1}^2\n"
                    "GARCH:    h_t = omega + alpha eps_{t-1}^2 + beta h_{t-1}\n"
                    "EGARCH:   log h_t = omega + beta log h_{t-1} + alpha(|z_{t-1}|-E|z|)+gamma z_{t-1}"
                ),
                formula(
                    "AIC = 2k - 2 log L,   BIC = k log T - 2 log L,\n"
                    "QLIKE(h,r) = r_t^2/h_t + log h_t."
                ),
                p(
                    "El fractional Volterra kernel es una extensión research-only para memoria rough en volatilidad. Debe ser causal: K(t,s)=0 para s>t."
                ),
                formula(
                    "sigma_t^2 = sigma_0^2 + integral_0^t K_H(t-s) b(sigma_s) ds\n"
                    "            + integral_0^t K_H(t-s) nu(sigma_s) dW_s,\n"
                    "K_H(u) = u^{H-1/2}/Gamma(H+1/2), u>0."
                ),
            ],
        )
    )

    sections.append(
        (
            "13. PELT y cambios de régimen",
            [
                p(
                    "PELT detecta cambios estructurales en retornos, volatilidad realizada, drawdown y tracking error. Su uso correcto es diagnóstico y throttle de estado, no excusa para ajustar retrospectivamente el portafolio."
                ),
                formula(
                    "min_{m,tau_1,...,tau_m} sum_{j=0}^{m} C(y_{tau_j+1:tau_{j+1}}) + beta m."
                ),
                p(
                    "El costo C puede ser negativo log-likelihood gaussiana por segmento, o suma de cuadrados centrada. La penalización beta evita sobresegmentación."
                ),
                theorem_block(
                    "Proposición 13.1 (Causalidad operativa de PELT)",
                    "Si PELT se recalibra únicamente con datos hasta t, el régimen z_t es F_t-medible.",
                    "El algoritmo puede usar todo el historial hasta t, pero no fechas posteriores. La segmentación final hasta t es función determinística de y_{<=t}; por tanto pertenece a F_t. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "14. Objetivos clásicos: Sharpe, Sortino, Treynor e Information Ratio",
            [
                p(
                    "La app permite varios objetivos, pero cada uno tiene dominio de validez. Sortino penaliza downside; Sharpe penaliza varianza total; Treynor requiere beta confiable; Information Ratio requiere benchmark coherente."
                ),
                formula(
                    "Sharpe = E[R_p-r_f] / sigma_p,\n"
                    "Sortino = E[R_p-r_f] / sqrt(E[min(R_p-r_f,0)^2]),\n"
                    "Treynor = E[R_p-r_f] / beta_{p,xi},\n"
                    "IR = E[R_p-R_xi] / sigma(R_p-R_xi)."
                ),
                theorem_block(
                    "Lema 14.1 (Inestabilidad del Sortino puro)",
                    "Si el número de observaciones negativas es pequeño, el estimador de downside deviation tiene alta varianza.",
                    "El estimador depende sólo de la submuestra negativa. Cuando esa submuestra es escasa, pequeñas perturbaciones cambian mucho la raíz de la media de cuadrados negativos. Por eso se requiere shrinkage, CVaR y nested validation. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "15. CVaR, drawdown y downside pathwise",
            [
                p(
                    "El framework prioriza riesgo de cola y pérdida pathwise sobre volatilidad total. Para usuarios no financieros, drawdown y CVaR son más interpretables que varianza."
                ),
                formula(
                    "VaR_alpha(L) = inf{x : P(L <= x) >= alpha},\n"
                    "CVaR_alpha(L) = E[L | L >= VaR_alpha(L)]."
                ),
                formula(
                    "NAV_t = NAV_{t-1}(1 + w_{tau(t)}'r_t - TC_t),\n"
                    "DD_t = NAV_t / max_{s<=t} NAV_s - 1."
                ),
                theorem_block(
                    "Proposición 15.1 (Convexidad de CVaR empírico)",
                    "La formulación de Rockafellar-Uryasev de CVaR empírico es convexa en w para pérdidas lineales.",
                    "CVaR_alpha(L_w)=min_zeta zeta + c sum_t (L_{w,t}-zeta)^+. L_{w,t} es lineal en w y la función parte positiva es convexa; suma y minimización parcial preservan convexidad. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "16. Optimización robusta y restricciones institucionales",
            [
                p(
                    "El problema productivo no es maximizar retorno esperado crudo. Es construir pesos en el simplex bajo límites de concentración, sector, liquidez, factor beta, ADV, CVaR y suitability."
                ),
                formula(
                    "max_{w in Delta_N}  U(w;mu,Sigma) - lambda_TC TC(w) - lambda_H(1-H_N(w))\n"
                    "s.t. w_i <= w_max, sector_g(w)<=cap_g, CVaR(w)<=c, DD(w)<=d,\n"
                    "     |beta_factor(w)| <= beta_cap, |Q_i|/ADV_i <= theta."
                ),
                formula(
                    "Robust: max_w min_{mu in U_mu, Sigma in U_Sigma} U(w;mu,Sigma),\n"
                    "U_mu = {mu : (mu-mu_hat)'Omega^{-1}(mu-mu_hat) <= epsilon}."
                ),
                p(
                    "La solución SLSQP/multistart es aceptable para prototipo, pero todo resultado debe pasar validación OOS. HRP/HERC se usan cuando mu es demasiado incierta."
                ),
            ],
        )
    )

    sections.append(
        (
            "17. Black-Litterman bayesiano",
            [
                p(
                    "Black-Litterman estabiliza alpha combinando un prior de equilibrio con views del motor bayesiano. Es especialmente útil cuando el alpha cross-sectional tiene baja información."
                ),
                formula(
                    "mu_BL = [(tau Sigma)^(-1) + P' Omega^(-1) P]^(-1)\n"
                    "        [(tau Sigma)^(-1) pi + P' Omega^(-1) q]."
                ),
                p(
                    "pi representa retornos implícitos de equilibrio; q viene de Bayesian_Alpha_Mean; Omega se escala con Bayesian_Alpha_Std, CRLB y cobertura fundamental. P puede representar views absolutas por activo o relativas sectoriales."
                ),
                theorem_block(
                    "Proposición 17.1 (BL como precisión ponderada)",
                    "El posterior BL pondera prior y views por sus matrices de precisión.",
                    "La fórmula es la media posterior normal-normal multivariada. Los términos (tau Sigma)^(-1) y P'Omega^(-1)P son precisiones. Mayor incertidumbre reduce peso informacional. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "18. Hierarchical Risk Parity y HERC",
            [
                p(
                    "HRP evita inversión directa de una covarianza inestable. Construye clusters por correlación, ordena activos por árbol jerárquico y asigna riesgo recursivamente."
                ),
                formula(
                    "d_{ij} = sqrt((1-rho_{ij})/2),\n"
                    "cluster tree -> quasi diagonalization -> recursive bisection."
                ),
                p(
                    "HRP no garantiza alpha superior, pero es una defensa robusta cuando el vector mu no es confiable. En el framework se usa como candidato alternativo y benchmark interno de construcción."
                ),
            ],
        )
    )

    sections.append(
        (
            "19. XCDR-v2 y XODR: ratio propietario",
            [
                p(
                    "XCDR/XODR busca capturar upside relativo sin relajar downside. El objetivo no es superar el retorno máximo de Omega en todo período, sino dominar al benchmark xi y permanecer robusto contra Omega."
                ),
                formula(
                    "XCDR(w) = [w'mu_robust - mu_xi] /\n"
                    "sqrt(D_-^2(w)+lambda_C CVaR^2(w)+lambda_D DD^2(w)+lambda_R w'Sigma_RMT w + lambda_U w'U w)\n"
                    "          - lambda_T TO(w) + lambda_H H_N(w)."
                ),
                formula(
                    "Omega robust boundary:\n"
                    "D_Omega^robust = Q_q({D(omega): omega in Omega}),\n"
                    "U_Omega^robust = Q_tau({U(omega): omega in Omega})."
                ),
                formula(
                    "Penalty_Omega(w) = lambda_1[D_-(w)-D_Omega]_+^2 + lambda_2[CVaR(w)-CVaR_Omega]_+^2\n"
                    "                 + lambda_3[DD(w)-DD_Omega]_+^2."
                ),
            ],
        )
    )

    sections.append(
        (
            "20. Upside capture, downside capture y dominancia asimétrica",
            [
                p(
                    "El par UC/DC mide convexidad empírica respecto al benchmark. UC mayor a uno y DC menor a uno expresa mejor participación en días positivos que en días negativos."
                ),
                formula(
                    "UC_p = E[R_p | R_xi>0] / E[R_xi | R_xi>0],\n"
                    "DC_p = E[R_p | R_xi<0] / E[R_xi | R_xi<0]."
                ),
                p(
                    "Como el denominador de DC es negativo, la implementación debe cuidar signos. En reporting se usa convención de captura de pérdida positiva cuando el benchmark cae."
                ),
                theorem_block(
                    "Proposición 20.1 (UC/DC no implica dominancia estocástica)",
                    "UC>1 y DC<1 no garantizan dominancia estocástica de primer orden.",
                    "Las condiciones son momentos condicionales sobre dos particiones del benchmark. No ordenan toda la función de distribución acumulada ni controlan colas extremas fuera del promedio condicional. Por eso se agregan CVaR, DD y WRC/SPA. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "21. Downside Preserving Growth Policy",
            [
                p(
                    "DPG separa varianza positiva de downside. La idea es permitir growth si el retorno esperado OOS mejora y si DD, CVaR y downside no se deterioran frente al baseline defensivo."
                ),
                formula(
                    "w_t = alpha_t w_capital,t + beta_t w_growth,t + gamma_t w_alpha,t,\n"
                    "alpha_t+beta_t+gamma_t=1."
                ),
                formula(
                    "max_w [E(R_p-R_xi)]/[D_-(w)+lambda_C CVaR(w)+lambda_D DD(w)+lambda_U U(w)]\n"
                    "      + lambda_+ sigma_+(w)."
                ),
                p(
                    "La política cae a capital preservation si ICIR<=0, si downside capture >=1, o si validation detecta breach de CVaR/DD superior al umbral pre-registrado."
                ),
            ],
        )
    )

    sections.append(
        (
            "22. TailAwareGrowthSleeve y ConvexOpportunityUniverseBuilder",
            [
                p(
                    "El research mostró que el overlay protege downside, pero el alpha faltante está antes: en el universo de oportunidades y el growth sleeve. Por eso se introduce un builder que filtra activos con mayor convexidad upside real."
                ),
                formula(
                    "S_i = a1 M_i + a2 alpha_i + a3 Q_i + a4(beta_i^+ - beta_i^-)\n"
                    "      - a5 TailBeta_i - a6 MCVaR_i - a7 DDDepth_i - a8 Crowding_i."
                ),
                formula(
                    "beta_i^+ = E[r_i | r_xi>0]/E[r_xi | r_xi>0],\n"
                    "beta_i^- = E[r_i | r_xi<0]/E[r_xi | r_xi<0],\n"
                    "TailBeta_i = E[r_i | r_xi < q_alpha(r_xi)] / E[r_xi | r_xi < q_alpha(r_xi)]."
                ),
                p(
                    "Señales permitidas: residual momentum vs xi, momentum sectorial, quality/growth sectorial, ROIC, FCF yield, revenue growth, liquidez, tail beta cap, downside beta y RMT crowding penalty."
                ),
            ],
        )
    )

    sections.append(
        (
            "23. Optimización de estados y throttle causal",
            [
                p(
                    "El state optimizer decide exposición a growth/alpha según régimen, volatilidad, crowding, drawdown stress, entropía y breaches de validation. No selecciona activos usando test."
                ),
                formula(
                    "s_t = [trend, volRank, drawdownStress, RMTcrowding, dispersion,\n"
                    "       downsideCaptureValidation, CVaRBreachValidation, PELTregime,\n"
                    "       GARCHvol, entropy, CRLB]."
                ),
                formula(
                    "(beta_t,gamma_t) = f(s_t),\n"
                    "stress -> beta,gamma down; expansion -> beta up; recovery -> beta moderate."
                ),
                p(
                    "La lógica de control robusto se parece a un controlador con saturación: evita que una señal fuerte rompa budgets de downside."
                ),
            ],
        )
    )

    sections.append(
        (
            "24. Reinforcement learning y contextual bandit Kaizen",
            [
                p(
                    "RL directo para elegir tickers o pesos es estadísticamente peligroso con pocas trayectorias, baja SNR y no estacionariedad. La forma viable es contextual bandit para hiperparámetros y objetivos, sujeto a gates."
                ),
                formula(
                    "s_t = [z_t, IC_t, PBO_t, DD_t, CVaR_t, Turnover_t, profile_t],\n"
                    "a_t = [objective, lambda_Sigma, lambda_CVaR, w_max, sectorCap, targetVol]."
                ),
                formula(
                    "r_{t+1} = R_{p,t+1} - l1 CVaR_{t+1} - l2 DD_{t+1}\n"
                    "          - l3 Turnover_{t+1} - l4 Cost_{t+1} - l5 1_{SuitabilityBreach}."
                ),
                p(
                    "LinUCB o Thompson Sampling son suficientes como primera capa. Offline RL completo queda research-only hasta tener muchos episodios reales y policy evaluation robusta."
                ),
            ],
        )
    )

    sections.append(
        (
            "25. Validación nested walk-forward",
            [
                p(
                    "La validación correcta separa train, validation, test y final holdout. La selección de hiperparámetros ocurre en validation; el test queda intocable."
                ),
                formula(
                    "Train(t) -> Validation(t) -> Test(t,t+h) -> FinalHoldout.\n"
                    "D_train(t) subset F_t,\n"
                    "D_test(t,t+h) intersection D_train/validation(t) = empty."
                ),
                p(
                    "Purging elimina overlap de labels; embargo desplaza la fecha de ejecución para evitar contaminación por microestructura, reporting lag o ventanas solapadas."
                ),
                theorem_block(
                    "Proposición 25.1 (Test intocable)",
                    "Si theta se selecciona sólo con train/validation y se evalúa una vez en test, el estimador OOS no está contaminado por selección directa en test.",
                    "La política final theta*=A(D_train,D_val) es medible respecto a F_t. D_test no entra a A. La evaluación en D_test es posterior y no retroalimenta theta dentro de la misma partición. QED.",
                ),
            ],
        )
    )

    sections.append(
        (
            "26. White Reality Check, Hansen SPA y PBO",
            [
                p(
                    "Cuando se prueban muchos candidatos, el máximo performance observado está sesgado al alza. WRC y SPA testean si el mejor candidato supera un benchmark después de ajustar por data snooping."
                ),
                formula(
                    "d_{k,t} = R_{k,t} - R_{bench,t},\n"
                    "T_n = max_k sqrt(n) mean(d_k),\n"
                    "WRC_p = P^*(T_n^* >= T_n)."
                ),
                formula(
                    "PBO = P(lambda < 0),\n"
                    "lambda = logit(rank_OOS(selected_IS))."
                ),
                p(
                    "El framework usa bootstrap por bloques para respetar dependencia serial. La promoción productiva exige WRC<0.05, SPA<0.05 y PBO<0.10 en familia congelada."
                ),
            ],
        )
    )

    sections.append(
        (
            "27. Promotion gate y StrategyConstitution",
            [
                p(
                    "StrategyConstitution congela features permitidos, hiperparámetros, benchmark set, complexity budget y gates. Es una defensa contra flexibilidad excesiva del framework."
                ),
                formula(
                    "PromotionGate = { DXCDR>0, PBO<0.10, WRC_p<0.05, SPA_p<0.05,\n"
                    "                  ICIR>0, QLIKE_new<QLIKE_base, DD<DDmax, CVaR<CVaRmax }."
                ),
                p(
                    "Si una estrategia sólo funciona con un punto hiperparamétrico estrecho, se clasifica como research-only. Si falla PBO/WRC/SPA, no se promueve aunque tenga retorno alto."
                ),
                table(
                    [
                        ["Estado", "Condición", "UI"],
                        ["Promoted", "Pasa suitability y promotion gate", "Recommended allocation"],
                        ["Research-only", "Buen research pero falla gate estricto", "Evidence, no recommendation"],
                        ["Blocked", "Rompe suitability/risk hard limits", "Allocation blocked"],
                    ],
                    widths=[1.25 * inch, 2.55 * inch, 2.0 * inch],
                ),
            ],
        )
    )

    sections.append(
        (
            "28. Suitability CFA para usuarios no financieros",
            [
                p(
                    "La app está diseñada para usuarios no financieros. Por tanto, antes del optimizador debe existir un suitability engine que convierta horizonte, capital, liquidez y tolerancia a drawdown en restricciones duras."
                ),
                formula(
                    "Vol_p <= Vol_max(profile),\n"
                    "CVaR_p <= CVaR_max(profile),\n"
                    "DD_p <= DD_max(profile),\n"
                    "N <= N_capital, w_i <= w_max(profile)."
                ),
                p(
                    "El producto no debe mostrar 'recomendado' si el portafolio está fuera de perfil, aunque el backtest sea atractivo. Esta es una regla CFA de suitability y comunicación de riesgo."
                ),
            ],
        )
    )

    sections.append(
        (
            "29. Opciones, volatilidad implícita y límites del snapshot",
            [
                p(
                    "Yahoo ofrece snapshots de opciones: bid, ask, implied volatility, open interest, strikes y expiraciones. Esto permite diagnóstico contemporáneo, pero no backtest causal de opciones históricas."
                ),
                formula(
                    "IV_ATM = IV(K approximately S, DTE),\n"
                    "Skew = IV_put_OTM - IV_call_OTM,\n"
                    "PutCallOI = OI_put / OI_call."
                ),
                p(
                    "La superficie de volatilidad del portafolio se agrega por pesos, moneyness y DTE. Debe comunicarse como snapshot, no como serie histórica validada."
                ),
            ],
        )
    )

    sections.append(
        (
            "30. Termómetro geopolítico y noticias",
            [
                p(
                    "El termómetro geopolítico mide atención anormal, no probabilidad objetiva de conflicto. Su estadística correcta es within-topic robust z-score, no comparación cruda entre queries."
                ),
                formula(
                    "Z_{k,t}^{robust} = (V_{k,t} - median(V_{k,tau})) /\n"
                    "                 (1.4826 median_tau |V_{k,tau} - median(V_{k,tau})| + eps)."
                ),
                p(
                    "Los nulos aparecen cuando no hay suficiente historia/dispersión para estimar un z-score robusto. En ese caso, la tabla debe mostrar fallback cualitativo y no usar el dato como overlay de riesgo."
                ),
                p(
                    "El mapa mundial de noticias requiere geocodificación robusta. Regex por país ayuda, pero GDELT/RSS puede mencionar países no centrales al evento. La app debe etiquetar esto como attention map."
                ),
            ],
        )
    )

    sections.append(
        (
            "31. Carry trade y tasas globales",
            [
                p(
                    "La tabla de carry trade compara diferenciales de tasas, volatilidad FX proxy y riesgo de evento. Sin datos de forwards, basis y hedging costs, es research diagnostic, no recomendación ejecutable."
                ),
                formula(
                    "CarryScore_{a,b} = (r_a - r_b) / (sigma_FX + lambda_event EventRisk + lambda_drawdown DD_FX)."
                ),
                p(
                    "SOFR, SONIA, ESTR y TONAR sustituyen referencias tipo LIBOR. La curva debe respetar frecuencias discretas por país y no interpolar visualmente como si todo fuera continuo."
                ),
            ],
        )
    )

    sections.append(
        (
            "32. Backtest, NAV sintético y precios",
            [
                p(
                    "El portafolio no tiene precio observado como un ETF. Por eso se reconstruye un NAV sintético a partir de holdings OOS y precios diarios observados. Debe etiquetarse como portfolio price path o synthetic NAV."
                ),
                formula(
                    "NAV_t = NAV_{t-1}(1 + sum_i w_{i,tau(t)} r_{i,t} - TC_t),\n"
                    "BenchmarkPrice_t = observed adjusted price of xi."
                ),
                p(
                    "El drawdown se calcula siempre desde NAV diario, no desde puntos de rebalance aislados. Esto evita la gráfica plana incorrecta."
                ),
            ],
        )
    )

    sections.append(
        (
            "33. Arquitectura de software: core, UI, DB y cloud",
            [
                p(
                    "La regla de ingeniería es Core = source of truth, UI = renderer, DB = audit layer. El frontend no debe recomputar métricas financieras."
                ),
                formula(
                    "DataLayer -> FeatureLayer -> SignalLayer -> PortfolioLayer -> BacktestLayer -> RiskLayer -> UILayer."
                ),
                table(
                    [
                        ["Módulo", "Función"],
                        ["quant_stockpicker_core.py", "Motor principal: datos, señales, optimización, backtest, risk diagnostics"],
                        ["run_xcdr_v3_parallel_research.py", "Research paralelo XCDR/XODR, WRC/SPA/PBO, weights artifact"],
                        ["stockpicker_app.py", "Renderer Streamlit, dashboard, tabs, user controls"],
                        ["supabase_store.py", "Persistencia de runs y artifacts"],
                        ["cloud_daily_refresh.py", "Precompute diario antes de mercado"],
                        ["quant_core/promotion_gate.py", "Contratos de promoción productiva"],
                    ],
                    widths=[2.0 * inch, 4.0 * inch],
                ),
            ],
        )
    )

    sections.append(
        (
            "34. Dashboard y funciones de la app",
            [
                p(
                    "La app expone módulos: Dashboard/Overview, Allocation, Research Strategy, Price Path, Risk, Validation, Market Regime, Options, Fundamentals, Data Freshness y Advanced. Para producción móvil se recomienda agrupar en Dashboard, Portfolio, Risk, Market, Evidence, Assistant y Advanced."
                ),
                table(
                    [
                        ["Vista", "Contenido", "Usuario objetivo"],
                        ["Overview", "Estado, suitability, promotion, macro", "Todos"],
                        ["Allocation", "Pesos, sectores, razones de inclusión", "Todos"],
                        ["Research Strategy", "XCDR/XODR vs xi, WRC/SPA/PBO, pesos research", "Analyst"],
                        ["Risk", "Volatilidad, CVaR, drawdown, PELT, GARCH", "Analyst/Expert"],
                        ["Validation", "Promotion tests, CPCV/PBO, SPA", "Expert"],
                        ["Market", "Tasas, régimen, noticias, carry diagnostic", "Todos con microcopy"],
                        ["Advanced", "Raw artifacts y debugging", "Admin"],
                    ],
                    widths=[1.4 * inch, 3.0 * inch, 1.4 * inch],
                ),
                p(
                    "La precarga diaria escribe dashboard_payload en Supabase. Al abrir, la app intenta cargar ese artifact; sólo recalcula si el usuario presiona Run Allocation Engine."
                ),
            ],
        )
    )

    sections.append(
        (
            "35. Supabase, jobs, artifacts y despliegue costo-cero",
            [
                p(
                    "La arquitectura cloud recomendada es render-first: GitHub Actions actualiza a las 08:40 CT, Supabase guarda artifacts y la UI online sólo consulta."
                ),
                formula(
                    "GitHub Actions -> cloud_daily_refresh.py -> run_pipeline(config)\n"
                    "-> Supabase runs + run_artifacts.dashboard_payload -> Streamlit/Next.js renderer."
                ),
                table(
                    [
                        ["Tabla", "Uso"],
                        ["runs", "Auditoría de configuración, benchmark, estado y versión"],
                        ["run_artifacts", "dashboard_payload, backtest_path_bundle, gates, freshness"],
                        ["portfolio_weights", "Pesos por run"],
                        ["backtest_perf", "Performance walk-forward"],
                        ["risk_diagnostics", "Métricas de riesgo"],
                        ["jobs", "Cola de optimización futura"],
                    ],
                    widths=[1.8 * inch, 4.0 * inch],
                ),
                p(
                    "Para 10 usuarios, el dato de mercado debe ser global y compartido; portafolios, perfiles, runs y chats deben estar aislados por user_id y RLS."
                ),
            ],
        )
    )

    sections.append(
        (
            "36. Seguridad, privacidad y firewall de información privada",
            [
                p(
                    "La service_role de Supabase nunca debe estar en frontend. Debe vivir sólo en jobs confiables o server-side. El side/private portfolio no debe alimentar scoring público, RAG ni recomendaciones para terceros."
                ),
                formula(
                    "PublicResearchData intersection PrivateMNPI = empty,\n"
                    "RAGContext_user_a intersection Portfolio_user_b = empty."
                ),
                p(
                    "La app implementa auth local con bcrypt/JWT cookie y roles admin/analyst/viewer. En cloud multiusuario, Supabase Auth + RLS debe ser frontera oficial."
                ),
            ],
        )
    )

    sections.append(
        (
            "37. UX para usuarios no financieros",
            [
                p(
                    "El rigor matemático no debe obligar al usuario a interpretar tablas crudas. El primer pantallazo debe mostrar estado, retorno, downside, benchmark y acción recomendada. WRC/SPA/PBO pertenecen a Evidence o Expert Mode."
                ),
                table(
                    [
                        ["Término técnico", "Texto usuario"],
                        ["Promotion gate", "Strategy confidence"],
                        ["CVaR", "Tail loss risk"],
                        ["Drawdown", "Peak-to-trough loss"],
                        ["Synthetic NAV", "Portfolio price path"],
                        ["Research-only", "Needs more evidence"],
                    ],
                    widths=[2.4 * inch, 3.2 * inch],
                ),
                p(
                    "El chatbot debe explicar, guiar y tutorializar; no debe inventar recomendaciones ni saltarse suitability/promotion gates."
                ),
            ],
        )
    )

    sections.append(
        (
            "38. Red-team y failure modes",
            [
                p(
                    "La estrategia debe intentarse destruir antes de promocionarse: cambiar ventanas, universo, benchmark set, costos, seed, bootstrap, crisis windows, excluir sectores ganadores y eliminar top winners."
                ),
                bullet(
                    [
                        "Si el retorno desaparece al eliminar top winners, el alpha puede ser concentración retrospectiva.",
                        "Si WRC/SPA falla, el máximo observado puede ser data snooping.",
                        "Si PBO es alto, la selección de hiperparámetros no es estable.",
                        "Si CVaR sube aunque DD baje, la cola diaria sigue vulnerable.",
                        "Si xi cambia drásticamente por ventana, el mandato no está bien identificado.",
                    ]
                ),
                formula(
                    "Research-only if: WRC_p>=0.05 or SPA_p>=0.05 or PBO>=0.10\n"
                    "or UC<=1 or DC>=1 or CVaR_p>CVaR_xi or DD_p>DD_xi."
                ),
            ],
        )
    )

    sections.append(
        (
            "39. Resultados de research y lectura actual",
            [
                p(
                    "El research reciente identificó candidatos con active return positivo, pero también mostró una frontera real entre mayor retorno y control de cola. enhanced_growth_anchor_dd_budget_policy fue defendible; tail-aware no mejoró sin aumentar breach de cola; el estado final permanece research-grade, not promoted."
                ),
                formula(
                    "Observed smoke pattern:\n"
                    "Return_p > Return_xi, Vol_p <= Vol_xi, DD_p <= DD_xi,\n"
                    "but UC_p <= 1 or CVaR_p > CVaR_xi in some candidates."
                ),
                p(
                    "La conclusión correcta es no declarar producción hasta pasar full research: windows>=12, universe>=90, bootstrap>=300/500, WRC/SPA/PBO estrictos y final holdout congelado."
                ),
            ],
        )
    )

    sections.append(
        (
            "40. Teoremas finales de gobernanza cuantitativa",
            [
                theorem_block(
                    "Teorema 40.1 (No hay dominancia gratuita)",
                    "Bajo restricciones long-only equity sin leverage, shorts, derivados ni cash/bond sleeve dinámico, no puede garantizarse para todo periodo R_p>max Omega y D_p<min Omega.",
                    "Si Omega contiene benchmarks con perfiles incompatibles, por ejemplo QQQ growth y USMV low-vol, superar simultáneamente el upside del primero y downside del segundo exige convexidad que no está disponible en un simplex long-only estático salvo coincidencias de muestra. La garantía universal violaría la heterogeneidad de payoff del conjunto. QED.",
                ),
                theorem_block(
                    "Teorema 40.2 (Dominancia promocionable)",
                    "Una estrategia puede considerarse promocionable sólo si su superioridad OOS sobre xi permanece después de controlar multiple testing y downside.",
                    "El active return positivo prueba diferencia muestral, no skill. WRC/SPA corrigen data snooping; PBO evalúa inestabilidad de selección; DD/CVaR/downside protegen suitability. La conjunción de gates es condición operacional de promoción. QED.",
                ),
                theorem_block(
                    "Teorema 40.3 (Valor agregado de Kaizen)",
                    "El valor del framework reside en adaptar exposición a uncertainty state, no en maximizar una métrica aislada.",
                    "RMT, Kalman/state-space, GARCH/Volterra, Fisher/CRLB y entropía reducen dimensiones distintas de incertidumbre. Si todas se usan como restricciones/gobernadores y no como grados libres sin control, reducen probabilidad de asignar capital a señales espurias. QED.",
                ),
            ],
        )
    )

    appendices = [
        ("A. Pseudocódigo del pipeline", "Input config, load cached market data, build PIT fundamentals, infer regime, choose xi, rank sector-normalized candidates, estimate uncertainty state, optimize sleeves, run nested walk-forward, compute WRC/SPA/PBO, persist dashboard_payload."),
        ("B. Contrato dashboard_payload", "status: suitability, promotion, data_freshness; allocation: recommended_portfolio, weights; charts: price_paths, drawdowns, forecast_cone, conditional_vol, rate_curves; tables: fundamentals, risk, validation, rejections."),
        ("C. Tests mínimos", "future contamination must fail; RMT covariance PSD; Volterra kernel causal; Kalman state no future data; Optuna/PSO no test access; XCDR degrades with CRLB, turnover or entropy collapse."),
        ("D. Checklist de producción", "Rotate keys; apply RLS; keep service_role server-side; daily refresh before 09:00 CT; Streamlit render-first; user-safe explanations; no recommendation if gates fail."),
        ("E. Glosario", "xi: optimal mandate benchmark; Omega: benchmark stress set; UC: upside capture; DC: downside capture; PIT: point-in-time; WRC: White Reality Check; SPA: Superior Predictive Ability; PBO: Probability of Backtest Overfitting."),
        ("F. Limitaciones costo-cero", "No survivorship-free institutional universe, no historical options chain, no true PIT Yahoo fundamentals, no bid/ask historical microstructure, no tax-aware broker execution."),
        ("G. Roadmap research", "TailAwareGrowthSleeve, ConvexOpportunityUniverseBuilder, Student-t EGARCH, OOS QLIKE, benchmark-cluster validation, perturbation red-team, Next.js PWA render-only."),
        ("H. Roadmap UX", "Investor Mode, Expert Mode, dashboard-first, mobile bottom nav, evidence panel, assistant RAG, daily freshness status, clear blocked/research/promoted states."),
    ]
    for title, text in appendices:
        sections.append((title, [p(text), formula("Invariant: frontend renders artifacts; quant engine computes; database audits."), p("Este apéndice forma parte del contrato operativo del framework y debe mantenerse alineado con cambios de versión de modelo, app y schema.")]))

    return sections


def build_pdf():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title="Quant Portfolio-Kaizen Monografia Formal",
        author="Christopher Jacob Ahumada Robles",
    )
    story = [
        Paragraph("Quant Portfolio-Kaizen", S["Title"]),
        Paragraph(
            "Framework formal de stock picking, benchmark governance, XCDR/XODR, control robusto de downside, validación causal y dashboard cloud costo-cero",
            S["Subtitle"],
        ),
        Paragraph("Autor: Christopher Jacob Ahumada Robles", S["Subtitle"]),
        Paragraph("En atención a Roberto Carlos Guzmán Orduño", S["Subtitle"]),
        Paragraph("Fecha: Junio 2026", S["Subtitle"]),
        Spacer(1, 16),
        HR(),
        Spacer(1, 14),
        p(
            "Este documento expone la arquitectura matemática, financiera, estadística y computacional de Quant Portfolio-Kaizen. "
            "Incluye definiciones formales, pruebas, objetivos de optimización, construcción del benchmark xi, conjunto Omega, "
            "gobierno de promoción, módulos de la app, arquitectura cloud y riesgos de implementación."
        ),
        box("Nota de uso: documento de investigación y arquitectura. No constituye recomendación de inversión personalizada."),
        PageBreak(),
        h1("Tabla de contenidos resumida"),
        bullet([title for title, _ in content_sections()]),
    ]
    for title, body in content_sections():
        story.extend(section(title, body))
    story.append(PageBreak())
    story.append(h1("Cierre formal"))
    story.append(
        p(
            "Quant Portfolio-Kaizen debe entenderse como una plataforma de decisión bajo incertidumbre. Su rigor no depende de una métrica aislada, "
            "sino de la coherencia entre filtración, disponibilidad de datos, benchmark governance, reducción de incertidumbre, optimización robusta, "
            "validación adversarial y comunicación responsable al usuario final."
        )
    )
    story.append(
        formula(
            "Final contract:\n"
            "If Suitability != Approved -> Blocked.\n"
            "If PromotionGate != Passed -> Research-only.\n"
            "If DataFreshness == Stale -> Require refresh.\n"
            "Only then may the app display a recommended allocation."
        )
    )

    def on_page(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(REGULAR, 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(0.72 * inch, 0.38 * inch, "Quant Portfolio-Kaizen — Monografía formal")
        canvas.drawRightString(7.78 * inch, 0.38 * inch, f"Página {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def clean_tex_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def build_tex():
    lines = [
        r"\documentclass[11pt,letterpaper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[spanish,es-nodecimaldot]{babel}",
        r"\usepackage{amsmath,amssymb,amsthm,mathtools,booktabs,longtable,geometry,setspace,hyperref}",
        r"\geometry{margin=1in}",
        r"\onehalfspacing",
        r"\newtheorem{theorem}{Teorema}",
        r"\newtheorem{proposition}{Proposición}",
        r"\newtheorem{lemma}{Lema}",
        r"\title{\textbf{Quant Portfolio-Kaizen}\\Framework formal de stock picking, benchmark governance, XCDR/XODR y asset allocation robusto}",
        r"\author{Christopher Jacob Ahumada Robles\\En atención a Roberto Carlos Guzmán Orduño}",
        r"\date{Junio 2026}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        "Monografía formal del framework Quant Portfolio-Kaizen, incluyendo filtración causal, construcción del benchmark xi, conjunto Omega, XCDR/XODR, reducción de incertidumbre, validación walk-forward y arquitectura de aplicación.",
        r"\end{abstract}",
        r"\tableofcontents",
        r"\newpage",
    ]
    for title, body in content_sections():
        lines.append(r"\section{" + clean_tex_text(title) + "}")
        for item in body:
            # The PDF flowables already contain enough text; the TeX source is an editable companion.
            raw = getattr(item, "text", "")
            if raw:
                lines.append(clean_tex_text(raw))
                lines.append("")
    lines.append(r"\end{document}")
    TEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    build_pdf()
    build_tex()
    print(PDF_PATH)
    print(TEX_PATH)


if __name__ == "__main__":
    main()
