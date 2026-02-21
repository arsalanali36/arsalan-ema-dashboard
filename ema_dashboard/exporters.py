from __future__ import annotations

"""Export helpers for printable/shareable chart artifacts."""

import json

from .charts import build_lwc_payload


def build_tv_style_html_document(day_results: list[dict], interval_value: int, chart_height: int) -> str:
    """Build standalone multi-page HTML with TradingView-like chart style."""
    pages = []
    for day_entry in day_results:
        candles, ema10, ema20, dema, _, markers = build_lwc_payload(day_entry["df"])
        pages.append(
            {
                "day": day_entry["day"],
                "interval": interval_value,
                "candles": candles,
                "ema10": ema10,
                "ema20": ema20,
                "dema": dema,
                "markers": markers,
            }
        )

    pages_json = json.dumps(pages)
    # Standalone HTML so user can open directly and print/save as PDF.
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
  <title>EMA Multi-Day TV Style</title>
  <script src=\"https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js\"></script>
  <style>
    @page {{ size: A4 landscape; margin: 0; }}
    html, body {{ margin: 0; padding: 0; background: #f5f5f5; font-family: Segoe UI, Arial, sans-serif; }}
    .wrap {{ margin: 0; padding: 0; }}
    .page {{ width: 297mm; height: 210mm; box-sizing: border-box; padding: 12mm; background: #f5f5f5; display: grid; grid-template-rows: 10mm 1fr; gap: 1mm; page-break-after: always; break-after: page; overflow: hidden; }}
    .page:last-child {{ page-break-after: auto; break-after: auto; }}
    .title {{ font-size: 7mm; font-weight: 700; line-height: 1; margin: 0; text-align: center; align-self: center; }}
    .chart {{ width: 100%; height: 100%; min-height: 0; }}
    @media print {{
      html, body {{ width: 297mm; height: auto; }}
      body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
      .wrap {{ margin: 0; padding: 0; }}
      .page {{ width: 297mm; height: 210mm; }}
      .page:last-child {{ break-after: auto; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\" id=\"root\"></div>
  <script>
    const pages = {pages_json};
    const root = document.getElementById("root");
    pages.forEach((p, idx) => {{
      const page = document.createElement("div");
      page.className = "page";
      const title = document.createElement("div");
      title.className = "title";
      title.textContent = `${{p.day}} | EMA Crossover | TF: ${{p.interval}} min`;
      const chartDiv = document.createElement("div");
      chartDiv.className = "chart";
      page.appendChild(title);
      page.appendChild(chartDiv);
      root.appendChild(page);

      const chart = LightweightCharts.createChart(chartDiv, {{
        layout: {{ background: {{ type: "solid", color: "#f5f5f5" }}, textColor: "#222" }},
        grid: {{ vertLines: {{ color: "#e0e0e0" }}, horzLines: {{ color: "#e0e0e0" }} }},
        rightPriceScale: {{ borderVisible: false, minimumWidth: 62 }},
        leftPriceScale: {{ visible: false }},
        timeScale: {{
          borderVisible: false,
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 0,
          fixLeftEdge: true,
          fixRightEdge: true
        }},
        crosshair: {{ mode: 0 }},
        width: chartDiv.clientWidth,
        height: chartDiv.clientHeight
      }});

      const candles = chart.addCandlestickSeries({{
        upColor: "#26a69a", downColor: "#ef5350", borderVisible: false, wickUpColor: "#26a69a", wickDownColor: "#ef5350"
      }});
      candles.setData(p.candles);
      candles.setMarkers(p.markers);
      const dema = chart.addLineSeries({{ color: "#000000", lineWidth: 3, lineType: 1, priceLineVisible: false }});
      dema.setData(p.dema);
      const ema10 = chart.addLineSeries({{ color: "#2962FF", lineWidth: 1, priceLineVisible: false }});
      ema10.setData(p.ema10);
      const ema20 = chart.addLineSeries({{ color: "#FF1744", lineWidth: 2, priceLineVisible: false }});
      ema20.setData(p.ema20);
      chart.timeScale().fitContent();
      const resizeChart = () => {{
        chart.applyOptions({{ width: chartDiv.clientWidth, height: chartDiv.clientHeight }});
        chart.timeScale().fitContent();
      }};
      window.addEventListener("resize", resizeChart);
    }});
  </script>
</body>
</html>"""
