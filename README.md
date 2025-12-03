# T.H.E.R.M.
### (Thermal Health & Efficiency Reporting Module)

**T.H.E.R.M.** is a modular, forensic analytics engine designed to audit the performance of Samsung Heat Pumps (specifically the EHS Mono Gen 6). It transforms raw Grafana/Home Assistant CSV logs into actionable intelligence, highlighting efficiency gaps, hydraulic issues, and data quality problems.

---

## 🚀 Features

### 🔍 Forensic Run Inspector
* **Deep-Dive Analysis:** Visualizes individual heating and DHW runs with high-resolution charts (1-minute intervals).
* **Ghost Pumping Detection:** Automatically flags runs where heating zone pumps are active during Hot Water cycles, destroying efficiency.
* **Immersion Heater Auditing:** Detects and quantifies expensive immersion heater usage.
* **AI-Ready Context:** Generates rich JSON payloads for every run, ready to be pasted into LLMs for expert analysis.

### 📈 Long-Term Trends
* **Weather Compensation Analysis:** Visualizes Flow Temperature vs. Outdoor Temperature to verify curve optimization.
* **Efficiency Tracking:** Calculates daily SCOP (Seasonal Coefficient of Performance) and cost-per-kWh.
* **Environmental Correlation:** Overlays wind speed, humidity, and solar radiation against system efficiency.

### 🛡️ Data Quality Studio
* **Sensor Health Matrix:** A tiered scoring system (Gold/Silver/Bronze) for data completeness.
* **Adaptive Heartbeats:** Automatically learns "normal" reporting intervals for every sensor to detect gaps without false alarms.
* **Missing Data Visualization:** Heatmap style grid to instantly spot sensor outages across weeks or months.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/THERM.git](https://github.com/YourUsername/THERM.git)
    cd THERM
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 💻 Usage

1.  **Launch the Dashboard:**
    ```bash
    streamlit run app.py
    ```

2.  **Upload Data:**
    * Export your data from Grafana/Home Assistant History as CSVs (For Grafana, ensure you export both "Numeric" and "State" data if they are separate).
    * Drag and drop the CSV files into the sidebar uploader.

3.  **Analyze:**
    * Use the **Run Inspector** to find "bad runs" (short cycles, poor COP).
* Use the **Data Quality Audit** to verify your sensors are reporting correctly.

---

## 📂 Project Structure

* **`app.py`**: The main UI controller (Streamlit).
* **`processing.py`**: The "Physics Engine" – run detection, energy calculations, and hydraulic logic.
* **`baselines.py`**: Logic for adaptive sensor heartbeats and gap analysis.
* **`data_loader.py`**: Parsers for raw Grafana/CSV data handling.
* **`config.py`**: Central configuration for thresholds, tariffs, and sensor mappings.
* **`utils.py`**: Shared helper functions.

---

## 📄 License

**CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0 International)**

You are free to:
* **Share** — copy and redistribute the material in any medium or format.
* **Adapt** — remix, transform, and build upon the material.

**Under the following terms:**
* **Attribution** — You must give appropriate credit.
* **NonCommercial** — You may not use the material for commercial purposes (monetary gain).

Full license text: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)

---

*Disclaimer: This software is for educational and analytical purposes only. It is not a substitute for professional heating advice. The authors are not responsible for decisions made based on this data.*