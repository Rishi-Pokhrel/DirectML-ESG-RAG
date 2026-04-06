# AutoDiag: Screenshot Replication Guide

This document provides the exact queries and steps used to generate the screenshots in the `/screenshots` directory. This is intended to help developers and testers verify the system's performance.

## 1. Swagger UI (`1_swagger_ui.png`)
- **Access:** Start the server and navigate to `http://localhost:8000/docs`.
- **Purpose:** Verifies that all endpoints are correctly registered and accessible via FastAPI's automatic documentation.

## 2. Health Check (`2_health_endpoint.png`)
- **Action:** Open `http://localhost:8000/health` or view the header in the UI.
- **Verification:** Ensure `indexed_documents` matches the number of micro-chunks (typically 700+) and `memory_usage_mb` is within the target range (~500-600MB).

## 3. Multimodal Ingestion (`3_ingestion_success.png`)
- **Action:** Click the **"Re-Ingest"** button in the web interface.
- **Verification:** The UI should display a confirmation message indicating that 3 PDFs have been processed with text, table, and image extraction.

## 4. Text Query (`4_text_query.png`)
- **Query:** *"What are the main components of a planetary gear set?"*
- **Result:** The system should return a detailed list including the Sun Gear, Planet Gears, Ring Gear, and Planet Carrier, with citations from the `AutoTrans` and `Crawfords` manuals.

## 5. Table Query (`5_table_query.png`)
- **Query:** *"What are the shift solenoid specifications for the Aisin transmission?"*
- **Result:** The system should retrieve structured data from the shift solenoid tables, providing exact resistance values and operation modes.

## 6. Image Query (`6_image_query.png`)
- **Query:** *"Explain the hydraulic flow diagram on page 42."* (Note: The specific page depends on the document version).
- **Result:** Demonstrates Gemini 2.0 Flash's ability to analyze and describe complex automotive technical diagrams extracted from the manuals.

## 7. Bonus: Hardware Guardrail (`7_bonus_screenshot.png`)
- **Condition:** Simulate a low-RAM environment (Available RAM < 1GB).
- **Action:** Click the **"Query"** button before manual ingestion.
- **Result:** The system should display a **RED SYSTEM ALERT** guiding the user to use the "Re-Ingest" button first to prevent a memory overlap crash.

---
**Verification Tip:** For the most accurate replication, ensure your `GOOGLE_API_KEY` is active and you have downloaded the 4-bit quantized Qwen model via `setup_model.py`.
