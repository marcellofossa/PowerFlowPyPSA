1. Introduction

Hello, here is Marcello Fossa, master student in Energy Engineering at Politecnico di Milano. Check out my thesys project.

2. Inputs Distribution Tool

3. Inputs Power Flow Tool

- PF Excel workbook ([pf_input_esempio.xlsx](https://github.com/user-attachments/files/24957530/pf_input_esempio.xlsx)) with sheets:
    - Network: global electrical parameters (v_nom, power_factor, line_r, line_x, slack_pole_id, ph_at_slack, crs_epsg, ...)
      Format: columns [Parameter, Value].
    - Dispatch: hourly time series (Preliminary Sizning Tool-style columns supported).
- Nodes GeoJSON: poles (points). Must contain an id column (preferred: "id" or "pole_id"), Distribution Tool output supperted.
- Edges GeoJSON: LV network segments (lines) with endpoints (preferred: "source"/"target"), Distribution Tool output supperted.
- Associations CSV: building-to-pole mapping (building_id, pole_id), Distribution Tool output supperted.

4. Outputs
