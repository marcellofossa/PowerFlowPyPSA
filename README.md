# Patch Task 1 (cost model) + Task 2 (criterio standalone differenziale)

I file sono organizzati con i path del progetto: copia ciascuno in
`LV-Distribution-Topology-Streamlit-main/` sovrascrivendo l'esistente.

## File NUOVI

| File | Contenuto |
|---|---|
| `core/costs.py` | Cost model: `DistributionUnitCosts` (default dalla tabella costi), `StandaloneEconomics`, `StandaloneGate`, breakdown LV / last-mile / MV / trafo, legge di scala trafo C(S) = 3390·(S/25)^0.329 (ancorata sui due step-down Uganda V2B; il 250 kVA è uno step-up, escluso), spaziatura pali MV 60 m @11 kV / 120 m @33 kV |
| `pages/ui_sections/cost_sections.py` | Expander "Distribution cost analysis" (chiuso di default) con slider per tutti i costi unitari + breakdown + CSV. Due entry point: `render_validation_cost_section` (pag. 2) e `render_reinforcement_cost_section` (pag. 3, con costi per sottorete e tabella differenziale tra iterazioni k) |
| `tests/test_costs.py` | 10 test sul cost model |
| `tests/test_standalone_criterion.py` | 8 test sul nuovo criterio (incl. non-regressione del criterio legacy) |

## File MODIFICATI

| File | Modifica |
|---|---|
| `core/distribution_algos.py` | `place_poles_for_unassociated_buildings`: nuovi kwargs `standalone_gate`, `gdf_existing_poles`. Con gate economico il cluster candidato è connesso solo se `n·fisso + drop + palo + estensione ≤ n·(c_sa−c_gen)·E`; se rifiutato va negli standalone e la ricerca continua col cluster successivo. Con `standalone_gate=None` comportamento identico a prima |
| `core/distribution_service.py` | `run_low_voltage(..., standalone_economics=None, unit_costs=None)`: costruisce il gate una volta per run (l'estensione include i pali di supporto ogni `max_pole_span_m`) e passa i pali stradali come rete esistente. `None` = comportamento invariato (Page 3 e test esistenti non toccati) |
| `pages/ui_sections/topology_sections.py` | Blocco "Coverage behaviour": radio Economic (default) / Topological (legacy). Economico: 4 slider (c_sa, c_gen, kWh/anno, orizzonte) + caption col budget $ per edificio. `params` ritorna anche `standalone_criterion` e `standalone_economics` |
| `pages/1_Grid_Topology.py` | Costruisce `StandaloneEconomics` dai params e lo passa a `run_low_voltage` |
| `pages/2_Grid_Validation.py` | Nuova sezione "Cost analysis" (expander chiuso) DOPO il power flow, come in Grid Reinforcement; per topologie esterne (OGP/OMG/manuale) il drop medio è uno slider (default 32 m, media DRC) |
| `pages/3_Grid_Reinforcement.py` | Sezione "Cost analysis" dopo i risultati: breakdown iterazione finale (LV aggregata + MV + trafo), tabella per sottorete, comparativa differenziale tra iterazioni k |
| `pages/ui_sections/reinforcement_sections.py` | Summary metrics: "Total poles" sostituito da **LV poles** e **MV poles** (pali MV = backbone / spaziatura 60 m @11 kV o 120 m @33 kV, +1; tensione letta dalla request). Rimossa la metrica "V min [p.u.]" da tutte le sottoreti. ATTENZIONE: patch basata sulla copia del progetto Claude — se il tuo file locale è più recente, applica a mano le due modifiche (funzioni `_render_overall_summary` e `_render_subnetwork_expanders`) |
| `app.py` | Riscritta la home: 3 sezioni impilate verticalmente con formato identico (tagline, "What it does", Key inputs / Key outputs, link alla pagina). Aggiunta la sezione Grid Reinforcement; le descrizioni di Topology e Validation aggiornate alla versione corrente (criterio standalone economico, import OGP/OMG, cost analysis). Se la tua app.py locale ha contenuti che vuoi conservare, confrontala prima di sovrascrivere |

## Revisione 2 (richieste post-consegna)

- Tutti i costi aggregati in **k$ con 1 decimale** (metriche, tabelle breakdown,
  tabella per sottorete, comparativa iterazioni). I costi unitari degli slider
  restano in $ / $/m / $/km. Il CSV scaricabile mantiene i valori esatti in $.
- Cost analysis di Grid Validation spostata **dopo il power flow**.
- Grid Reinforcement: **LV poles / MV poles** al posto di Total poles;
  **V min [p.u.] eliminata** dalle sottoreti.
- **app.py** nuova con le 3 sezioni uniformi (vedi tabella sopra).

## Revisione 3 (confronto con BoQ Kwenge, ~1700 case)

- **Palo MV**: default 200 -> **250 $** (Kwenge 280, stima precedente 200).
- **Cavo MV**: default 1373 -> **2500 $/km**. Il rapporto ~1/5 rispetto al LV 3F
  del BoQ Uganda vale per il conduttore leggerissimo 7x4.26 mm2; i parametri
  elettrici del codice (`MvLineParams`: r=0.54 ohm/km, 185 A) implicano un
  ACSR ~50 mm2, per cui il default e' stato riallineato a meta' del range
  osservato (1373-3870 $/km, Kwenge). Resta comunque sotto il cavo LV 3F,
  come atteso per conduttori nudi MV vs fascio ABC isolato.

## Revisione 4 (costi per tipo di cavo + sidebar)

- **Costo per tipo di cavo LV**: `line_types.csv` accetta la colonna opzionale
  `cost_usd_per_m` (validata in `core/line_params.py` e propagata alla tabella
  risolta per-edge). In Grid Validation, se il catalogo risolve i costi, il
  breakdown sostituisce la riga generica "Backbone cable" con una riga per
  tipo (somma esatta lunghezza x costo); altrimenti resta lo slider. Costi
  parziali (alcuni tipi senza prezzo) -> fallback allo slider per ambiguita'.
- **`examples/line_types_with_costs.csv`**: catalogo ABC Al XLPE 16-150 mm2
  (X=0.083 costante, R~28.3/A, s_nom=sqrt(3)x0.4xImax) con prezzi derivati dal
  fit sui dati DRC (~0.017 $/mm2 totale al metro): 4x16 1.4 / 4x25 1.9 /
  4x35 2.6 / 4x50 3.5 / 4x70 4.7 / 4x95 6.2 / 4x120 7.7 / 4x150 9.5 $/m.
  I prezzi Uganda-isole osservati sono ~2x (premio logistico): usa la banda.
- **Sidebar "How to use"** aggiunte a Grid Validation e Grid Reinforcement
  (`render_sidebar` in validation_sections.py e reinforcement_sections.py,
  richiamate nelle pagine), stesso formato di Grid Topology.
- GR usa un solo cavo LV per il PF delle sottoreti, quindi il costo per tipo
  si applica solo in GV; in GR resta lo slider (estendibile se in futuro le
  sottoreti avranno cataloghi per-edge).

## Note

- `classify_standalone` (percorso manuale Page 3) resta sul criterio legacy: la
  classificazione economica passa da Page 1 quando usi "Use results from Grid
  Topology". Estendibile in un secondo momento se serve.
- Default criterio economico: c_sa 0.90 $/kWh, c_gen 0.38 $/kWh, 180 kWh/anno,
  20 anni → budget di connessione ≈ 1.872 $/edificio (break-even ≈ 105 m di
  estensione per edificio singolo con i costi di default).
- Verifiche eseguite in sandbox: 18/18 pytest + smoke end-to-end
  (41 edifici sintetici: l'isolato a 900 m diventa standalone col criterio
  economico; full coverage invariato).
