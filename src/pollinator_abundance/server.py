from fastapi import FastAPI
from .handler import pollinator_abundance_calculation, pollinator_abundance_calculation_V2

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "App is alive!"}

@app.get("/calculate")
async def calculate(plantation_id: int, plantations_polygons_id: int, resolution: str, ca_id: int, roi_id: int, override_bee: bool, how: str, compute_pa_ns: bool, compute_only_msa: bool):
    all_kpis = pollinator_abundance_calculation_V2(plantation_id, plantations_polygons_id, resolution, ca_id, roi_id, override_bee, how, compute_pa_ns, compute_only_msa)
    result = {
        "CA": all_kpis["result_values"]["CA"]["PA"],
        "ROI": all_kpis["result_values"]["ROI"]["PA"],
        "Delta": all_kpis["result_values"]["Delta"]["PA"],
    }
    return result

@app.get("/calculate_all_kpis")
async def calculate(plantation_id: int, plantations_polygons_id: int, resolution: str, ca_id: int, roi_id: int, override_bee: bool, how: str, compute_pa_ns: bool, compute_only_msa: bool):
    result = pollinator_abundance_calculation_V2(plantation_id, plantations_polygons_id, resolution, ca_id, roi_id, override_bee, how, compute_pa_ns, compute_only_msa)
    return result["result_values"]


@app.get("/calculate_v1")
async def calculate(plantation_id: int, plantations_polygons_id: int, resolution: str, ca_id: int, roi_id: int, override_bee: bool, how: str, compute_pa_ns: bool, compute_only_msa: bool):
    all_kpis = pollinator_abundance_calculation(plantation_id, plantations_polygons_id, resolution, ca_id, roi_id, override_bee, how, compute_pa_ns, compute_only_msa)
    result = {
        "CA": all_kpis["result_values"]["CA"]["PA"],
        "ROI": all_kpis["result_values"]["ROI"]["PA"],
        "Delta": all_kpis["result_values"]["Delta"]["PA"],
    }
    return result

@app.get("/calculate_all_kpis_v1")
async def calculate(plantation_id: int, plantations_polygons_id: int, resolution: str, ca_id: int, roi_id: int, override_bee: bool, how: str, compute_pa_ns: bool, compute_only_msa: bool):
    result = pollinator_abundance_calculation(plantation_id, plantations_polygons_id, resolution, ca_id, roi_id, override_bee, how, compute_pa_ns, compute_only_msa)
    return result["result_values"]