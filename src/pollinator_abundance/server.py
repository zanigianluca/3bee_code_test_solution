from fastapi import FastAPI
from .handler import pollinator_abundance_calculation

app = FastAPI()

@app.get("/calculate")
async def calculate():
    all_kpis = pollinator_abundance_calculation()
    result = {
        "CA": all_kpis["result_values"]["CA"]["PA"],
        "ROI": all_kpis["result_values"]["ROI"]["PA"],
        "Delta": all_kpis["result_values"]["Delta"]["PA"],
    }
    return result

@app.get("/calculate_all_kpis")
async def calculate():
    result = pollinator_abundance_calculation()
    return result["result_values"]
