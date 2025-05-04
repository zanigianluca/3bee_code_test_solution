from fastapi import FastAPI
from .handler import pollinator_abundance_calculation, pollinator_abundance_calculation_V2

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
    resultv1 = pollinator_abundance_calculation()
    resultv2 = pollinator_abundance_calculation_V2()
