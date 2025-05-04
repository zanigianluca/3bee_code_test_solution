from fastapi import FastAPI
from .handler import pollinator_abundance_calculation

app = FastAPI()


@app.get("/calculate")
async def calculate():
    result = pollinator_abundance_calculation()
    return result["result_values"]
