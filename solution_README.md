# 3Bee code test solution

## Requirements

The solution involves the usage of the following packages:

```shell
# FastAPI
pip install "fastapi[standard]"
```

```shell
# pytest
pip install pytest
```

### Run instructions

Assuming you are in the project root, execute the following commands to run the fastapi server:

```shell
cd src
cd pollinator_abundance
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

The `--reload` option makes uvicorn to listen for changes in the code and to restart automatically.

You can now navigate to `http://localhost:8000` and you should see the following output: `{"message":"App is alive!"}`

### Run tests

To run tests execute the following command:

```shell
pytest src/pollinator_abundance/tests/test_end_user.py
```

## Code analysis

### Key basic concepts

Business domain:

* ROI (Region of Interest): the focal study area. Example: a plantation.
* CA (Context Area): larger surrounding landscape providing ecological context. Example: a portion of a region.
* Plantation: could be narrower than ROI, but from the code it looks like it is a 1:1 mapping with the ROI. Example: a vineyard inside the farm.
* MSA (Mean Species Abundance): measure average species richness or diversity (plants or animals).
* MSA in LU (Mean Species Abundance in Land Use): how the usage of land (agriculture, urban development, forestry, ...) affects MSA.
* PA (Pollinator Abundance): reflects the relative number of pollinators expected in an area based on habitat suitability.
* NS (Nesting Suitability): Measures how well the landscape provides nesting habitats for pollinators.Floral Availability (FA): Indicates the availability of floral resources (nectar and pollen) necessary for pollinator foraging.
* NP (Nectar Potential): Quantifies nectar production potential as a proxy for food resources. 
* FA (Floral Availability): Indicates the availability of floral resources (nectar and pollen) necessary for pollinator foraging.

Technical domain:

* (?) Polygon pixels: pixels that fall inside a specific BBox given a certain resolution. These pixels hold data that can later be used to perform KPIs calculations and obtain spatial metrics.

### Code sections in depth

In this chapter code is explained more in detail. Please refer to the comments in the code to understand which section is being analysed.

#### `pollinator_abundance_calculation()` - Section #1

This is the setup and initialization section:

* Prepare inputs later used to individuate ROI, CAI and KPIs to calculate.
* Prepare spatial data related to ROI and CAI:
  * Relate image pixels to real world distances.
  * Define metadata such as bounding boxes and alignment points (later used to merge ROI and CA images) 
* Initialize structures to output results

#### `pollinator_abundance_calculation()` - Section #2

This section is focused on map / image operations: 

* Load starting numpy arrays representing CLC maps from local `data` folder and convert them to images for further processing
* Retrieve pixel polygons and bounding box for the ROI
* Merge ROI and CA images alignment-wise, output is a single image with ROI correctly placed over the CA
* Coverts from CLC color-scaled images to K-value-scaled images, where colors are mapped to new metrics (e.g. K = Pollinator Abundance). ROI and CA K-value-scaled arrays are then merged alignment-wise.
* Prepare masks that isolate ROI and CAI, helpful to isolate calculations to relevant spatial areas. Points that have no information are filtered out

Everything is now ready to calculate KPIs.

#### `pollinator_abundance_calculation()` - Section #3

This is the KPI calculation section. Here metrics are computed and image reports are generated. Based on initial configuration, the following KPIs can be computed:

* CLC (Corine Land Cover)
* NP (Nectar Potential)
* FA (Flower Availability)
* MSA (Mean Species Abundance)
* MSA_LU_animals: animals MSA in Land Use
* MSA_LU_plants: plants MSA in Land Use

For each KPI to calculate, based on which KPI to calculate, input images, masks and reference, the `kpi_elements_generation` function: 

* Prepares (again) the input images 
* Calculates the KPI
* Generates the full image with computed metrics(ROI + CA)
* Applies masks to isolate ROI and CAI and generates the final report image, together with title, scale, legend and so on.

## Code changes

### FastAPI integration

FastAPI has been added to the project, the following endpoints have been registered:

* `/`: root, just to test the app is alive
* `/calculate`: this endpoint returns the PA KPI calculated for the CA, the ROI and their delta. Accepts all the params needed for the analysis.
* `/calculate_all_kpis`: this endpoint returns all the KPIs
* `/calculate_v1`: same as above but uses v1 functions (not parallelized)
* `/calculate_all_kpis_v1`: same as above but uses v1 functions (not parallelized)

Here's an example of parametrized url:

`http://localhost:8000/calculate?plantation_id=9827&plantations_polygons_id=9773&resolution=low&ca_id=284085&roi_id=284086&override_bee=true&how=local&compute_pa_ns=true&compute_only_msa=false`

### Performance improvements

The average code execution before the changes was 12s.

What I've implemented is KPI calculation parallelization. In order to achieve this in a cleaner way, I've first created a new `KPIConfig` model which will represent all the settings to pass to the `kpi_elements_generation` function.
What I can do with this new model is to prepare an array of configurations before actually executing them, in this way I can instantiate a `ThreadPoolExecutor` and run `kpi_elements_generation` function in parallel, passing a different config to each thread.

The average execution now is 10.5s, with an expected 12.5% of performance improvement.

Moreover, I've compared results of the two functions (V1 and V2) and they are the same.

### Future improvements

The following improvements could lead to a easier code maintenance and readability:

* Creating enum classes to represent all the string constants used in the app (e.g. kpi names)
* The function `image_to_clc_ns_v3` is called both outside and inside the `kpi_elements_generation`, could be refactored to be called just once.
* Wrap endpoints input params in a Pydantic model
* Add metadata to endpoint input params (or to Pydantic model)
* Add defaults for input params
* Wrap returned jsons in a Pydantic model
* Write unit tests for core functions
* Divide main functions into smaller pieces for reusability and maintenance