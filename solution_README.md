# 3Bee code test solution

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

    

