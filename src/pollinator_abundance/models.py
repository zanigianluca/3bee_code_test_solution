class KPIConfig:
    def __init__(
        self,
        roi_id,
        ca_id,
        kpi,
        result_values,
        image_all,
        mask_roi,
        mask_ca,
        ref_array,
        palette,
        units,
        speed_factor,
        max_val,
        webp_img,
        webp_report,
        clc_values_roi,
        clc_values_ca,
        input_image_roi,
        input_image_ca,
        alignment_point_x,
        alignment_point_y,
        palette_min,
        palette_max,
        report_palette,
        filename,
        report_ext,
        title_report,
        title_bar,
        width_km_ca,
        height_km_ca,
        width_km_roi,
        height_km_roi,
        bounding_box_roi,
        site_pixel_polygons,
        filename_report,
        value_roi=None,
        value_ca=None,
        min_array_val=0,
        cbar_digits=1,
    ):
        self.roi_id = roi_id
        self.ca_id = ca_id
        self.kpi = kpi
        self.result_values = result_values
        self.image_all = image_all
        self.mask_roi = mask_roi
        self.mask_ca = mask_ca
        self.ref_array = ref_array
        self.palette = palette
        self.units = units
        self.speed_factor = speed_factor
        self.max_val = max_val
        self.webp_img = webp_img
        self.webp_report = webp_report
        self.clc_values_roi = clc_values_roi
        self.clc_values_ca = clc_values_ca
        self.input_image_roi = input_image_roi
        self.input_image_ca = input_image_ca
        self.alignment_point_x = alignment_point_x
        self.alignment_point_y = alignment_point_y
        self.palette_min = palette_min
        self.palette_max = palette_max
        self.report_palette = report_palette
        self.filename = filename
        self.report_ext = report_ext
        self.title_report = title_report
        self.title_bar = title_bar
        self.width_km_ca = width_km_ca
        self.height_km_ca = height_km_ca
        self.width_km_roi = width_km_roi
        self.height_km_roi = height_km_roi
        self.bounding_box_roi = bounding_box_roi
        self.site_pixel_polygons = site_pixel_polygons
        self.filename_report = filename_report
        self.value_roi = value_roi
        self.value_ca = value_ca
        self.min_array_val = min_array_val
        self.cbar_digits = cbar_digits