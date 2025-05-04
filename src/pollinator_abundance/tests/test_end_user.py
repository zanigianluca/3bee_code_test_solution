from fastapi import status
from fastapi.testclient import TestClient

from pollinator_abundance.server import app

client = TestClient(app)


def test_calculate_ok():
    response = client.get(
        "/calculate?plantation_id=9827&plantations_polygons_id=9773&resolution=low&ca_id=284085&roi_id=284086&override_bee=true&how=local&compute_pa_ns=true&compute_only_msa=false")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"CA": 0.21923621293181564, "ROI": 0.13875595591184908, "Delta": 0.08048025701996656}


def test_calculate_all_kpis_ok():
    response = client.get(
        "/calculate_all_kpis?plantation_id=9827&plantations_polygons_id=9773&resolution=low&ca_id=284085&roi_id=284086&override_bee=true&how=local&compute_pa_ns=true&compute_only_msa=false")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "CA": {"PA": 0.21923621293181564, "FA": 0.5947155595376497, "NP": 380.3780211407706, "NS": 0.33681290582086915,
               "MSA": 0.6174597271337974, "MSA_LU_ANIMALS": 0.6630010409668441, "MSA_LU_PLANTS": 0.558013551025832},
        "ROI": {"PA": 0.13875595591184908, "FA": 0.4001030451157433, "NP": 208.5972984938646, "NS": 0.24954741744706083,
                "MSA": 0.40736631133820805, "MSA_LU_ANIMALS": 0.4442271165536076, "MSA_LU_PLANTS": 0.3642214606642587},
        "Delta": {"PA": 0.08048025701996656, "FA": 0.19461251442190636, "NP": 171.780722646906,
                  "NS": 0.08726548837380832, "MSA": 0.21009341579558932, "MSA_LU_ANIMALS": 0.2187739244132365,
                  "MSA_LU_PLANTS": 0.19379209036157335}}


def test_calculate_wrong_param_type_422():
    response = client.get(
        "/calculate?plantation_id=string_not_int&plantations_polygons_id=9773&resolution=low&ca_id=284085&roi_id=284086&override_bee=true&how=local&compute_pa_ns=true&compute_only_msa=false")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
