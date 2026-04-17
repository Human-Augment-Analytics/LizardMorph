def test_predictor_library_module_importable():
    import predictor_library  # noqa: F401


def test_list_predictors_empty(client):
    res = client.get("/predictors")
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["predictors"] == []


def test_upload_and_delete_predictor_roundtrip(client):
    import io

    res = client.post(
        "/predictors",
        data={"predictor": (io.BytesIO(b"not-a-real-dat"), "example.dat")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200, res.get_data(as_text=True)
    body = res.get_json()
    assert body["success"] is True
    pred = body["predictor"]
    assert pred["id"]
    assert pred["display_name"] == "example.dat"

    res2 = client.get("/predictors")
    assert res2.status_code == 200
    body2 = res2.get_json()
    assert len(body2["predictors"]) == 1
    assert body2["predictors"][0]["id"] == pred["id"]

    res3 = client.delete(f"/predictors/{pred['id']}")
    assert res3.status_code == 200
    assert res3.get_json()["success"] is True

    res4 = client.get("/predictors")
    assert res4.status_code == 200
    assert res4.get_json()["predictors"] == []


def test_upload_predictor_rejects_non_dat_extension(client):
    import io

    res = client.post(
        "/predictors",
        data={"predictor": (io.BytesIO(b"whatever"), "example.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code in (400, 422)
    body = res.get_json()
    assert body["success"] is False


def test_upload_predictor_rejects_oversize(client, monkeypatch):
    import io

    import app as app_mod

    monkeypatch.setattr(app_mod, "PREDICTOR_MAX_BYTES", 4, raising=False)
    res = client.post(
        "/predictors",
        data={"predictor": (io.BytesIO(b"12345"), "example.dat")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 422
    body = res.get_json()
    assert body["success"] is False


def test_free_autoplace_requires_predictor_id(client):
    res = client.post("/free_autoplace?filename=does_not_matter.jpg", json={})
    assert res.status_code in (400, 422)

