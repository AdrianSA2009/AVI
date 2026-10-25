import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app import routes


def test_predict_accepts_image_and_returns_model_output(monkeypatch):
    expected = [{"label": "A", "confidence": 0.9, "box": [1.0, 2.0, 3.0, 4.0]}]
    monkeypatch.setattr(routes, "predict_frame", lambda image, conf: expected)
    ok, encoded = cv2.imencode(".jpg", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok

    response = TestClient(app).post(
        "/api/v1/predict",
        files={"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {"predictions": expected}


def test_predict_rejects_non_image_content_type():
    response = TestClient(app).post(
        "/api/v1/predict",
        files={"image": ("frame.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
