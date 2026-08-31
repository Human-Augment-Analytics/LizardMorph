import pytest
from werkzeug.exceptions import NotFound

import app


def _match(url):
    return app.app.url_map.bind("127.0.0.1:3005").match(url)


def test_the_advertised_annotated_image_url_is_served_by_this_app():
    url = app.session_image_url("0123456789abcdef", "/data/sessions/s/annotated/output_1.png")

    endpoint, view_args = _match(url)

    assert endpoint == "serve_session_image"
    assert view_args == {"session_id_short": "01234567", "filename": "output_1.png"}


def test_the_dev_proxy_prefix_matches_no_route_on_the_backend():
    with pytest.raises(NotFound):
        _match("/api/images/01234567/output_1.png")
