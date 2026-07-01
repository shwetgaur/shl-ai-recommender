from app.catalog import KEY_TO_LETTER, load_catalog
from app.config import get_settings


def test_catalog_loads_nonempty(catalog):
    assert len(catalog) > 100


def test_every_assessment_has_name_and_url(catalog):
    for a in catalog.assessments:
        assert a.name
        assert a.url.startswith("http")


def test_test_type_letters_valid(catalog):
    valid = set(KEY_TO_LETTER.values())
    for a in catalog.assessments:
        for letter in [x for x in a.test_type.split(",") if x]:
            assert letter in valid


def test_lookup_by_url_and_id(catalog):
    a = catalog.assessments[0]
    assert catalog.get(a.entity_id) is a
    assert catalog.by_url(a.url) is a
    assert catalog.by_url(a.url.rstrip("/") + "/") is a  # trailing slash tolerant


def test_resolve_by_name(catalog):
    a = catalog.assessments[5]
    assert catalog.resolve(a.name) is a


def test_url_uniqueness(catalog):
    urls = [a.url.lower().rstrip("/") for a in catalog.assessments]
    assert len(urls) == len(set(urls))
