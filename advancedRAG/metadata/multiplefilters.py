def build_filters(
    department=None,
    year=None,
    document_type=None
):
    filters = {}

    if department:
        filters["department"] = department

    if year:
        filters["year"] = year

    if document_type:
        filters["document_type"] = document_type

    return filters


filters = build_filters(
    department="Finance",
    year=2026,
    document_type="policy"
)