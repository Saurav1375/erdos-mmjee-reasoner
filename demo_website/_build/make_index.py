"""Wraps page.html (the page body, as published online) into a full index.html for opening offline."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
page = (HERE / "page.html").read_text(encoding="utf-8")
head, sep, body = page.partition("</style>")
html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + head + sep + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
(HERE.parent / "index.html").write_text(html, encoding="utf-8")
print("wrote index.html")
