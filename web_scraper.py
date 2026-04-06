from bs4 import BeautifulSoup
import pandas as pd
from langchain_core.documents import Document
from io import StringIO
import requests


def web_scrape(url: str) -> list[Document]:

    html = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    ).text

    soup = BeautifulSoup(html, "html.parser")

    # Extract paragraphs
    all_paragraphs = "\n\n".join(
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if p.get_text(strip=True)
    )

    # Extract tables
    table_texts = []

    for table in soup.find_all("table"):
        try:
            df = pd.read_html(StringIO(str(table)))[0]
            table_texts.append(df.to_markdown(index=False))
        except Exception as e:
            print("Skipping table:", e)

    all_tables = "\n\n".join(table_texts)

    combined_text = all_paragraphs + "\n\n" + all_tables

    document = [Document(
        page_content=combined_text,
        metadata={"type": "full_page", "source": url}
    )]

    return document

document = web_scrape("https://247wallst.com/investing/2026/04/03/price-prediction-nvidia-stock-will-be-worth-this-much-in-2027/")
print(document)