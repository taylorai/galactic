# basically what we're gonna do here is give the user a tool to go from column of URLs to column of scraped pages,
# using tool of choice (requests, selenium, puppeteer, trafilatura, etc.)
from typing import Literal
import requests
import justext
import trafilatura
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from readabilipy import simple_json_from_html_string
from boilerpy3 import extractors

extractor = extractors.ArticleExtractor()


def scrape_urls(
    self,
    url_column: str,
    output_column: str,
    backend: Literal["requests", "trafilatura"],
):
    if backend == "selenium":
        # Initialize Selenium in headless mode
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        driver_service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(
            service=driver_service, options=chrome_options
        )

        def _scrape_single_url(url):
            driver.get(url)
            return driver.page_source

    elif backend == "requests":

        def _scrape_single_url(url):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return response.text
                else:
                    return "Failed to scrape"
            except Exception as e:
                return f"Failed to scrape."

    elif backend == "trafilatura":

        def _scrape_single_url(url):
            downloaded = trafilatura.fetch_url(url)
            result = trafilatura.extract(downloaded)
            if result:
                return result
            else:
                return "Failed to scrape"

    else:
        raise ValueError("Invalid backend")

    self.dataset = self.dataset.map(
        lambda sample: {output_column: _scrape_single_url(sample[url_column])}
    )


def postprocess_scraped_pages(
    self,
    input_column: str,
    output_column: str,
    backend: Literal[
        "readabilipy", "justext", "boilerpy3", "goose3"
    ] = "readabilipy",
):
    if backend == "readabilipy":

        def _postprocess_single_page(page):
            json_page = simple_json_from_html_string(page)
            result = ""
            if json_page["title"]:
                result += f"Title: {json_page['title']}\n\n"
            if json_page["byline"]:
                result += f"By: {json_page['byline']}\n\n"
            if json_page["plain_content"]:
                result += json_page["plain_content"]

            return result

    elif backend == "justext":

        def _postprocess_single_page(page):
            ## TODO: detect the language automatically
            paragraphs = justext.justext(page, justext.get_stoplist("English"))
            result = ""
            for paragraph in paragraphs:
                if not paragraph.is_boilerplate:
                    result += paragraph.text + "\n\n"

            return result

    elif backend == "boilerpy3":

        def _postprocess_single_page(page):
            if len(page) == 0:
                return "Failed to scrape"
            try:
                doc = extractor.get_doc(page)
                content = doc.content
                title = doc.title
                return f"Title: {title}\n\n{content}"
            except Exception as e:
                return "Failed to scrape"

    elif backend == "goose3":
        from goose3 import Goose

        g = Goose()

        def _postprocess_single_page(page):
            try:
                article = g.extract(raw_html=page)
                return article.cleaned_text
            except Exception as e:
                return "Failed to scrape"

    else:
        raise ValueError("Invalid postprocess")

    self.dataset = self.dataset.map(
        lambda sample: {
            output_column: _postprocess_single_page(sample[input_column])
        }
    )

    return self
