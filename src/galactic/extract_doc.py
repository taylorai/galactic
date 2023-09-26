import platform
from tempfile import TemporaryDirectory
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import logging

logger = logging.getLogger("galactic")

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    path_to_poppler_exe = Path(r"C:\.....")
    out_directory = Path(r"~\Desktop").expanduser()
else:
    out_directory = Path("~").expanduser()


def extract_doc(path: str) -> str:
    image_file_list = []
    extracted_text = ""  # This variable will accumulate the extracted text

    with TemporaryDirectory() as tempdir:
        if platform.system() == "Windows":
            pdf_pages = convert_from_path(
                path, 500, poppler_path=path_to_poppler_exe
            )
        else:
            pdf_pages = convert_from_path(path, 500)

        for page_enumeration, page in enumerate(pdf_pages, start=1):
            filename = Path(tempdir) / f"page_{page_enumeration:03}.jpg"
            page.save(filename, "JPEG")
            image_file_list.append(filename)

        for image_file in image_file_list:
            with Image.open(image_file) as img:
                text = str(pytesseract.image_to_string(img))
                text = text.replace("-\n", "")
                extracted_text += text + "\n"  # Accumulate the text

    return extracted_text  # Return the accumulated text


"""
from src.galactic import GalacticDataset
from src.galactic import extract_doc
extract_doc("../Downloads/example.pdf")
"""
