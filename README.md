# PdfImageOCR
**Requirement:**<br/>
Pytorch<br/>
MMOCR<br/>
MMDetection<br/>
CUDA<br/>
tesseract-ocr<br/>
Python 3.6<br/>
**Objective:**<br/>
The objective is to recognize character from pdf images using deep learning based techniques.<br/>

**Here is a result of Text Extration:**

![Model](https://github.com/cvkworld/PdfImageOCR/blob/main/test-img/TRimg.png)

## Improved OCR Script

An example script `improved_ocr.py` has been added to provide a simple command line
utility for extracting text from images or PDFs. The script applies basic
denosing and adaptive thresholding before running Tesseract OCR.

```bash
python3 improved_ocr.py path/to/image_or_pdf -o output.txt
```

The extracted text will be printed to the console or saved to the specified
output file.
