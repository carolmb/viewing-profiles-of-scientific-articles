from PyPDF2 import PdfFileMerger
import glob

pdfs = glob.glob('hist*.pdf')[:100]
pdfs = sorted(pdfs)
merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(open(pdf, 'rb'))

with open('hist.pdf', 'wb') as fout:
    merger.write(fout)