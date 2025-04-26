import fitz   # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf_with_ocr(pdf_path):
    raw = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        # 1) grab the real text
        raw += page.get_text()

        # 2) OCR every embedded image
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                img_dict = doc.extract_image(xref)
            except Exception:
                continue
            image_bytes = img_dict["image"]           # raw image bytes
            pil = Image.open(io.BytesIO(image_bytes)) # load into PIL
            raw += "\n" + pytesseract.image_to_string(pil)

    doc.close()
    return raw


api_key = ""

import datetime
from openai import OpenAI

# assume `text` is the full OCR+parsed text of one paper
# e.g. text = extract_text_from_pdf_with_ocr("paper.pdf")


# build a single prompt that both instructs the model and supplies the text
# 3) Build a prompt that does both: JSON + lay sections + extra section
def openapi(text):
    client = OpenAI(api_key=api_key)


    prompt = f"""
      I want you to give me a technical summary. As in go through the paper fully. Go through the methodology or the proposed methodology or the method or similar titles to see what they have done. As one paragraph in detail give the full technical details without leaving anything. I don’t need results. Just all the things he did in one paragraph. Don’t give any mathematical formulas. Give it to me like you are suggesting me how to do the problem if I asked you how to solve llm hallucination can be solved based on exactly what the author did. Don’t add anything else in the prompt. Start out by explaining like you are suggesting how to go about solving LLM hallucination. Don’t give me links and citations.
    
      Give me the title of the paper. just give those words. nothing extra
    
      Does the paper do it for white box or black box llm or something else. Just give me that. Nothing extra. Just those words only
    
      Which LLM is used in this paper which have been taken for checking hallucination. Just give me the name of the LLMs. Nothing else. No sentence. Just this. Put commas and give it to me
    
      What is the dataset which is used. Give it to me. Just give me the dataset names. Nothing else. if there are multiple just put commas and give it to me
    
      give me answer of each section
      \"\"\"{text}\"\"\"
      """

    # 4) Send to model
    start = datetime.datetime.now()
    resp = client.chat.completions.create(
      model="o4-mini",
      messages=[
          {"role":"system","content":"You extract structured data and explain AI research in plain English."},
          {"role":"user","content":prompt}
      ],
  )
    print("Time:", datetime.datetime.now() - start)

  # 5) Show the result
  # print(resp.choices[0].message.content)
    summary = resp.choices[0].message.content
    return summary

import os, pandas as pd

PDF_DIR   = "../LLM_Hallucination/research_paper_download"
OUT_XLSX  = "./paper_summaries.xlsx"
TXT_DIR   = "./summary_txt"   # safety copies

os.makedirs(TXT_DIR, exist_ok=True)    # make sure the folder exists
summaries = []
count=0
for fname in os.listdir(PDF_DIR):
    if not fname.lower().endswith(".pdf"):
        continue
    count+=1
    # if count <=119: continue
    pdf_path  = os.path.join(PDF_DIR, fname)
    print(count,pdf_path)
    paper_txt = extract_text_from_pdf_with_ocr(pdf_path)
    summary   = openapi(paper_txt)

    # 1) save per-paper .txt safety file
    base      = os.path.splitext(fname)[0]            # drop .pdf
    txt_path  = os.path.join(TXT_DIR, f"{base}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary)

    # 2) collect for Excel
    summaries.append({"summary": summary})

# 3) one-column Excel with all summaries
pd.DataFrame(summaries).to_excel(OUT_XLSX, index=False)
print(f"Wrote {len(summaries)} summaries\n• Excel → {OUT_XLSX}\n• Text files → {TXT_DIR}")
