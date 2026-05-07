# extractors.py
import os, tempfile, re, io, logging
from typing import Optional, Tuple, Generator, Dict, Any
from rapidocr_onnxruntime import RapidOCR
from docx import Document
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from config import GOOGLE_SCOPES

logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self):
        self._ocr: Optional[RapidOCR] = None

    @property
    def ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def extract_text(self, path: str, ftype: str) -> str:
        ftype = ftype.lower().strip(".")
        try:
            if ftype == "pdf":
                return self._extract_pdf(path)
            elif ftype == "docx":
                return self._extract_docx(path)
            elif ftype in {"png", "jpg", "jpeg", "webp"}:
                return self._ocr_image(path)
        except Exception as e:
            logger.error(f"Extraction failed [{ftype}]: {e}")
        return ""

    def _extract_pdf(self, path: str) -> str:
        import fitz
        parts = []
        with fitz.open(path) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                if len(text) > 50:
                    parts.append(text)
                else:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        result, _ = self.ocr(pix.tobytes("png"))
                        if result:
                            parts.append("\n".join(l[1] for l in result))
                    except Exception as e:
                        logger.warning(f"OCR page failed: {e}")
        return "\n".join(parts).strip()

    def _extract_docx(self, path: str) -> str:
        doc = Document(path)
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    def _ocr_image(self, path: str) -> str:
        result, _ = self.ocr(path)
        return "\n".join(l[1] for l in result) if result else ""


class GoogleDriveHandler:
    """Handles Google Drive integration for sheet processing."""
    
    def __init__(self, extractor: DocumentExtractor):
        self.extractor = extractor
        self.gc = None
        self.drive = None
        self._init_credentials()

    def _init_credentials(self):
        """Initialize Google credentials from various sources."""
        creds = None
        
        # Try loading from token.json first
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", GOOGLE_SCOPES)

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        # Initialize OAuth flow if no valid credentials
        if not creds:
            if not os.path.exists("credentials.json"):
                logger.warning("Google credentials unavailable. Drive integration disabled.")
                return
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
            with open("token.json", "w") as f:
                f.write(creds.to_json())

        if creds:
            self.gc = gspread.authorize(creds)
            self.drive = build("drive", "v3", credentials=creds)

    @property
    def is_available(self) -> bool:
        return self.gc is not None and self.drive is not None

    def process_sheet(self, url: str) -> Generator[Tuple[Dict[str, Any], int, int], None, None]:
        if not self.is_available:
            yield {"error": "Google Drive integration not configured."}, 0, 0
            return

        try:
            sheet = self.gc.open_by_url(url).get_worksheet(0)
            records = sheet.get_all_records()
            link_col = next(
                (k for r in records[:5] for k, v in r.items()
                 if isinstance(v, str) and "drive.google.com" in v),
                None
            )
            if not link_col:
                yield {"error": "No Google Drive links found in sheet"}, 0, 0
                return

            total = len(records)
            for idx, row in enumerate(records, 1):
                link = row.get(link_col, "").strip()
                if not link:
                    continue
                match = re.search(r"/file/d/([a-zA-Z0-9_-]+)|id=([a-zA-Z0-9_-]+)", link)
                if not match:
                    continue
                fid = match.group(1) or match.group(2)
                try:
                    text = self._download_and_extract(fid)
                    if text:
                        yield {"text": text, "metadata": row}, idx, total
                except Exception as e:
                    logger.error(f"Drive row {idx} failed: {e}")

        except Exception as e:
            yield {"error": str(e)}, 0, 0

    def _download_and_extract(self, fid: str) -> str:
        meta = self.drive.files().get(fileId=fid).execute()
        fname = f"tmp_{fid}_{meta.get('name', 'file')}"
        request = self.drive.files().get_media(fileId=fid)
        fh = io.BytesIO()
        dl = MediaIoBaseDownload(fh, request)
        while not dl.next_chunk()[1]:
            pass
        with open(fname, "wb") as f:
            f.write(fh.getvalue())
        ext = os.path.splitext(fname)[1].lower().strip(".") or "pdf"
        text = self.extractor.extract_text(fname, ext)
        try:
            os.remove(fname)
        except PermissionError:
            pass
        return text