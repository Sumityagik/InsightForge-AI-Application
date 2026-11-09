"""
Sprint 5 tests: document loading, upload validation, secure storage,
analysis flow and ownership isolation.

Run with: python -m pytest tests/test_documents.py -v

The AI calls are mocked, so these tests need no internet or API key.
"""

import io
import os

import pytest

from app import create_app
from src.db import db
from src.document_loader import load_document, DocumentError, MAX_EXTRACTED_CHARS
from src.llm import AIServiceError


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def app(tmp_path):
    test_app = create_app()
    test_app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path / "uploads"))
    with test_app.app_context():
        db.create_all()
        yield test_app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(autouse=True)
def mock_ai(monkeypatch):
    """Replace the AI with fixed answers and record what it was given."""
    calls = {"analysis": [], "chat": []}

    def fake_analysis(text):
        calls["analysis"].append(text)
        return {"narrative": "## Summary\nMock analysis.", "summary": "", "keywords": [], "entities": {}}

    def fake_chat(user_message, history=None, document_context=None):
        calls["chat"].append({"message": user_message, "context": document_context})
        return "Mock chat answer."

    monkeypatch.setattr("services.document_service.generate_text_analysis", fake_analysis)
    monkeypatch.setattr("services.document_service.generate_chat_reply", fake_chat)
    monkeypatch.setattr("services.chat_service.generate_chat_reply", fake_chat)
    return calls


def register(client, email="user1@example.com"):
    return client.post("/api/auth/register", json={
        "name": "Test User", "email": email,
        "password": "password123", "confirm_password": "password123",
    })


def upload(client, filename, content, **form):
    data = {"file": (io.BytesIO(content), filename)}
    data.update(form)
    return client.post("/api/upload", data=data, content_type="multipart/form-data")


# ------------------------------------------------------------------ sample files
def make_txt(path, text="Hello research world. This is a plain text file."):
    path.write_text(text, encoding="utf-8")
    return str(path)


def make_pdf_bytes(text="InsightForge PDF test content about climate change."):
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_blank_pdf_bytes():
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def make_docx_bytes():
    import docx
    d = docx.Document()
    d.add_paragraph("Quarterly research report")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Region"
    table.cell(0, 1).text = "Revenue"
    table.cell(1, 0).text = "North"
    table.cell(1, 1).text = "1200"
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def make_xlsx_bytes():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Region", "Units", "Price"])
    ws.append(["North", 10, 2.5])
    ws.append(["South", 30, 4.0])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["Comment"])
    ws2.append(["Check the South numbers"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ================================================================== LOADER
class TestLoader:
    def test_txt_utf8(self, tmp_path):
        p = make_txt(tmp_path / "a.txt", "Résumé of the study: café data.")
        assert "café" in load_document(p, "txt")["text"]

    def test_txt_latin1_fallback(self, tmp_path):
        p = tmp_path / "b.txt"
        p.write_bytes("Müller measured 5 cm".encode("latin-1"))
        assert "Müller" in load_document(str(p), "txt")["text"]

    def test_txt_utf16_with_bom(self, tmp_path):
        p = tmp_path / "u16.txt"
        p.write_bytes("Unicode study notes".encode("utf-16"))
        assert "Unicode study notes" in load_document(str(p), "txt")["text"]

    def test_txt_binary_rejected(self, tmp_path):
        p = tmp_path / "c.txt"
        p.write_bytes(b"\x00\x01\x02\x03" * 100)
        with pytest.raises(DocumentError):
            load_document(str(p), "txt")

    def test_txt_empty_rejected(self, tmp_path):
        p = make_txt(tmp_path / "d.txt", "   \n  ")
        with pytest.raises(DocumentError):
            load_document(p, "txt")

    def test_pdf_text_extracted(self, tmp_path):
        p = tmp_path / "a.pdf"
        p.write_bytes(make_pdf_bytes())
        result = load_document(str(p), "pdf")
        assert "climate change" in result["text"]
        assert result["meta"]["pages"] == 1

    def test_pdf_corrupt_rejected(self, tmp_path):
        p = tmp_path / "bad.pdf"
        p.write_bytes(b"%PDF-1.4 this is not really a pdf")
        with pytest.raises(DocumentError):
            load_document(str(p), "pdf")

    def test_pdf_without_text_mentions_ocr(self, tmp_path):
        p = tmp_path / "blank.pdf"
        p.write_bytes(make_blank_pdf_bytes())
        with pytest.raises(DocumentError) as exc:
            load_document(str(p), "pdf")
        assert "OCR" in str(exc.value)

    def test_docx_paragraphs_and_tables(self, tmp_path):
        p = tmp_path / "a.docx"
        p.write_bytes(make_docx_bytes())
        text = load_document(str(p), "docx")["text"]
        assert "Quarterly research report" in text
        assert "Region | Revenue" in text
        assert "North | 1200" in text

    def test_docx_corrupt_rejected(self, tmp_path):
        p = tmp_path / "bad.docx"
        p.write_bytes(b"PK this is not a real docx")
        with pytest.raises(DocumentError):
            load_document(str(p), "docx")

    def test_xlsx_sheets_columns_and_numbers(self, tmp_path):
        p = tmp_path / "a.xlsx"
        p.write_bytes(make_xlsx_bytes())
        result = load_document(str(p), "xlsx")
        text = result["text"]
        assert result["meta"]["sheets"] == 2
        assert "Sheet: Sales" in text and "Sheet: Notes" in text
        assert "Columns: Region, Units, Price" in text
        assert "Units: min=10, max=30" in text

    def test_xlsx_corrupt_rejected(self, tmp_path):
        p = tmp_path / "bad.xlsx"
        p.write_bytes(b"PK not an excel file")
        with pytest.raises(DocumentError):
            load_document(str(p), "xlsx")

    def test_long_text_is_truncated(self, tmp_path):
        p = make_txt(tmp_path / "long.txt", "word " * (MAX_EXTRACTED_CHARS // 4))
        result = load_document(p, "txt")
        assert result["truncated"] is True
        assert len(result["text"]) <= MAX_EXTRACTED_CHARS

    def test_unknown_type_rejected(self, tmp_path):
        with pytest.raises(DocumentError):
            load_document(str(tmp_path / "x.exe"), "exe")


# ================================================================== UPLOAD ENDPOINT
class TestUpload:
    def test_requires_login(self, client):
        resp = upload(client, "a.txt", b"hello there")
        assert resp.status_code == 401

    def test_txt_upload_runs_analysis(self, client, mock_ai):
        register(client)
        resp = upload(client, "notes.txt", b"Research notes about renewable energy adoption.")
        assert resp.status_code == 201
        data = resp.get_json()["data"]
        assert data["document"]["filename"] == "notes.txt"
        assert "Uploaded:" in data["user_message"]["content"]
        assert "Mock analysis" in data["ai_message"]["content"]
        assert len(mock_ai["analysis"]) == 1
        assert "renewable energy" in mock_ai["analysis"][0]

    def test_upload_creates_conversation_with_two_messages(self, client):
        register(client)
        resp = upload(client, "notes.txt", b"Some real text for analysis.")
        convo_id = resp.get_json()["data"]["conversation"]["id"]
        msgs = client.get(f"/api/chat/{convo_id}/messages").get_json()["data"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]

    def test_pdf_docx_xlsx_all_work(self, client):
        register(client)
        assert upload(client, "a.pdf", make_pdf_bytes()).status_code == 201
        assert upload(client, "a.docx", make_docx_bytes()).status_code == 201
        assert upload(client, "a.xlsx", make_xlsx_bytes()).status_code == 201

    def test_instruction_uses_chat_with_document_context(self, client, mock_ai):
        register(client)
        upload(client, "notes.txt", b"The study found a 12 percent increase.",
               message="What was the increase?")
        assert mock_ai["analysis"] == []
        assert mock_ai["chat"][0]["message"] == "What was the increase?"
        assert "12 percent" in mock_ai["chat"][0]["context"]

    def test_spreadsheet_uses_chat_path(self, client, mock_ai):
        register(client)
        resp = upload(client, "data.xlsx", make_xlsx_bytes())
        assert resp.status_code == 201
        assert mock_ai["analysis"] == []
        assert "Columns: Region, Units, Price" in mock_ai["chat"][0]["context"]

    def test_follow_up_chat_receives_document_text(self, client, mock_ai):
        register(client)
        convo_id = upload(client, "notes.txt", b"Tidal energy produced 450 gigawatt hours.") \
            .get_json()["data"]["conversation"]["id"]
        mock_ai["chat"].clear()
        resp = client.post("/api/chat", json={"message": "How much energy?", "conversation_id": convo_id})
        assert resp.status_code == 200
        assert "450 gigawatt" in mock_ai["chat"][0]["context"]

    def test_upload_into_existing_conversation(self, client):
        register(client)
        convo_id = client.post("/api/chat", json={"message": "hi"}).get_json()["data"]["conversation"]["id"]
        resp = upload(client, "notes.txt", b"More text to read here.", conversation_id=str(convo_id))
        assert resp.status_code == 201
        assert resp.get_json()["data"]["conversation"]["id"] == convo_id

    def test_long_document_adds_truncation_note(self, client):
        register(client)
        resp = upload(client, "long.txt", b"word " * 6000)
        assert "first 12,000 characters" in resp.get_json()["data"]["ai_message"]["content"]


# ================================================================== VALIDATION & SECURITY
class TestValidation:
    def test_no_file(self, client):
        register(client)
        resp = client.post("/api/upload", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_unsupported_extension(self, client):
        register(client)
        resp = upload(client, "virus.exe", b"MZ....")
        assert resp.status_code == 400
        assert "Unsupported" in resp.get_json()["message"]

    def test_images_not_yet_supported(self, client):
        register(client)
        resp = upload(client, "photo.png", b"\x89PNG\r\n")
        assert resp.status_code == 400
        assert "later update" in resp.get_json()["message"]

    def test_fake_pdf_rejected(self, client):
        register(client)
        resp = upload(client, "fake.pdf", b"just some text pretending to be a pdf")
        assert resp.status_code == 400

    def test_fake_docx_rejected(self, client):
        register(client)
        assert upload(client, "fake.docx", b"plain text, not a zip").status_code == 400

    def test_empty_file_rejected(self, client):
        register(client)
        assert upload(client, "empty.txt", b"").status_code == 400

    def test_unreadable_pdf_gives_clean_error_and_leaves_no_file(self, client, app):
        register(client)
        resp = upload(client, "bad.pdf", b"%PDF-1.4 broken")
        assert resp.status_code == 422
        # nothing should remain in the user's folder
        user_dir = os.path.join(app.config["UPLOAD_FOLDER"], "1")
        assert not os.path.exists(user_dir) or os.listdir(user_dir) == []

    def test_invalid_conversation_id(self, client):
        register(client)
        resp = upload(client, "a.txt", b"some text here", conversation_id="abc")
        assert resp.status_code == 400

    def test_unknown_conversation_404_and_no_file_stored(self, client, app):
        register(client)
        resp = upload(client, "a.txt", b"some text here", conversation_id="999")
        assert resp.status_code == 404
        user_dir = os.path.join(app.config["UPLOAD_FOLDER"], "1")
        assert not os.path.exists(user_dir) or os.listdir(user_dir) == []

    def test_too_large_returns_json_413(self, client, app):
        register(client)
        app.config["MAX_CONTENT_LENGTH"] = 1024
        resp = upload(client, "big.txt", b"x" * 5000)
        assert resp.status_code == 413
        assert resp.get_json()["status"] == "error"

    def test_malicious_filename_cannot_escape_uploads_folder(self, client, app):
        register(client)
        resp = upload(client, "../../evil name.txt", b"harmless text content")
        assert resp.status_code == 201
        doc = resp.get_json()["data"]["document"]
        assert "/" not in doc["filename"] and ".." not in doc["filename"]
        base = os.path.abspath(app.config["UPLOAD_FOLDER"])
        for root, _dirs, files in os.walk(os.path.dirname(base)):
            for f in files:
                assert os.path.abspath(os.path.join(root, f)).startswith(os.path.dirname(base))
        stored = os.listdir(os.path.join(base, "1"))
        assert len(stored) == 1 and stored[0].endswith(".txt") and "evil" not in stored[0]

    def test_non_ascii_filename_keeps_extension(self, client):
        register(client)
        resp = upload(client, "रिपोर्ट.txt", b"some content to read")
        assert resp.status_code == 201
        assert resp.get_json()["data"]["document"]["filename"].endswith(".txt")


# ================================================================== AI FAILURE
class TestAIFailure:
    def test_ai_failure_keeps_document_and_returns_conversation_id(self, client, monkeypatch):
        register(client)

        def broken(text):
            raise AIServiceError("The AI service is currently rate-limited.")

        monkeypatch.setattr("services.document_service.generate_text_analysis", broken)
        resp = upload(client, "notes.txt", b"Some text that should be saved anyway.")
        assert resp.status_code == 503
        body = resp.get_json()
        assert "rate-limited" in body["message"]
        assert body["conversation_id"] is not None

        docs = client.get("/api/documents").get_json()["data"]
        assert len(docs) == 1


# ================================================================== OWNERSHIP
class TestOwnership:
    def test_list_shows_only_own_documents(self, client):
        register(client, "user1@example.com")
        upload(client, "mine.txt", b"my private document text")
        client.post("/api/auth/logout")
        register(client, "user2@example.com")
        assert client.get("/api/documents").get_json()["data"] == []

    def test_cannot_delete_someone_elses_document(self, client):
        register(client, "user1@example.com")
        doc_id = upload(client, "mine.txt", b"my private document text").get_json()["data"]["document"]["id"]
        client.post("/api/auth/logout")
        register(client, "user2@example.com")
        assert client.delete(f"/api/documents/{doc_id}").status_code == 404

    def test_cannot_upload_into_someone_elses_conversation(self, client):
        register(client, "user1@example.com")
        convo_id = upload(client, "mine.txt", b"my private text").get_json()["data"]["conversation"]["id"]
        client.post("/api/auth/logout")
        register(client, "user2@example.com")
        resp = upload(client, "x.txt", b"hijack attempt text", conversation_id=str(convo_id))
        assert resp.status_code == 404

    def test_other_users_document_text_never_reaches_my_chat(self, client, mock_ai):
        register(client, "user1@example.com")
        upload(client, "secret.txt", b"TOP SECRET payroll data")
        client.post("/api/auth/logout")
        register(client, "user2@example.com")
        mock_ai["chat"].clear()
        client.post("/api/chat", json={"message": "What is in the payroll file?"})
        assert mock_ai["chat"][0]["context"] is None

    def test_delete_removes_row_and_file(self, client, app):
        register(client)
        doc_id = upload(client, "mine.txt", b"text to be deleted soon").get_json()["data"]["document"]["id"]
        user_dir = os.path.join(app.config["UPLOAD_FOLDER"], "1")
        assert len(os.listdir(user_dir)) == 1

        assert client.delete(f"/api/documents/{doc_id}").status_code == 200
        assert os.listdir(user_dir) == []
        assert client.get("/api/documents").get_json()["data"] == []