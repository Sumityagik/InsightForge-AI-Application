"""
Sprint 6 tests: chunking, embeddings, the FAISS vector store, and the full
retrieval-augmented question answering flow.

Run with: python -m pytest tests/test_rag.py -v

The tests use the lightweight "hash" embedder (set in tests/conftest.py), so
they need no model download and no internet. That means they check the
plumbing (chunks, index, retrieval, security, fallbacks) but NOT how good the
real sentence-transformer model is at matching meaning. Check that by hand.
"""

import io
import os
import re

import numpy as np
import pytest

from app import create_app
from src.db import db
from src.chunker import split_text
from src import embeddings, vector_store
from src.embeddings import EmbeddingError, embed_texts
from src.vector_store import VectorStoreError
from services import rag_service
from services.rag_service import FULL_TEXT_LIMIT


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
    calls = {"analysis": [], "chat": []}

    def fake_analysis(text):
        calls["analysis"].append(text)
        return {"narrative": "Mock analysis.", "summary": "", "keywords": [], "entities": {}}

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


# Eight paragraphs on clearly different topics, each with its own vocabulary.
TOPICS = {
    "solar": "photovoltaic panels sunlight silicon inverter rooftop irradiance",
    "wind": "turbine blades rotor gearbox offshore gusts nacelle",
    "hydro": "reservoir dam spillway penstock watershed hydroelectric",
    "geothermal": "magma steam wells reservoirs drilling underground heat",
    "nuclear": "uranium reactor fission cooling containment isotopes",
    "battery": "lithium cathode anode electrolyte charging cycles",
    "hydrogen": "electrolyzer fuel cell compression pipeline storage tanks",
    "biomass": "crops compost methane digesters pellets forestry residue",
}


def long_document():
    paragraphs = []
    for topic, words in TOPICS.items():
        sentence = f"The {topic} project studies {words}. "
        paragraphs.append(f"Section on {topic}. " + sentence * 40)
    text = "\n\n".join(paragraphs)
    assert len(text) > FULL_TEXT_LIMIT
    return text


def make_doc_and_get_ids(client, text=None, name="energy.txt", **form):
    text = text if text is not None else long_document()
    resp = upload(client, name, text.encode("utf-8"), **form)
    assert resp.status_code == 201, resp.get_json()
    data = resp.get_json()["data"]
    return data["conversation"]["id"], data["document"]["id"]


def indexes_folder(app):
    return os.path.join(app.config["UPLOAD_FOLDER"], "indexes")


# ================================================================== CHUNKER
class TestChunker:
    def test_empty_text(self):
        assert split_text("") == []
        assert split_text("   \n\n  ") == []

    def test_short_text_is_one_chunk(self):
        assert split_text("A short note.") == ["A short note."]

    def test_chunks_respect_size_limit(self):
        text = long_document()
        chunks = split_text(text, chunk_size=500, overlap=100)
        assert len(chunks) > 3
        assert all(len(c) <= 500 + 100 + 1 for c in chunks)
        assert all(c.strip() for c in chunks)

    def test_every_word_survives(self):
        text = long_document()
        chunks = split_text(text, chunk_size=400, overlap=80)
        joined = " ".join(chunks)
        for word in set(re.findall(r"\w+", text)):
            assert word in joined

    def test_neighbouring_chunks_overlap(self):
        text = " ".join(f"word{i}" for i in range(600))
        chunks = split_text(text, chunk_size=300, overlap=80)
        assert len(chunks) > 2
        for a, b in zip(chunks, chunks[1:]):
            assert set(a.split()) & set(b.split())

    def test_deterministic(self):
        text = long_document()
        assert split_text(text) == split_text(text)

    def test_long_paragraph_splits_at_sentences(self):
        text = "This is a sentence. " * 200  # one 4000-character line
        chunks = split_text(text, chunk_size=500, overlap=50)
        assert len(chunks) > 5
        assert all(c.rstrip().endswith(".") for c in chunks)

    def test_huge_word_is_cut(self):
        chunks = split_text("x" * 2500, chunk_size=1000, overlap=100)
        assert len(chunks) == 3
        assert all(len(c) <= 1000 + 100 + 1 for c in chunks)
        # chunks overlap by design, so the total is at least the original length
        assert "".join(chunks).count("x") >= 2500

    def test_line_breaks_are_kept(self):
        rows = "\n".join(f"row{i},north,{i}" for i in range(20))
        chunks = split_text(rows, chunk_size=1000)
        assert chunks == [rows]

    def test_unicode(self):
        chunks = split_text("Résumé de l'étude. 研究の概要です。 " * 80, chunk_size=300, overlap=50)
        assert chunks and all(len(c) <= 351 for c in chunks)

    def test_bad_parameters(self):
        with pytest.raises(ValueError):
            split_text("x", chunk_size=0)
        with pytest.raises(ValueError):
            split_text("x", chunk_size=100, overlap=100)
        with pytest.raises(ValueError):
            split_text("x", chunk_size=100, overlap=-1)


# ================================================================== EMBEDDINGS
class TestEmbeddings:
    def test_shape_and_normalised(self):
        v = embed_texts(["solar panels", "wind turbines", "nuclear reactor"])
        assert v.shape == (3, embeddings.HASH_DIMENSIONS)
        assert v.dtype == np.float32
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)

    def test_deterministic(self):
        assert np.array_equal(embed_texts(["same text"]), embed_texts(["same text"]))

    def test_related_text_scores_higher(self):
        a, b, c = embed_texts([
            "solar panels convert sunlight into electricity",
            "photovoltaic solar panels and sunlight",
            "nuclear reactor uranium fission",
        ])
        assert float(a @ b) > float(a @ c)

    def test_empty_text_gives_zero_vector_not_nan(self):
        v = embed_texts(["", "!!"])
        assert not np.isnan(v).any()

    def test_no_texts(self):
        assert embed_texts([]).shape[0] == 0

    def test_unknown_backend(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "magic")
        with pytest.raises(EmbeddingError):
            embed_texts(["x"])

    def test_real_model_failure_becomes_embedding_error(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BACKEND", "sentence-transformers")

        def boom():
            raise OSError("no internet")

        monkeypatch.setattr("src.registry.get_embedding_model", boom)
        with pytest.raises(EmbeddingError):
            embed_texts(["x"])


# ================================================================== VECTOR STORE
class TestVectorStore:
    def test_search_returns_best_first(self, tmp_path):
        vectors = np.eye(4, dtype="float32")  # four perpendicular directions
        path = str(tmp_path / "a" / "idx.faiss")
        vector_store.save_index(vectors, path)
        hits = vector_store.search(path, np.array([[0, 0, 1, 0]], dtype="float32"), k=2)
        assert hits[0][0] == 2 and hits[0][1] == pytest.approx(1.0)
        assert len(hits) == 2

    def test_k_larger_than_index(self, tmp_path):
        path = str(tmp_path / "idx.faiss")
        vector_store.save_index(np.eye(3, dtype="float32"), path)
        assert len(vector_store.search(path, np.ones((1, 3), dtype="float32"), k=10)) == 3

    def test_survives_reload_from_disk(self, tmp_path):
        path = str(tmp_path / "idx.faiss")
        vector_store.save_index(np.eye(3, dtype="float32"), path)
        assert vector_store.index_exists(path)
        again = vector_store.search(path, np.array([[1, 0, 0]], dtype="float32"), k=1)
        assert again[0][0] == 0

    def test_non_ascii_path(self, tmp_path):
        path = str(tmp_path / "रिपोर्ट" / "idx.faiss")
        vector_store.save_index(np.eye(2, dtype="float32"), path)
        assert vector_store.search(path, np.array([[0, 1]], dtype="float32"), k=1)[0][0] == 1

    def test_no_temp_file_left_behind(self, tmp_path):
        path = str(tmp_path / "idx.faiss")
        vector_store.save_index(np.eye(2, dtype="float32"), path)
        assert os.listdir(tmp_path) == ["idx.faiss"]

    def test_overwrite_replaces_index(self, tmp_path):
        path = str(tmp_path / "idx.faiss")
        vector_store.save_index(np.eye(2, dtype="float32"), path)
        vector_store.save_index(np.eye(5, dtype="float32"), path)
        assert len(vector_store.search(path, np.ones((1, 5), dtype="float32"), k=10)) == 5

    def test_wrong_dimension_is_a_clear_error(self, tmp_path):
        path = str(tmp_path / "idx.faiss")
        vector_store.save_index(np.eye(3, dtype="float32"), path)
        with pytest.raises(VectorStoreError):
            vector_store.search(path, np.ones((1, 7), dtype="float32"), k=1)

    def test_no_vectors_rejected(self, tmp_path):
        with pytest.raises(VectorStoreError):
            vector_store.save_index(np.zeros((0, 4), dtype="float32"), str(tmp_path / "x.faiss"))

    def test_corrupt_file_rejected(self, tmp_path):
        path = tmp_path / "bad.faiss"
        path.write_bytes(b"this is not an index")
        with pytest.raises(VectorStoreError):
            vector_store.search(str(path), np.ones((1, 3), dtype="float32"), k=1)

    def test_delete(self, tmp_path):
        path = str(tmp_path / "idx.faiss")
        vector_store.save_index(np.eye(2, dtype="float32"), path)
        vector_store.delete_index(path)
        assert not vector_store.index_exists(path)
        vector_store.delete_index(path)  # deleting twice is harmless


# ================================================================== INDEXING ON UPLOAD
class TestIndexing:
    def test_upload_creates_chunks_and_index_file(self, client, app):
        register(client)
        _convo, doc_id = make_doc_and_get_ids(client)
        with app.app_context():
            from models import DocumentChunk, Document
            rows = DocumentChunk.query.filter_by(document_id=doc_id).order_by(DocumentChunk.chunk_index).all()
            assert len(rows) > 3
            assert [r.chunk_index for r in rows] == list(range(len(rows)))
            doc = db.session.get(Document, doc_id)
            assert os.path.isfile(rag_service.index_path(doc))
            assert "indexes" in rag_service.index_path(doc) and f"{doc.user_id}" in rag_service.index_path(doc)

    def test_index_files_are_not_mixed_into_the_uploads_folder(self, client, app):
        register(client)
        make_doc_and_get_ids(client)
        user_files = os.listdir(os.path.join(app.config["UPLOAD_FOLDER"], "1"))
        assert len(user_files) == 1 and user_files[0].endswith(".txt")

    def test_indexing_failure_does_not_break_upload(self, client, monkeypatch):
        register(client)

        def boom(texts):
            raise EmbeddingError("model unavailable")

        monkeypatch.setattr("services.rag_service.embed_texts", boom)
        resp = upload(client, "long.txt", long_document().encode())
        assert resp.status_code == 201
        # Without an index the note must not promise a whole-document search.
        assert "only the first" in resp.get_json()["data"]["ai_message"]["content"]

    def test_missing_faiss_package_does_not_break_upload_or_chat(self, client, monkeypatch, mock_ai):
        register(client)

        def no_faiss():
            raise VectorStoreError("The 'faiss-cpu' package is not installed on this server.")

        monkeypatch.setattr("src.vector_store._faiss", no_faiss)
        resp = upload(client, "long.txt", long_document().encode())
        assert resp.status_code == 201
        convo_id = resp.get_json()["data"]["conversation"]["id"]
        mock_ai["chat"].clear()
        chat = client.post("/api/chat", json={"message": "hydrogen tanks", "conversation_id": convo_id})
        assert chat.status_code == 200
        assert mock_ai["chat"][0]["context"].startswith("[Document: long.txt]")

    def test_successful_index_changes_the_long_document_note(self, client):
        register(client)
        resp = upload(client, "long.txt", long_document().encode() * 3)
        content = resp.get_json()["data"]["ai_message"]["content"]
        assert "search the whole document" in content


# ================================================================== QUESTION ANSWERING
class TestRetrieval:
    def ask(self, client, convo_id, question):
        resp = client.post("/api/chat", json={"message": question, "conversation_id": convo_id})
        assert resp.status_code == 200, resp.get_json()
        return resp.get_json()["data"]["ai_message"]["content"]

    def test_long_document_sends_only_relevant_sections(self, client, mock_ai):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client)
        mock_ai["chat"].clear()

        self.ask(client, convo_id, "How does the hydrogen electrolyzer and storage tanks work?")
        context = mock_ai["chat"][0]["context"]

        assert "electrolyzer" in context
        # topics that sit far away from "hydrogen" in the document must not come along
        assert "photovoltaic" not in context and "uranium" not in context
        assert len(context) < len(long_document()) / 2

    def test_different_questions_get_different_sections(self, client, mock_ai):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client)

        mock_ai["chat"].clear()
        self.ask(client, convo_id, "Tell me about the uranium reactor and fission containment")
        nuclear = mock_ai["chat"][0]["context"]

        mock_ai["chat"].clear()
        self.ask(client, convo_id, "Tell me about turbine blades and the offshore nacelle")
        wind = mock_ai["chat"][0]["context"]

        assert "uranium" in nuclear and "turbine" not in nuclear
        assert "turbine" in wind and "uranium" not in wind

    def test_answer_shows_where_it_came_from(self, client):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client, name="energy.txt")
        answer = self.ask(client, convo_id, "What about lithium cathode charging cycles?")
        assert "Answered using" in answer and "energy.txt" in answer

    def test_excerpts_are_in_reading_order(self, client, mock_ai):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client)
        mock_ai["chat"].clear()
        self.ask(client, convo_id, "solar photovoltaic wind turbine hydro dam geothermal magma")
        numbers = [int(n) for n in re.findall(r"excerpt (\d+)\]", mock_ai["chat"][0]["context"])]
        assert numbers == sorted(numbers) and len(numbers) >= 2

    def test_short_document_is_sent_whole_without_source_line(self, client, mock_ai):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client, text="Tidal power produced 450 gigawatt hours last year.", name="tiny.txt")
        mock_ai["chat"].clear()
        answer = self.ask(client, convo_id, "How much power?")
        assert "450 gigawatt" in mock_ai["chat"][0]["context"]
        assert "Answered using" not in answer

    def test_document_just_under_the_limit_is_sent_whole(self, client, mock_ai):
        register(client)
        text = ("alpha " * 1000)[:FULL_TEXT_LIMIT]
        convo_id, _ = make_doc_and_get_ids(client, text=text, name="edge.txt")
        mock_ai["chat"].clear()
        self.ask(client, convo_id, "anything")
        assert text.strip() in mock_ai["chat"][0]["context"]

    def test_question_sent_with_upload_uses_retrieval(self, client, mock_ai):
        register(client)
        resp = upload(client, "energy.txt", long_document().encode(),
                      message="Explain the geothermal magma steam wells")
        assert resp.status_code == 201
        context = mock_ai["chat"][0]["context"]
        assert "magma" in context
        assert "biomass" not in context and "electrolyzer" not in context
        assert "Answered using" in resp.get_json()["data"]["ai_message"]["content"]

    def test_two_documents_are_searched_together(self, client, mock_ai):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client, name="first.txt")
        second = long_document().replace("solar", "tidal").replace("photovoltaic", "lagoon")
        resp = upload(client, "second.txt", second.encode(), conversation_id=str(convo_id))
        assert resp.status_code == 201

        mock_ai["chat"].clear()
        answer = self.ask(client, convo_id, "lagoon tidal and also electrolyzer hydrogen tanks")
        context = mock_ai["chat"][0]["context"]
        assert "first.txt" in context and "second.txt" in context
        assert "first.txt" in answer and "second.txt" in answer

    def test_document_from_another_conversation_is_not_used(self, client, mock_ai):
        register(client)
        make_doc_and_get_ids(client)
        other = client.post("/api/chat", json={"message": "hello there"}).get_json()["data"]["conversation"]["id"]
        mock_ai["chat"].clear()
        self.ask(client, other, "What about the hydrogen electrolyzer?")
        assert mock_ai["chat"][0]["context"] is None

    def test_other_users_documents_are_never_searched(self, client, mock_ai):
        register(client, "user1@example.com")
        make_doc_and_get_ids(client)
        client.post("/api/auth/logout")
        register(client, "user2@example.com")
        convo = client.post("/api/chat", json={"message": "hi"}).get_json()["data"]["conversation"]["id"]
        mock_ai["chat"].clear()
        self.ask(client, convo, "hydrogen electrolyzer tanks")
        assert mock_ai["chat"][0]["context"] is None


# ================================================================== FALLBACKS
class TestFallbacks:
    def test_retrieval_failure_falls_back_to_start_of_document(self, client, mock_ai, monkeypatch):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client)

        def boom(text):
            raise EmbeddingError("model unavailable")

        monkeypatch.setattr("services.rag_service.embed_query", boom)
        mock_ai["chat"].clear()
        resp = client.post("/api/chat", json={"message": "hydrogen electrolyzer", "conversation_id": convo_id})
        assert resp.status_code == 200
        assert mock_ai["chat"][0]["context"].startswith("[Document: energy.txt]")
        assert "Answered using" not in resp.get_json()["data"]["ai_message"]["content"]

    def test_corrupt_index_file_falls_back_instead_of_failing(self, client, app, mock_ai):
        register(client)
        convo_id, doc_id = make_doc_and_get_ids(client)
        with app.app_context():
            from models import Document
            path = rag_service.index_path(db.session.get(Document, doc_id))
        with open(path, "wb") as f:
            f.write(b"garbage")

        mock_ai["chat"].clear()
        resp = client.post("/api/chat", json={"message": "hydrogen electrolyzer", "conversation_id": convo_id})
        assert resp.status_code == 200
        assert mock_ai["chat"][0]["context"] is not None

    def test_missing_index_is_rebuilt_on_demand(self, client, app, mock_ai):
        """Documents uploaded before indexing existed get indexed on first question."""
        register(client)
        convo_id, doc_id = make_doc_and_get_ids(client)
        with app.app_context():
            from models import Document, DocumentChunk
            doc = db.session.get(Document, doc_id)
            path = rag_service.index_path(doc)
            os.remove(path)
            DocumentChunk.query.filter_by(document_id=doc_id).delete()
            db.session.commit()

        mock_ai["chat"].clear()
        resp = client.post("/api/chat", json={"message": "uranium reactor fission", "conversation_id": convo_id})
        assert resp.status_code == 200
        assert "uranium" in mock_ai["chat"][0]["context"]
        assert os.path.isfile(path)

    def test_chat_without_documents_still_works(self, client, mock_ai):
        register(client)
        resp = client.post("/api/chat", json={"message": "plain question"})
        assert resp.status_code == 200
        assert mock_ai["chat"][0]["context"] is None


# ================================================================== CLEAN-UP
class TestCleanup:
    def index_files(self, app):
        found = []
        for root, _dirs, files in os.walk(indexes_folder(app)):
            found.extend(files)
        return found

    def test_deleting_a_document_removes_its_index_and_chunks(self, client, app):
        register(client)
        _convo, doc_id = make_doc_and_get_ids(client)
        assert len(self.index_files(app)) == 1

        assert client.delete(f"/api/documents/{doc_id}").status_code == 200
        assert self.index_files(app) == []
        with app.app_context():
            from models import DocumentChunk
            assert DocumentChunk.query.filter_by(document_id=doc_id).count() == 0

    def test_deleting_a_conversation_removes_its_documents_files_and_indexes(self, client, app):
        register(client)
        convo_id, _ = make_doc_and_get_ids(client)
        assert client.delete(f"/api/chat/{convo_id}").status_code == 200

        assert client.get("/api/documents").get_json()["data"] == []
        assert self.index_files(app) == []
        assert os.listdir(os.path.join(app.config["UPLOAD_FOLDER"], "1")) == []
        with app.app_context():
            from models import DocumentChunk
            assert DocumentChunk.query.count() == 0

    def test_deleting_one_conversation_keeps_the_others_documents(self, client, app):
        register(client)
        convo_a, _ = make_doc_and_get_ids(client, name="a.txt")
        convo_b, doc_b = make_doc_and_get_ids(client, name="b.txt")
        client.delete(f"/api/chat/{convo_a}")
        docs = client.get("/api/documents").get_json()["data"]
        assert [d["id"] for d in docs] == [doc_b]
        assert len(self.index_files(app)) == 1

    def test_cannot_delete_someone_elses_conversation_documents(self, client, app):
        register(client, "user1@example.com")
        convo_id, _ = make_doc_and_get_ids(client)
        client.post("/api/auth/logout")
        register(client, "user2@example.com")
        assert client.delete(f"/api/chat/{convo_id}").status_code == 404
        assert len(self.index_files(app)) == 1


# ================================================================== SOURCE NOTE
class TestSourceNote:
    def test_wording(self):
        one = rag_service.RetrievalResult("x", "retrieval", ["a.pdf"], 1)
        many = rag_service.RetrievalResult("x", "retrieval", ["a.pdf", "b.pdf"], 4)
        assert "1 relevant section of: a.pdf" in rag_service.source_note(one)
        assert "4 relevant sections of: a.pdf, b.pdf" in rag_service.source_note(many)

    def test_only_shown_for_retrieval_mode(self):
        for mode in ("none", "full", "truncated"):
            assert rag_service.source_note(rag_service.RetrievalResult("x", mode, [], 0)) is None
