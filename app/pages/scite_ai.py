from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag.llm import LLMClient, LLMError
from rag.scite import SciteClient, SciteError, SciteMCPClient


def _init_state() -> None:
    if "scite_history" not in st.session_state:
        st.session_state["scite_history"] = []


def _render_references(references: list[dict]) -> None:
    if not references:
        return

    for number, reference in enumerate(references, start=1):
        title = reference.get("title") or reference.get("paper", {}).get("title") or "Untitled source"
        with st.expander(f"{number}. {title}", expanded=False):
            _render_field(reference, "Abstract", ("abstract", "abstractText", "abstracts"))
            _render_field(
                reference,
                "Evidence excerpts",
                ("excerpt", "excerpts", "snippet", "snippets", "context", "highlights"),
            )


def _evidence_records(references: list[dict]) -> list[dict]:
    records = []
    for number, reference in enumerate(references, start=1):
        abstract = _find_values(reference, ("abstract", "abstractText", "abstracts"))
        excerpts = _find_values(
            reference,
            ("excerpt", "excerpts", "snippet", "snippets", "context", "highlights"),
        )
        records.append(
            {
                "source_id": number,
                "abstract": _unique_text(_text_values(abstract)),
                "evidence_excerpts": _unique_text(_text_values(excerpts)),
            }
        )
    return [record for record in records if record["abstract"] or record["evidence_excerpts"]]


def _save_evidence_jsonl(records: list[dict]) -> None:
    directory = Path(__file__).resolve().parents[2] / "logs" / "scite"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = directory / f"evidence-{timestamp}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _unique_text(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _record_text(record: dict) -> str:
    parts = [f"[{record['source_id']}]" ]
    if record["abstract"]:
        parts.append("ABSTRACT\n" + "\n".join(record["abstract"]))
    if record["evidence_excerpts"]:
        excerpts = "\n".join(record["evidence_excerpts"])
        parts.append("EVIDENCE EXCERPTS\n" + excerpts[:4500])
    return "\n".join(parts)


def _pack_records(records: list[dict], max_chars: int = 12000) -> list[str]:
    batches = []
    current = []
    current_size = 0
    for record in records:
        text = _record_text(record)
        if current and current_size + len(text) > max_chars:
            batches.append("\n\n".join(current))
            current, current_size = [], 0
        current.append(text)
        current_size += len(text)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _summarize_with_qwen(question: str, references: list[dict]) -> str:
    records = _evidence_records(references)
    _save_evidence_jsonl(records)
    if not records:
        return "No abstract or evidence excerpt was returned for these sources."
    client = LLMClient.from_config()
    map_system = (
        "Summarize only the numbered scholarly evidence supplied by the user. "
        "Do not use outside knowledge or invent details. Keep the summary under "
        "120 words and end claims with source numbers like [1]."
    )
    partials = []
    for batch in _pack_records(records):
        partials.append(
            client.chat(
                map_system,
                f"QUESTION\n{question.strip()}\n\nEVIDENCE\n{batch}\n\n"
                "Write the short evidence summary now.",
            )
        )
    if len(partials) == 1:
        return partials[0]
    reduce_system = (
        "Combine the numbered evidence summaries into one concise answer. Use "
        "only their content, remove repetition, do not invent facts, and keep "
        "source citations such as [1]. Stay under 180 words."
    )
    return client.chat(
        reduce_system,
        f"QUESTION\n{question.strip()}\n\nPARTIAL SUMMARIES\n"
        + "\n\n".join(partials)
        + "\n\nWrite the final evidence-grounded summary.",
    )


def _render_field(reference: dict, label: str, keys: tuple[str, ...]) -> None:
    values = _find_values(reference, keys)
    if not values:
        return
    st.markdown(f"**{label}**")
    for value in values:
        _render_text_value(value)


def _render_text_value(value) -> None:
    if isinstance(value, list):
        for item in value:
            _render_text_value(item)
        return
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or value.get("excerpt")
        if text:
            _render_text_value(text)
        else:
            for item in value.values():
                _render_text_value(item)
        return
    st.markdown(str(value))


def _text_values(values: list) -> list[str]:
    text = []
    for value in values:
        if isinstance(value, list):
            text.extend(_text_values(value))
        elif isinstance(value, dict):
            child = value.get("text") or value.get("content") or value.get("excerpt")
            text.extend(_text_values([child] if child else list(value.values())))
        elif value not in (None, ""):
            text.append(str(value))
    return text


def _find_values(value, keys: tuple[str, ...]) -> list:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and child not in (None, "", [], {}):
                found.append(child)
            found.extend(_find_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_values(child, keys))
    return found


def _render_history() -> None:
    _init_state()
    for turn in st.session_state["scite_history"]:
        with st.chat_message(turn["role"]):
            if turn["role"] == "user":
                st.markdown(turn["content"])
                continue

            if turn.get("content"):
                st.markdown("**Evidence summary**")
                st.markdown(turn["content"])
            if turn.get("error"):
                st.error(turn["error"])
            _render_references(turn.get("references", []))


def render() -> None:
    st.title("Scite AI")
    st.caption(
        "Ask a research question. Answers will be grounded in scholarly sources "
        "and displayed with their references."
    )

    client = SciteClient.from_env()
    mcp_client = SciteMCPClient.from_env()
    with st.sidebar:
        st.subheader("Assistant settings")
        st.selectbox(
            "Answer length",
            ["short", "medium", "long"],
            index=1,
            key="scite_answer_length",
        )
        st.selectbox(
            "Citation style",
            ["apa", "ieee", "mla", "ama", "chicago", "harvard", "vancouver"],
            index=0,
            key="scite_citation_style",
        )
        st.toggle("Use scholarly references", value=True, key="scite_use_references")
        if client:
            st.success("Scite API connected")
        elif mcp_client:
            st.success("Scite MCP connected")
        else:
            st.info("Add SCITE_API_KEY or SCITE_MCP_API_KEY to .env to enable literature answers.")
        if st.button("Clear conversation", use_container_width=True, key="clear_scite"):
            st.session_state["scite_history"] = []
            st.rerun()

    _render_history()
    question_nonce = st.session_state.get("scite_question_nonce", 0)
    question = st.text_area(
        "Ask a research question...",
        key=f"scite_question_{question_nonce}",
        placeholder="Type your question, then press Send.",
        height=100,
    )
    if not st.button("Send", use_container_width=True, key="send_scite"):
        return
    if not question or not question.strip():
        st.warning("Enter a question before sending.")
        return

    _init_state()
    error = None
    st.session_state["scite_history"].append({"role": "user", "content": question.strip()})
    with st.chat_message("assistant"):
        with st.spinner("Searching Scite..."):
            if not client:
                if mcp_client:
                    try:
                        result = mcp_client.search(question.strip())
                        content, references, details, status = (
                            result.content,
                            result.references,
                            result.details,
                            "searched",
                        )
                    except SciteError as exc:
                        content, references, details, status = "", [], None, "error"
                        error = f"Scite request failed: {exc}"
                else:
                    content, references, details, status = "", [], None, "error"
                    error = "Scite is not connected. Add SCITE_API_KEY or SCITE_MCP_API_KEY to .env and restart the app."
            else:
                try:
                    history = st.session_state["scite_history"][:-1]
                    result = client.ask(
                        question.strip(),
                        history,
                        answer_length=st.session_state["scite_answer_length"],
                        citation_style=st.session_state["scite_citation_style"],
                        use_references=st.session_state["scite_use_references"],
                    )
                    content, references, details, status = (
                        result.content,
                        result.references,
                        result.details,
                        "answered",
                    )
                except SciteError as exc:
                    if mcp_client:
                        try:
                            result = mcp_client.search(question.strip())
                            content, references, details, status = (
                                result.content,
                                result.references,
                                result.details,
                                "searched",
                            )
                        except SciteError as mcp_exc:
                            content, references, details, status = "", [], None, "error"
                            error = f"Scite request failed: {mcp_exc}"
                    else:
                        content, references, details, status = "", [], None, "error"
                        error = f"Scite request failed: {exc}"
            if references:
                with st.spinner("Summarizing evidence with Qwen... Please wait"):
                    try:
                        summary = _summarize_with_qwen(question.strip(), references)
                    except LLMError as exc:
                        content = ""
                        error = f"Qwen summary failed: {exc}"
                    else:
                        content = summary
    st.session_state["scite_history"].append(
        {
            "role": "assistant",
            "content": content,
            "references": references,
            "details": details,
            "status": status,
            "error": error,
        }
    )
    st.session_state["scite_question_nonce"] = question_nonce + 1
    st.rerun()


render()
