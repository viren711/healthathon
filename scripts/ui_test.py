#!/usr/bin/env python3
"""UI regression tests for the portal, driven by Streamlit's AppTest harness.

scripts/smoke_test.py covers the pipeline but never renders a page, so it
cannot catch interaction bugs. This does: it clicks the real buttons and
asserts what ends up in the element tree.

It exists because of one specific bug that no amount of pipeline testing would
have found. "Go to section" rendered its page correctly and then hid it,
because the button lives inside an expander, clicking a button reruns the
script, and the expander was rebuilt with a hard-coded `expanded=False`. The
page was there in the element tree the whole time, inside a container that had
snapped shut. To the user it looked like a dead button.

    python scripts/ui_test.py            # everything (Ask and drafting need the LLM)
    python scripts/ui_test.py --no-llm   # no model required
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The entrypoint is now the home page; each tool is its own view, driven
# directly so the tests do not depend on navigating through the home page.
ENTRY = str(ROOT / "app" / "streamlit_app.py")
APP = str(ROOT / "app" / "views" / "oncology_reference.py")
STUDIO = str(ROOT / "app" / "views" / "content_studio.py")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'  ok  ' if condition else ' FAIL '}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def scope_word_in(text: str) -> bool:
    return "Searching" in text


def goto_buttons(at):
    return [i for i, b in enumerate(at.button) if "Go to section" in b.label]


def goto_page(at, page_name: str) -> bool:
    page_hash = next(
        (
            page_hash
            for page_hash, info in at._registered_pages.items()
            if info.get("page_name") == page_name
        ),
        None,
    )
    if page_hash is None:
        return False
    at._page_hash = page_hash
    at.run()
    return True


def check_landing_and_pages(AppTest) -> None:
    print("\n== Landing and assistant pages ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    # Default page is now Oncology Reference
    check("app opens to Oncology Reference", [title.value for title in at.title] == ["Oncology Reference"],
          str([title.value for title in at.title]))

    # Navigate to Scite AI page
    at.switch_page("pages/scite_ai.py").run()
    check("Scite page opens", [title.value for title in at.title] == ["Scite AI"],
          str([title.value for title in at.title]))
    check("Scite page has a question editor", len(at.text_area) == 1,
          f"{len(at.text_area)} editor(s)")
    check("Scite page has assistant settings", len(at.sidebar.selectbox) == 2,
          f"{len(at.sidebar.selectbox)} setting(s)")
    at.text_area[0].set_value("What is Scite?").run()
    at.button(key="send_scite").click().run()
    check("Scite page returns an assistant response", len(at.chat_message) >= 2,
          f"{len(at.chat_message)} message(s)")
    check("Scite question field clears after summarizing", at.text_area[0].value == "",
          repr(at.text_area[0].value))
    check(
        "Scite page handles the response without an exception",
        not at.exception,
        f"{len(at.expander)} source card(s)",
    )
    check("Scite page has no exception", not at.exception)

    # Navigate back to Oncology Reference using its page hash
    onco_hash = next(
        h for h, info in at._registered_pages.items()
        if info.get("page_name") == "Oncology Reference"
    )
    at._page_hash = onco_hash
    at.run()
    check("Oncology Reference page reopens", [title.value for title in at.title] == ["Oncology Reference"],
          str([title.value for title in at.title]))
    check("Oncology Reference has no exception", not at.exception)


def test_find_mode(AppTest) -> None:
    print("\n== Find mode ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    check("app starts without an exception", not at.exception)
    check("oncology reference page opens", goto_page(at, "Oncology Reference"),
          str(list(at._registered_pages.values())))
    if [title.value for title in at.title] != ["Oncology Reference"]:
        return

    at.radio[0].set_value("Find a topic").run()
    at.text_input[0].set_value("brachytherapy").run()
    check("locator returns results", len(goto_buttons(at)) > 0,
          f"{len(goto_buttons(at))} places")
    check("no page is open before clicking", len(at.image) == 0)

    idx = goto_buttons(at)[0]
    at.button[idx].click().run()
    check("clicking 'Go to section' renders the page", len(at.image) == 1,
          f"{len(at.image)} image(s)")
    check("no exception on opening the page", not at.exception)
    check("a close control appears", any("Close" in b.label for b in at.button))
    check("page navigation appears", any("Prev" in b.label for b in at.button)
          and any("Next" in b.label for b in at.button))

    # A nonsense topic must find nothing rather than the nearest five passages.
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    check("second app instance opens oncology reference", goto_page(at2, "Oncology Reference"),
          str(list(at2._registered_pages.values())))
    if [title.value for title in at2.title] != ["Oncology Reference"]:
        return
    at2.radio[0].set_value("Find a topic").run()
    at2.text_input[0].set_value("What is the capital of France?").run()
    check("an out-of-corpus topic returns no places", len(goto_buttons(at2)) == 0,
          f"{len(goto_buttons(at2))} places")


def test_scope_selector(AppTest) -> None:
    print("\n== Section selector ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    check("oncology reference page opens", goto_page(at, "Oncology Reference"),
          str(list(at._registered_pages.values())))
    if [title.value for title in at.title] != ["Oncology Reference"]:
        return
    labels = [sb.label for sb in at.selectbox]
    check("an organ-system selector exists", "Organ system" in labels, str(labels))
    if "Organ system" not in labels:
        return

    sysbox = at.selectbox[labels.index("Organ system")]
    opts = list(sysbox.options)
    check("selector offers every organ system", len(opts) >= 10, f"{len(opts)} options")
    hn = next((o for o in opts if o.startswith("Head & Neck")), None)
    check("Head & Neck is selectable", hn is not None, str(opts[:4]))
    if not hn:
        return

    sysbox.set_value(hn).run()
    labels = [sb.label for sb in at.selectbox]
    check("choosing a system reveals a pathology selector", "Pathology" in labels,
          str(labels))
    if "Pathology" not in labels:
        return

    pathbox = at.selectbox[labels.index("Pathology")]
    tongue = next((o for o in pathbox.options if "tongue" in o.lower()), None)
    check("'Oral cavity & tongue' is selectable", tongue is not None,
          str(list(pathbox.options)[:4]))
    if not tongue:
        return

    pathbox.set_value(tongue).run()
    check("the active section is confirmed to the user",
          any("Answering only from" in m.value for m in at.success), "")
    check("no exception when scoping", not at.exception)

    # An active scope must be visible in the MAIN panel, not just the sidebar.
    # A section left selected from an earlier question is otherwise invisible,
    # and the platform looks broken when it refuses something the books cover.
    check("an active section is announced in the main panel",
          any("only" in w.value and scope_word_in(w.value) for w in at.warning),
          str([w.value[:60] for w in at.warning]))
    check("a one-click 'Search all' escape exists",
          any("Search all" in b.label for b in at.button))

    # Clearing the scope must actually clear it.
    clear = [i for i, b in enumerate(at.button) if "Search all" in b.label]
    if clear:
        at.button[clear[0]].click().run()
        check("'Search all' clears the section filter",
              not any("Answering only from" in m.value for m in at.success),
              str([m.value[:40] for m in at.success]))
        # Put it back for the remaining assertions.
        labels = [sb.label for sb in at.selectbox]
        at.selectbox[labels.index("Organ system")].set_value(hn).run()
        labels = [sb.label for sb in at.selectbox]
        pathbox = at.selectbox[labels.index("Pathology")]
        tongue = next((o for o in pathbox.options if "tongue" in o.lower()), None)
        if tongue:
            pathbox.set_value(tongue).run()

    # Scoped Find: a topic from a different organ system must not come back.
    # The topic has to be genuinely disjoint. "prostate brachytherapy" is not:
    # brachytherapy is discussed for T1/T2 tongue cancers, so the oral-cavity
    # scope returns it at a low score, correctly. Wilms tumour shares nothing.
    at.radio[0].set_value("Find a topic").run()
    at.text_input[0].set_value("Wilms tumour").run()
    n = len(goto_buttons(at))
    check("a foreign topic finds nothing inside the tongue scope", n == 0,
          f"{n} places returned")
    check("the miss points the user at where it does appear",
          any("appear elsewhere" in m.value for m in at.info), "")


def test_ask_mode(AppTest) -> None:
    print("\n== Ask mode (one real LLM call) ==")
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    check("oncology reference page opens", goto_page(at, "Oncology Reference"),
          str(list(at._registered_pages.values())))
    if [title.value for title in at.title] != ["Oncology Reference"]:
        return
    at.chat_input[0].set_value("What is the lymphatic drainage of the oral tongue?").run()
    check("question answered without an exception", not at.exception)
    check("the answer carries a source card", len(at.expander) > 0,
          f"{len(at.expander)} expander(s)")

    btns = goto_buttons(at)
    check("source card offers 'Go to section'", len(btns) > 0)
    if not btns:
        return

    at.button[btns[0]].click().run()
    check("clicking renders the page", len(at.image) == 1, f"{len(at.image)} image(s)")
    # The regression itself: the page is useless if its container closes on it.
    expanded = [e.proto.expanded for e in at.expander]
    check("the source card stays OPEN so the page is visible", any(expanded),
          f"expanded flags: {expanded}")
    check("no exception after opening the page", not at.exception)


def test_home(AppTest) -> None:
    print("\n== Home ==")
    at = AppTest.from_file(ENTRY, default_timeout=300)
    at.run()
    check("home renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("home carries the platform title",
          any("Doctor Knowledge" in t.value for t in at.title),
          str([t.value for t in at.title]))
    headings = " ".join(m.value for m in at.markdown)
    for name in ("Knowledge Assistant", "Content Studio", "Research Platform"):
        check(f"home describes the {name}", name in headings)
    links = at.get("page_link")
    check("home links to all three tools", len(links) == 3, f"{len(links)} links")
    check("the research platform is marked as coming soon",
          any("Coming soon" in i.value for i in at.info))


def test_studio_brief(AppTest) -> None:
    print("\n== Content Studio (no model needed) ==")
    at = AppTest.from_file(STUDIO, default_timeout=300)
    at.run()
    check("studio renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("studio asks for a topic and a description",
          [t.label for t in at.text_input] == ["Topic"]
          and [t.label for t in at.text_area] == ["Describe the piece"])
    check("studio asks what kind of piece",
          any(sb.label == "Kind of piece" for sb in at.selectbox))
    check("a writer model is offered in the sidebar",
          len(at.sidebar.radio) == 1 and len(at.sidebar.radio[0].options) >= 1,
          str([r.options for r in at.sidebar.radio]))

    # An empty brief must be stopped before any model call is made.
    [b for b in at.button if "Next" in b.label][0].click().run()
    warnings = [w.value for w in at.warning]
    check("an empty brief is refused with a reason",
          any("topic" in w.lower() for w in warnings), str(warnings))
    check("an empty brief stays on step 1",
          any(t.label == "Topic" for t in at.text_input))


def test_studio_draft(AppTest) -> None:
    print("\n== Content Studio (real draft, a few minutes) ==")
    at = AppTest.from_file(STUDIO, default_timeout=900)
    at.run()
    at.text_input[0].set_value("Early warning signs of oral cancer")
    at.text_area[0].set_value(
        "A friendly blog post for my clinic website encouraging adults over 40, "
        "especially smokers, to get their mouth checked.")
    [b for b in at.button if "Next" in b.label][0].click().run()
    check("step 2 is reached without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("topic-specific questions are asked", len(at.text_input) >= 1,
          f"{len(at.text_input)} questions")

    short = [r for r in at.radio if any(o.startswith("Short") for o in r.options)]
    if short:
        short[0].set_value([o for o in short[0].options if o.startswith("Short")][0])
    [b for b in at.button if "Write the article" in b.label][0].click().run()
    check("a draft is produced without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    article = [m.value for m in at.markdown if m.value.startswith("# ")]
    check("the draft opens with a single H1 title", len(article) == 1)
    check("the draft is marked for physician review",
          any("Draft for your review" in w.value for w in at.warning))
    check("the draft can be downloaded", len(at.get("download_button")) == 1)
    if article:
        # A draft that stops mid-sentence is the specific failure the tidy-up
        # step exists to prevent.
        body = article[0].split("\n---\n")[0].rstrip()
        check("the draft ends on a complete sentence",
              body[-1:] in ".!?)*", repr(body[-40:]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip Ask mode, which needs a running model server")
    args = ap.parse_args()

    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit.testing is unavailable -- upgrade streamlit", file=sys.stderr)
        return 1

    check_landing_and_pages(AppTest)

    test_find_mode(AppTest)
    test_scope_selector(AppTest)
    test_studio_brief(AppTest)

    if args.no_llm:
        print("\n(skipping Ask mode and drafting: --no-llm)")
    else:
        from rag.llm import LLMClient

        ok, detail = LLMClient.from_config().health()
        if ok:
            test_ask_mode(AppTest)
            test_studio_draft(AppTest)
        else:
            print(f"\n(skipping Ask mode and drafting: model server unreachable -- {detail})")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all UI checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
