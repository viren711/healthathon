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

    python scripts/ui_test.py            # both modes (Ask needs the LLM)
    python scripts/ui_test.py --no-llm   # Find mode only, no model required
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

APP = str(ROOT / "app" / "streamlit_app.py")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'  ok  ' if condition else ' FAIL '}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def scope_word_in(text: str) -> bool:
    return "Searching" in text


def goto_buttons(at):
    return [i for i, b in enumerate(at.button) if "Go to section" in b.label]


def test_find_mode(AppTest) -> None:
    print("\n== Find mode ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    check("app starts without an exception", not at.exception)

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
    at2.radio[0].set_value("Find a topic").run()
    at2.text_input[0].set_value("What is the capital of France?").run()
    check("an out-of-corpus topic returns no places", len(goto_buttons(at2)) == 0,
          f"{len(goto_buttons(at2))} places")


def test_scope_selector(AppTest) -> None:
    print("\n== Section selector ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
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

    test_find_mode(AppTest)
    test_scope_selector(AppTest)

    if args.no_llm:
        print("\n(skipping Ask mode: --no-llm)")
    else:
        from rag.llm import LLMClient

        ok, detail = LLMClient.from_config().health()
        if ok:
            test_ask_mode(AppTest)
        else:
            print(f"\n(skipping Ask mode: model server unreachable -- {detail})")

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
