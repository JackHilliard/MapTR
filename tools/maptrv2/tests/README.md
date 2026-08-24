# dataset_viewer tests

Covers the browse tab's **view angle** and **per-tile tagging** (CLAUDE.md,
2026-08-24). Both files build their own throwaway dataset in `/tmp`; neither
touches real data.

## Python side — `test_viewer_views_tags.py`

Drives the real Flask app through its test client and **parses the emitted
HTML with `html.parser`**. Never curl a URL you assembled yourself: the
`&gt_frame=` entity trap in CLAUDE.md's "Transferable debugging lessons"
is invisible to curl because nothing HTML-parses the URL.

    python3 tools/maptrv2/tests/test_viewer_views_tags.py

Needs **matplotlib >= 3.3** — this file has always passed `labelcolor=` to
`ax.legend()`, which older versions reject on any legend-drawing path (this
is pre-existing and unrelated to the tests; it fails identically on the
top-down view). The host's system python has 3.1.2, so run it in the
container:

    docker run --rm -e PYTHONPATH=/MapTR \
      -v $(pwd)/tools:/MapTR/tools -v /tmp/vt:/tmpx \
      -w /MapTR jhd0ck3r/maptrv2:latest \
      python3 /MapTR/tools/maptrv2/tests/test_viewer_views_tags.py

The container runs as **root**, which bypasses directory permission bits, so
the read-only-dataset-dir check skips itself there and prints `SKIP`. Run the
file as a normal user on the host to exercise that one — between the two
runs everything is covered.

## Browser side — `test_tag_js.js`

Tagging posts with `fetch` and rewrites the DOM, so half the feature only
ever runs in a browser. This loads the **real page HTML** into jsdom with
`fetch` stubbed, and checks the POST payloads, the checkbox reverting on a
failed write, a new tag propagating to every tile, `Enter` not submitting the
surrounding form, and event delegation reaching JS-created checkboxes.

Regenerate the page first — otherwise it tests a stale copy of the script.
`dump_page.py` must run from its place in this directory (it reads the python
test beside it); `PAGE_OUT` says where to drop the HTML:

    mkdir -p /tmp/vt && cp tools/maptrv2/tests/test_tag_js.js /tmp/vt/
    docker run --rm -e PYTHONPATH=/MapTR -e PAGE_OUT=/tmpx/page.html \
      -v $(pwd)/tools:/MapTR/tools -v /tmp/vt:/tmpx \
      -w /MapTR jhd0ck3r/maptrv2:latest \
      python3 /MapTR/tools/maptrv2/tests/dump_page.py
    docker run --rm -v /tmp/vt:/w -w /w node:20-alpine \
      sh -c "npm install --silent jsdom && node test_tag_js.js"

This pass is worth keeping: it is what caught an unhandled promise rejection
on the failed-write path and a dependency on `CSS.escape`, neither of which
any server-side test can see.
