"""Generate the code reference pages automatically."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src_root = Path(__file__).parent.parent / "src"
package_root = src_root / "neurospatial"

for path in sorted(package_root.rglob("*.py")):
    module_path = path.relative_to(src_root).with_suffix("")
    doc_path = path.relative_to(src_root).with_suffix(".md")
    full_doc_path = Path("api") / doc_path

    parts = tuple(module_path.parts)

    # Skip __pycache__ and test files
    if "__pycache__" in parts or "test_" in path.name:
        continue

    # Skip __init__ files for navigation (but still generate docs)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Generate the markdown file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src_root.parent))

# Write the navigation file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
