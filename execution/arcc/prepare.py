"""
Prepare a local upload folder for a Newton experiment.

Each recipe lists the files its experiment needs. By default this command prints
what the upload includes, which inputs to supply separately and how to check
them on Newton. Use --destination to create the folder or --update to compare
an existing folder with the recipe. Add --apply to write that update locally.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from textwrap import indent

from encoder_swaps import package_spec as encoder_swaps
from fusion_studies import package_spec as fusion_studies
from grouped_stability import package_spec as grouped_stability
from partner_tasks import package_spec as partner_tasks
from input_guidance import describe_inputs
from package_update import update_package


RECIPES = {
    "encoder_swaps": encoder_swaps,
    "grouped_stability": grouped_stability,
    "fusion_studies": fusion_studies,
    "partner_tasks": partner_tasks,
}


# -------- Choose an experiment and a local folder action --------


def read_command_line_options(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    recipes = parser.add_subparsers(dest="recipe", required=True)
    for name in RECIPES:
        recipe = recipes.add_parser(name)
        recipe.add_argument(
            "--source-root", type=Path, default=Path(__file__).resolve().parents[2],
            help="repository to read (defaults to this checkout)",
        )
        folder_action = recipe.add_mutually_exclusive_group()
        folder_action.add_argument(
            "--destination", type=Path, help="create a folder beside the repository",
        )
        folder_action.add_argument(
            "--update", type=Path, help="show what would change in a prepared folder",
        )
        recipe.add_argument("--apply", action="store_true", help="write the update to the local folder")
        recipe.add_argument("--json", action="store_true", help="print the report and full file list as JSON")

        if name in {"encoder_swaps", "grouped_stability"}:
            recipe.add_argument(
                "--cache-root", type=Path,
                help="protect an embedding cache path on creation (updates keep the original path)",
            )
        if name == "encoder_swaps":
            recipe.add_argument("--experiment", choices=("full", "ecoli_ernie"), default="full")
            recipe.add_argument(
                "--selection-workspace", type=Path,
                help="read historical selection records from this workspace",
            )

    options = parser.parse_args(argv)
    if options.apply and options.update is None:
        parser.error("--apply requires --update FOLDER")
    return options


# -------- Preview, create or update the upload folder --------


def prepare(options):
    experiment = getattr(options, "experiment", "full")
    if options.recipe == "encoder_swaps":
        plan = encoder_swaps.build_plan(
            options.source_root, options.selection_workspace, options.cache_root,
            experiment=experiment,
        )
    elif options.recipe == "grouped_stability":
        plan = grouped_stability.build_plan(options.source_root, options.cache_root)
    else:
        plan = RECIPES[options.recipe].inventory(options.source_root)

    # Every mode uses the same file list and instructions for checking external inputs.
    report = {
        "recipe": options.recipe,
        "mode": "inventory preview",
        "file_count": len(plan["files"]),
        "total_bytes": plan["total_bytes"],
        **describe_inputs(options.recipe, experiment),
    }
    if options.recipe == "encoder_swaps":
        report["experiment"] = experiment
    if options.json:
        report["files"] = plan["files"]

    if options.destination is None and options.update is None:
        return report

    if options.destination is not None:
        folder = options.destination.expanduser()
        RECIPES[options.recipe].write_package(plan, folder)
        report["mode"] = "created locally"
    else:
        folder = options.update.expanduser()
        source_roots = {"maintained_repository": plan["source_root"]}
        if options.recipe == "encoder_swaps":
            source_roots["historical_selection_workspace"] = plan["historical_selection"]["workspace"]
        changes = update_package(options.recipe, plan, folder, source_roots, apply=options.apply)
        report["mode"] = "local update" if options.apply else "update preview"
        report["delta"] = changes

    # This fingerprint lets the reader compare the local file record with its Newton copy.
    snapshot_bytes = (folder / "source-snapshot.json").read_bytes()
    report["destination"] = str(folder.resolve())
    report["snapshot_sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    if "delta" in report and report["delta"]["applied"]:
        report["previous_snapshot_sha256"] = json.loads(snapshot_bytes)["previous_snapshot_sha256"]
    return report


# -------- Show which files travel and what to check on Newton --------


def display(report):
    recipe = report["recipe"]
    if "experiment" in report:
        recipe += f" ({report['experiment']})"
    print(
        f"{recipe} | {report['mode']} | {report['file_count']} files, "
        f"{report['total_bytes'] / 1_000_000:.1f} MB"
    )
    if "destination" in report:
        print(report["destination"])
    if "delta" in report:
        delta = report["delta"]
        for label in ("added", "changed", "removed"):
            print(f"{label.capitalize()} ({len(delta[label])})")
            for path in delta[label]:
                print(f"  {path}")
        print(f"Unchanged: {delta['unchanged']}")
    if "previous_snapshot_sha256" in report:
        print("Previous snapshot SHA-256:", report["previous_snapshot_sha256"])
    if "snapshot_sha256" in report:
        print("Local snapshot SHA-256:", report["snapshot_sha256"])
    sections = (
        ("In the upload folder", "packaged"),
        ("Newton needs", "newton_needs"),
        ("Verify on Newton", "verify"),
    )
    for label, key in sections:
        print(label)
        for entry in report[key]:
            print(indent(entry, "  "))


def main(argv=None):
    options = read_command_line_options(argv)
    try:
        report = prepare(options)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Preparation stopped: {error}") from error
    if options.json:
        print(json.dumps(report, indent=2))
    else:
        display(report)


if __name__ == "__main__":
    main()
