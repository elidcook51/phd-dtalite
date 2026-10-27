"""bigloop1_fix1.py -- bigloop-1 empty-path fix (Layer 1 + secondary fixes).

Applies the finalized fix for the agent-9864 []-path bug (run_106492):

  * msa.py predict + actual phases: route paths through
    normalize_route_path() instead of the raw `is not None` check, so an
    empty choiceset cell ([]) can no longer overwrite a good path and kill
    DTALite (DTALite stops the whole program on one infeasible path).
    Invalid routes fall back to the prior iteration's path with a warning;
    a loud RuntimeError is raised only when no fallback exists.
    (The MSA iteration loop already uses normalize_route_path correctly;
    only predict + actual had the raw check. The UOinfor step is a copy,
    not a path write, so it needs no change here.)
  * msa_ue.py: same validation for its choiceset writes (predict + actual)
    and its ue_route writes (defense; UE's own bigloop-1 failure is still
    under separate investigation).
  * parallel_msa.py / parallel_msa_ue.py: stage_run_dir strips
    {n}actual_/{n}predict_/{n}UOinfor_ files from staged Runs (staging
    isolation), and run_parallel_msa[_ue] writes failures to
    sweep_failures.txt in the sweep sandbox (so a silent skip becomes
    visible instead of living only on the console).

Usage (from the repo root):
    python bigloop1_fix1.py

The script is idempotent (a second run reports ALREADY-FIXED and changes
nothing) and all-or-nothing per file: if any expected pattern is missing
from a file, that file is left untouched and the miss is reported.
Originals are backed up to <file>.pre_fix1 (never overwritten if present,
and never touching the existing .bak files from the debug patch, which
stay in place until the rerun confirms).
"""

import os
import re
import shutil
import sys


def _join(lines, nl):
    return nl.join(lines)


def _msa_block(indent, nl, var, idx, agent, label, phase):
    # Replacement for an msa.py predict/actual block. var is "_p" style
    # assignment target info is passed via idx/agent strings.
    lines = [
        "%s_p = choiceset[" % indent,
        "%s    int(%s), int(%s)" % (indent, idx[0], idx[1]),
        "%s]" % indent,
        "%s_norm, _norm_nodes = normalize_route_path(_p)" % indent,
        "%sif _norm is not None:" % indent,
        "%s    inputagent[%s][11] = _norm" % (indent, agent),
        "%s    inputagent[%s][4] = _norm_nodes[0]" % (indent, agent),
        "%s    inputagent[%s][5] = _norm_nodes[-1]" % (indent, agent),
        "%selse:" % indent,
        '%s    print(f"  [%s] bigloop={bigloop} agent={inputagent[%s][0]}: "'
        % (indent, label, agent),
        '%s          f"predicted route invalid ({_p!r}), keeping prior path", flush=True)'
        % indent,
        "%s    if normalize_route_path(inputagent[%s][11])[0] is None:" % (indent, agent),
        "%s        raise RuntimeError(" % indent,
        '%s            f"bigloop={bigloop} phase=%s agent={inputagent[%s][0]}: "'
        % (indent, phase, agent),
        '%s            f"predicted route invalid and no prior path to fall back to")'
        % indent,
    ]
    return _join(lines, nl)


def _msa_ue_choiceset_block(indent, nl, idx, agent, label, phase):
    # Replacement for an msa_ue.py choiceset write (no OD overwrite in the
    # original branches, so none is added here).
    lines = [
        "%s_p_ue = choiceset[" % indent,
        "%s    int(%s), int(%s)" % (indent, idx[0], idx[1]),
        "%s]" % indent,
        "%s_norm, _norm_nodes = normalize_route_path(_p_ue)" % indent,
        "%sif _norm is not None:" % indent,
        "%s    inputagent[%s][11] = _norm" % (indent, agent),
        "%selse:" % indent,
        '%s    print(f"  [%s] bigloop={bigloop} agent={inputagent[%s][0]}: "'
        % (indent, label, agent),
        '%s          f"predicted route invalid ({_p_ue!r}), keeping prior path", flush=True)'
        % indent,
        "%s    if normalize_route_path(inputagent[%s][11])[0] is None:" % (indent, agent),
        "%s        raise RuntimeError(" % indent,
        '%s            f"bigloop={bigloop} phase=%s agent={inputagent[%s][0]}: "'
        % (indent, phase, agent),
        '%s            f"predicted route invalid and no prior path to fall back to")'
        % indent,
    ]
    return _join(lines, nl)


def _msa_ue_ue_route_block(indent, nl):
    lines = [
        "%s_norm, _norm_nodes = normalize_route_path(ue_route[jj])" % indent,
        "%sif _norm is not None:" % indent,
        "%s    inputagent[j][11] = _norm" % indent,
        "%s    inputagent[j][4] = _norm_nodes[0]" % indent,
        "%s    inputagent[j][5] = _norm_nodes[-1]" % indent,
        "%selse:" % indent,
        '%s    print(f"  [ue-validate] bigloop={bigloop} agent={inputagent[j][0]}: "'
        % indent,
        '%s          f"UE route invalid ({ue_route[jj]!r}), keeping prior path", flush=True)'
        % indent,
        "%s    if normalize_route_path(inputagent[j][11])[0] is None:" % indent,
        "%s        raise RuntimeError(" % indent,
        '%s            f"bigloop={bigloop} phase=ue-route agent={inputagent[j][0]}: "'
        % indent,
        '%s            f"UE route invalid and no prior path to fall back to")'
        % indent,
    ]
    return _join(lines, nl)


def _stage_block(indent, nl):
    lines = [
        "%sshutil.copytree(src, dst)" % indent,
        "%s# bigloop1-fix: staged inputs must never carry another run's outputs." % indent,
        "%s# The sweep sandbox accumulates {n}actual_/{n}predict_/{n}UOinfor_ files" % indent,
        "%s# via collect_results, so a late-staging Run would otherwise inherit them." % indent,
        "%sfor _stale in os.listdir(dst):" % indent,
        "%s    if _stale and _stale[0].isdigit() and (" % indent,
        '%s        "actual_" in _stale or "predict_" in _stale or "UOinfor_" in _stale' % indent,
        "%s    ):" % indent,
        "%s        _stale_path = os.path.join(dst, _stale)" % indent,
        "%s        if os.path.isfile(_stale_path):" % indent,
        "%s            os.remove(_stale_path)" % indent,
        "%sreturn run_dir" % indent,
    ]
    return _join(lines, nl)


def _failures_block(indent, nl):
    cont = " " * (len(indent) + 20)
    lines = [
        indent + "if all_failed:",
        indent + "    # bigloop1-fix: failures must be visible in a file, not just console.",
        indent + "    try:",
        indent + '        with open(os.path.join(repo_root, "sweep_failures.txt"), "w") as _ff:',
        indent + "            for _bl in sorted(all_failed):",
        indent + '                print("bigloop %s: %s" % (_bl, all_failed[_bl]), file=_ff)',
        indent + "    except Exception:",
        indent + "        pass",
        indent + "return SweepResults([all_ok[bl] for bl in sorted(all_ok)],",
        cont + "failures=all_failed)",
    ]
    return nl.join(lines)


# Each site: (name, regex pattern, builder, expected match count, marker)
# The builder receives (indent, nl) and returns the replacement text.
def _sites_for(fname):
    if fname == "msa.py":
        pat_predict = re.compile(
            r"(?m)^([ \t]*)_p\s*=\s*choiceset\[\s*int\(pre_choice\[j\]\),\s*"
            r"int\(pre_rposition\[j\]\)\s*\]\s*"
            r"if\s+_p\s+is\s+not\s+None:\s*"
            r"inputagent\[j\]\[11\]\s*=\s*_p\s*"
            r"_nodes\s*=\s*\[x\.strip\(\)\s*for\s+x\s+in\s+str\(_p\)\.split\(';'\)\s*"
            r"if\s+x\.strip\(\)\]\s*"
            r"if\s+len\(_nodes\)\s*>=\s*2:\s*"
            r"inputagent\[j\]\[4\]\s*=\s*_nodes\[0\]\s*"
            r"inputagent\[j\]\[5\]\s*=\s*_nodes\[-1\]"
        )
        pat_actual = re.compile(
            r"(?m)^([ \t]*)_p\s*=\s*choiceset\[\s*int\(com_choice\[_k\]\),\s*"
            r"int\(com_rposition\[_k\]\)\s*\]\s*"
            r"if\s+_p\s+is\s+not\s+None:\s*"
            r"inputagent\[_row\]\[11\]\s*=\s*_p\s*"
            r"_nodes\s*=\s*\[x\.strip\(\)\s*for\s+x\s+in\s+str\(_p\)\.split\(';'\)\s*"
            r"if\s+x\.strip\(\)\]\s*"
            r"if\s+len\(_nodes\)\s*>=\s*2:\s*"
            r"inputagent\[_row\]\[4\]\s*=\s*_nodes\[0\]\s*"
            r"inputagent\[_row\]\[5\]\s*=\s*_nodes\[-1\]"
        )
        return [
            ("predict phase", pat_predict,
             lambda ind, nl: _msa_block(ind, nl, None,
                                        ("pre_choice[j]", "pre_rposition[j]"),
                                        "j", "predict-validate", "predict"),
             1, "[predict-validate]", False),
            ("actual phase", pat_actual,
             lambda ind, nl: _msa_block(ind, nl, None,
                                        ("com_choice[_k]", "com_rposition[_k]"),
                                        "_row", "actual-validate", "actual"),
             1, "[actual-validate]", False),
        ]
    if fname == "msa_ue.py":
        pat_ue_route = re.compile(
            r"(?m)^([ \t]*)inputagent\[j\]\[11\]\s*=\s*ue_route\[jj\]\s*"
            r"_nd\s*=\s*\[x\.strip\(\)\s*for\s+x\s+in\s+str\(ue_route\[jj\]\)\.split\(';'\)\s*"
            r"if\s+x\.strip\(\)\]\s*"
            r"if\s+len\(_nd\)\s*>=\s*2:\s*"
            r"inputagent\[j\]\[4\]\s*=\s*_nd\[0\]\s*"
            r"inputagent\[j\]\[5\]\s*=\s*_nd\[-1\]"
        )
        pat_pred = re.compile(
            r"(?m)^([ \t]*)inputagent\[j\]\[11\]\s*=\s*choiceset\[\s*"
            r"int\(pre_choice\[j\]\),\s*int\(pre_rposition\[j\]\)\s*\]"
        )
        pat_act = re.compile(
            r"(?m)^([ \t]*)inputagent\[_row\]\[11\]\s*=\s*choiceset\[\s*"
            r"int\(com_choice\[_k\]\),\s*int\(com_rposition\[_k\]\)\s*\]"
        )
        return [
            ("ue_route writes (x2)", pat_ue_route,
             lambda ind, nl: _msa_ue_ue_route_block(ind, nl),
             2, "[ue-validate]", False),
            ("predict choiceset write", pat_pred,
             lambda ind, nl: _msa_ue_choiceset_block(
                 ind, nl, ("pre_choice[j]", "pre_rposition[j]"),
                 "j", "predict-validate", "predict"),
             1, "[predict-validate]", False),
            ("actual choiceset write", pat_act,
             lambda ind, nl: _msa_ue_choiceset_block(
                 ind, nl, ("com_choice[_k]", "com_rposition[_k]"),
                 "_row", "actual-validate", "actual"),
             1, "[actual-validate]", False),
        ]
    if fname in ("parallel_msa.py", "parallel_msa_ue.py"):
        pat_stage = re.compile(
            r"(?m)^([ \t]*)shutil\.copytree\(src, dst\)\s*return run_dir"
        )
        pat_fail = re.compile(
            r"(?m)^([ \t]*)return SweepResults\(\[all_ok\[bl\] for bl in sorted\(all_ok\)\],\s*"
            r"failures=all_failed\)"
        )
        return [
            ("stage_run_dir cleanup", pat_stage,
             lambda ind, nl: _stage_block(ind, nl),
             1, "bigloop1-fix: staged inputs", False),
            ("sweep_failures.txt logging", pat_fail,
             lambda ind, nl: _failures_block(ind, nl),
             1, 'with open(os.path.join(repo_root, "sweep_failures.txt")', True),
        ]
    return []


def _process_file(fname):
    if not os.path.isfile(fname):
        return ("SKIP", "%s: not found in current directory" % fname)
    try:
        with open(fname, "r", encoding="utf-8", newline="") as fh:
            text = fh.read()
    except UnicodeDecodeError as exc:
        return ("SKIP", "%s: cannot decode as UTF-8 (%s)" % (fname, exc))
    nl = "\r\n" if "\r\n" in text else "\n"
    if fname in ("msa.py", "msa_ue.py") and "def normalize_route_path" not in text:
        return ("SKIP", "%s: normalize_route_path not found; refusing to edit" % fname)
    sites = _sites_for(fname)
    new_text = text
    report = []
    for name, pattern, builder, expected, marker, insert_only in sites:
        matches = list(pattern.finditer(new_text))
        if marker in new_text:
            if insert_only or not matches:
                report.append("  ALREADY-FIXED: %s" % name)
                continue
            return ("SKIP", "%s: %s has both fix marker and old pattern; "
                    "inspect manually, file left untouched" % (fname, name))
        if len(matches) != expected:
            return ("SKIP", "%s: %s: expected %d match(es), found %d; "
                    "file left untouched" % (fname, name, expected, len(matches)))

        def _repl(m, _builder=builder, _nl=nl):
            return _builder(m.group(1), _nl)

        new_text = pattern.sub(_repl, new_text)
        report.append("  FIXED: %s (%d site(s))" % (name, expected))
    if new_text == text:
        return ("OK", "%s: already fixed, no changes\n%s" % (fname, "\n".join(report)))
    backup = fname + ".pre_fix1"
    if not os.path.exists(backup):
        shutil.copyfile(fname, backup)
        report.append("  backup: %s (kept; existing .bak files untouched)" % backup)
    else:
        report.append("  backup: %s already exists, kept as-is" % backup)
    with open(fname, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)
    try:
        compile(new_text, fname, "exec")
    except SyntaxError as exc:
        shutil.copyfile(backup, fname)
        return ("FAIL", "%s: syntax check failed after edit (%s); "
                "restored from %s" % (fname, exc, backup))
    return ("OK", "%s:\n%s" % (fname, "\n".join(report)))


def main():
    files = ["msa.py", "msa_ue.py", "parallel_msa.py", "parallel_msa_ue.py"]
    failed = False
    for fname in files:
        status, message = _process_file(fname)
        print(message)
        if status in ("SKIP", "FAIL"):
            failed = True
    if failed:
        print("\nOne or more files were skipped or failed; files left "
              "untouched where reported. Fix the reported issue and rerun.")
        return 1
    print("\nAll fixes applied (or already present). Next: rerun Test 5 only:")
    print("  python wholerunner.py --tests 5 --seed 42 --engine dtalite --keep-temp")
    print("Expect bigloop 1 back at 54,045 rows and the 10% row in Tables 5/6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
