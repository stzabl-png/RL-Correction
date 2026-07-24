# Summarize the REGRIP v2 sweep: parse every results/regripv2_sweep/<run>/ledger_probe.log (+ contact
# probe) into one ranked markdown table with PASS/FAIL against the pre-registered bar
# (survival>=0.2 AND min_share median>=0.05 AND handovers mean>=2). CPU only.
# Run: python rl_rebuild/scripts/regripv2_summarize.py [--root results/regripv2_sweep]
import argparse, glob, os, re

p = argparse.ArgumentParser()
p.add_argument("--root", default="results/regripv2_sweep")
args = p.parse_args()


def grab(txt, pat, idx=1, cast=float):
    m = re.search(pat, txt)
    return cast(m.group(idx)) if m else None


rows = []
for d in sorted(glob.glob(os.path.join(args.root, "regripv2*_*"))):
    run = os.path.basename(d)
    lp = os.path.join(d, "ledger_probe.log")
    row = dict(run=run, eps=None, surv=None, minshare=None, ho=None, distinct=None,
               switch=None, fail=None, resid=None, csw=None)
    if os.path.isfile(lp):
        t = open(lp).read()
        row["eps"] = grab(t, r"episodes scored[^:]*: (\d+)", cast=int)
        row["surv"] = grab(t, r"PALM-CLEAR SURVIVAL[^:]*: ([0-9.]+)")
        row["minshare"] = grab(t, r"MIN per-finger cumulative load share: median=([0-9.]+)")
        row["ho"] = grab(t, r"verified handovers/ep: mean=([0-9.]+)")
        row["distinct"] = grab(t, r"distinct fingers w/ >=1: mean=([0-9.]+)")
        row["switch"] = grab(t, r"switch-rate[^:]*: mean=([0-9.]+)")
        row["fail"] = grab(t, r"FAIL-terminated: ([0-9.]+)")
        row["resid"] = grab(t, r"wrench residual[^:]*: median=([0-9.]+)")
    cp = os.path.join(d, "contact_switch_probe.log")
    if os.path.isfile(cp):
        row["csw"] = grab(open(cp).read(), r"CONTACT-SWITCH RATE[^:]*: ([0-9.]+)")
    rows.append(row)


def f(v, spec="{:.3f}"):
    return spec.format(v) if v is not None else "—"


def passes(r):
    ok = [r["surv"] is not None and r["surv"] >= 0.2,
          r["minshare"] is not None and r["minshare"] >= 0.05,
          r["ho"] is not None and r["ho"] >= 2.0]
    return sum(ok), ok


rows.sort(key=lambda r: (-(r["surv"] or -1), -(r["minshare"] or -1)))
out = ["# REGRIP v2 sweep — summary (auto-generated)",
       "",
       "Pre-registered PASS bar: palm-clear survival >= 0.2 AND min load-share median >= 0.05 AND",
       "verified handovers/ep >= 2 (eval under SHARPA_EVAL_CLEAN, palm-up, -9.81).",
       "",
       "| run | eps | palm-clear surv | min share (med) | handovers/ep | distinct | switch | fail-term | resid | bar |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    npass, ok = passes(r)
    bar = "".join("✓" if b else "✗" for b in ok)
    out.append(f"| {r['run']} | {r['eps'] if r['eps'] is not None else '—'} | {f(r['surv'])} | "
               f"{f(r['minshare'], '{:.4f}')} | {f(r['ho'], '{:.2f}')} | {f(r['distinct'], '{:.2f}')} | "
               f"{f(r['switch'], '{:.4f}')} | {f(r['fail'])} | {f(r['resid'])} | {bar} |")
out += ["", "Columns from load_ledger_probe.log per run; switch also cross-checkable against",
        "contact_switch_probe.log. Sorted by palm-clear survival, then min share."]
path = os.path.join(args.root, "SUMMARY.md")
open(path, "w").write("\n".join(out) + "\n")
print("\n".join(out))
print(f"\nwritten -> {path}")
