#!/usr/bin/env python3
"""查「用了但没定义的名字」—— 这个仓库栽过的那一类 NameError。

    cd <仓库根目录> && .venv/bin/python sim/dev_undefined_names.py 文件...
    .venv/bin/python sim/dev_undefined_names.py            # 不给参数就查全仓

为什么要自己写:本机三个 venv 都没有 ruff/pyflakes,而这个仓库反复栽的正是
「某个分支里的名字拼错了,那段平时不执行,退出码还是 0」。语法检查抓不到它。

做法:用标准库的 symtable 拿到每个作用域的**自由变量与全局变量**,减掉模块
自己定义的、import 进来的、以及内置名。剩下的就是可疑的未定义名。

这是个近似:动态注入(exec、globals() 赋值、from x import *)会产生假阳性,
所以结果按文件分组打印,由人过一眼,而不是当成硬门禁。**它的价值在于把
「从没执行过的那条分支里有个错名字」这件事提前暴露出来。**
"""
from __future__ import annotations

import builtins
import pathlib
import symtable
import sys

# 模块层面永远存在的 dunder,不是未定义
DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
           "__loader__", "__builtins__", "__debug__", "__path__"}
SKIP_DIRS = {".venv", ".venv-isaac", ".venv-pico", ".git", "__pycache__",
             "assets", "logs", "data", "third_party", "thirdparty",
             # 第三方检出(git 已忽略,不属于本仓代码)。不排除的话它的
             # build/ 产物有 9 处上游自身的可疑名,会把「0 命中才有意义」
             # 的判据搅浑 —— 2026-08-10 深夜实测。
             "curobo"}


def module_names(st: symtable.SymbolTable) -> set[str]:
    """模块层面「有定义」的名字:赋值、import、def、class。"""
    out = set()
    for s in st.get_symbols():
        if s.is_assigned() or s.is_imported() or s.is_namespace():
            out.add(s.get_name())
    return out


def walk(st: symtable.SymbolTable, known: set[str], hits: list):
    for s in st.get_symbols():
        # 只看「用到了、但本作用域没赋值、也不是参数」的全局引用
        if s.is_global() and s.is_referenced() and not s.is_assigned():
            n = s.get_name()
            if n not in known and n not in DUNDERS \
                    and not hasattr(builtins, n):
                hits.append((st.get_name(), st.get_lineno(), n))
    for child in st.get_children():
        walk(child, known | {p.get_name() for p in st.get_symbols()
                             if p.is_assigned() or p.is_imported()}, hits)


def check(path: pathlib.Path) -> list:
    src = path.read_text(errors="replace")
    st = symtable.symtable(src, str(path), "exec")
    hits: list = []
    walk(st, module_names(st), hits)
    return hits


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1:
        files = [pathlib.Path(a) for a in sys.argv[1:]]
    else:
        files = [p for p in root.rglob("*.py")
                 if not (SKIP_DIRS & set(p.relative_to(root).parts))]
    bad = 0
    for f in sorted(files):
        try:
            hits = check(f)
        except SyntaxError as e:
            print(f"[语法错] {f}: {e}")
            bad += 1
            continue
        if hits:
            bad += 1
            print(f"\n{f.relative_to(root) if f.is_relative_to(root) else f}")
            for scope, line, name in hits:
                print(f"   第 {line} 行附近 · 作用域 {scope!r} · 可疑名字 {name!r}")
    print(f"\n查了 {len(files)} 个文件,{bad} 个有可疑项。"
          + ("" if bad else "  没有查到未定义名。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
