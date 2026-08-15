#!/usr/bin/env python3
"""depth_scale 合理性硬闸。超出区间 -> 退出码 1(调用方终止本条)。

★ 2026-08-14 教训: 这个检查原来是内联在 reconstruct_egodex.sh 的 `python -c "..."` 里,
  错误信息用了 f-string 且把 **bash 变量塞进了 `{}`**:

      f'排查: 看 {"$RECON_INTERIM_ROOT/$DATASET/$VID"}/egodex_source.json'

  bash 先展开成真实路径, f-string 的 `{}` 里就成了 `{/home/yanghong/...}` —— 那不是
  合法表达式, Python 直接**语法错误**、退出码非零, 于是 `if !` 把**每一条 take** 都
  判成"depth_scale 异常"跳过。试跑 5 条全被拦, 而 5 条实测值 0.69~1.04 全在区间内。

  真正的教训不是"f-string 用错了", 而是: **判据不该内联在 shell 字符串里**。内联的
  Python 一旦有语法错误, 表现和"判据不通过"完全一样 —— 一个坏掉的闸看起来就像一个
  严格的闸, 不会有任何人发现。独立文件至少能被 py_compile / 单独运行验证。

区间 [0.01, 10]: 比实测正常值(EgoDex pour 实测 0.33~1.04)宽两个数量级, 只毙掉
"ViPE 深度整个塌了"这种量级的错(clip 21/22 曾算出 208 / 156)。
"""
from __future__ import annotations

import sys

LO, HI = 0.01, 10.0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法: check_depth_scale.py <depth_scale> [take_interim_dir]", file=sys.stderr)
        return 2
    try:
        d = float(argv[1])
    except ValueError:
        print(f"[egodex] X depth_scale 不是数字: {argv[1]!r} —— 本条终止。", file=sys.stderr)
        return 1
    if not (LO <= d <= HI):
        where = argv[2] if len(argv) > 2 else "<take interim 目录>"
        print(f"[egodex] X depth_scale={d:g} 超出合理区间 [{LO}, {HI}] —— 本条终止。\n"
              f"    EgoDex 实测正常量级 0.33~1.04; 出现 100+ 说明 ViPE 深度塌了。\n"
              f"    排查: {where}/egodex_source.json 与 vipe 输出。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
