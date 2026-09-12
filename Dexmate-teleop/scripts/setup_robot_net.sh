#!/usr/bin/env bash
# 网口拔插 / 重启后,把真机相关的三个网段地址一次补齐(重复执行无害)。
#
#     sudo bash scripts/setup_robot_net.sh
#
# 为什么需要:enp3s0 上只有 NM 配置(dexmate-torso,192.168.50.10)的地址会自动恢复,下面这些
# 第二地址每次断开都会丢,而丢了之后的表现是「mDNS 能发现却连不上」/
# 「两只手都 ping 不通」—— 2026-08-10 真机现场一天栽了两次。
set -u
IF=enp3s0
ADDRS=(
    "192.168.50.100/24"    # Dexmate 机器人(zenoh 在 192.168.50.20:7447)
    "192.168.10.240/24"    # Sharpa 手(左 .10 / 右 .20)
    "192.168.5.100/24"     # 其它设备(SOP 网络表)
)
for a in "${ADDRS[@]}"; do
    if ip -4 addr show "$IF" | grep -q "${a%/*}/"; then
        echo "已存在  $a"
    elif ip addr add "$a" dev "$IF" 2>/dev/null; then
        echo "已添加  $a"
    else
        echo "添加失败 $a(需要 sudo?)" >&2
    fi
done
echo
echo "当前 $IF 的地址:"
ip -4 addr show "$IF" | grep inet
echo
echo "连通性:"
for ip in 192.168.50.20 192.168.10.10 192.168.10.20; do
    if ping -c1 -W1 "$ip" >/dev/null 2>&1; then
        echo "  $ip 通"
    else
        echo "  $ip 不通(设备未上电,或网线/交换机问题)"
    fi
done
