"""给 arctic15 各 take 写 frame_plan.json。

fp_register_frame = 交互开始帧 + 10(同事实测更优, 见 _common/frame_plan.py)。
onset 取自已有的 v17A `interaction_episodes.json` —— 那是在 **30fps** 视频上跑的,
本次视频是 15fps(stride=2), 所以 onset 要 //2 换算。
没有 v17A 产物的 take 留空(null) -> fp_pose 回退到人工标注帧, 并在此打印出来。

sam3d_frame 一律留 null = 用人工标注的那一帧(用户就是按"能看见短轴"的原则选的)。
"""
import json, sys
from pathlib import Path
RR = Path.home()/"Reconstruct_and_Retarget"
sys.path.insert(0, str(RR/"ego_pipeline/Reconstruction/recon_pipeline"))
from _common.frame_plan import write_frame_plan

V17A = RR/"experimental/hoi_detr_v17a/data/interim/arctic"      # 30fps 那批的产物
I15 = RR/"Output/ReconstructOutput/interim/arctic15"
FP_OFFSET = 10

for d in sorted(I15.glob("*/sam2_object/label_prompt.json")):
    vid = d.parent.parent.name
    prompt = json.loads(d.read_text())
    ep = V17A/vid/"instance_pipeline_v17a"/"interaction_episodes.json"
    onset15 = None
    if ep.is_file():
        try:
            eps = json.loads(ep.read_text())
            eps = eps if isinstance(eps, list) else eps.get("episodes", [])
            if eps:
                onset30 = min(int(e["start_frame"]) for e in eps)
                onset15 = onset30 // 2                    # 30fps -> 15fps
        except Exception as e:
            print(f"  {vid}: 读 onset 失败 {e}")
    objs = {}
    for o in prompt["objects"]:
        oid = o["object_id"]
        if onset15 is None:
            objs[oid] = {"fp_register_frame": None,
                         "fp_source": "no v17A onset (人工标注路径, 无实例发现产物)",
                         "sam3d_frame": None, "sam3d_source": "manual label frame"}
        else:
            objs[oid] = {"fp_register_frame": onset15 + FP_OFFSET,
                         "fp_source": f"interaction_onset({onset15}@15fps, 由30fps {onset15*2}换算)+{FP_OFFSET}",
                         "sam3d_frame": None, "sam3d_source": "manual label frame"}
    write_frame_plan(d.parent, objs)
    fp = objs[list(objs)[0]]["fp_register_frame"]
    print(f"  {vid:34s} 标注帧={prompt['objects'][0]['frame_idx']:4d}  fp_register_frame="
          f"{fp if fp is not None else '(回退标注帧)'}")
