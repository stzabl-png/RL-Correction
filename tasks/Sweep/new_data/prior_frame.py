"""Task3 canonical-to-input conversion, including the delivered COM origin."""
from pathlib import Path
import json
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

class CandidateUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(module.replace("numpy._core", "numpy.core"), name)

def load_candidate(path):
    with open(path, "rb") as stream:
        version = np.lib.format.read_magic(stream)
        np.lib.format._read_array_header(stream, version)
        return CandidateUnpickler(stream).load().item()

def matrix(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()

def convert(candidate, canonical):
    ric = matrix(canonical["canonical_from_input_rot_wxyz"]).T
    offset = np.asarray(canonical["com_offset"], dtype=float)
    def rows(key):
        original = np.atleast_2d(candidate[key])
        result = np.array(original, dtype=float)
        result[:,:3] = original[:,:3] @ ric.T + offset
        rot = ric @ Rotation.from_quat(original[:,[4,5,6,3]]).as_matrix()
        result[:,3:7] = Rotation.from_matrix(rot).as_quat()[:,[3,0,1,2]]
        return result
    contacts = np.asarray(candidate["ho_c"]["pos"]) @ ric.T + offset
    return dict(grasp=rows("grasp_qpos")[0], squeeze=rows("squeeze_qpos")[0],
                pregrasp=rows("pregrasp_qpos"), contact_pos=contacts,
                contact_normal=np.asarray(candidate["ho_c"]["normal"]) @ ric.T,
                contact_centroid=contacts.mean(0),
                canon_rot=np.asarray(canonical["canonical_from_input_rot_wxyz"]),
                input_com_offset=offset, frame_schema=np.array("canonical=Rci@(input-com)"))

def prepare(config_path):
    root = Path(__file__).resolve().parents[3]
    config = json.loads(Path(config_path).read_text())
    assert not config.get("asset_substitution"), "Use substitute_broom_asset.py for donor-registered priors"
    for role, spec in config["grasppose"].items():
        delivery = root / spec["delivery_dir"]
        path = delivery / "all_candidates" / spec["candidate"]
        if not path.exists():
            path = delivery / "grasp_data" / spec["candidate"]
        canonical = json.loads((delivery / "region_rank.json").read_text())["canonical_frame"]
        prior = convert(load_candidate(path), canonical)
        prior["source"] = np.frombuffer(str(path).encode(), dtype=np.uint8)
        target = root / spec["prior"]
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **prior)
        print(role, target, "wrist", prior["grasp"][:7], flush=True)

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    prepare(p.parse_args().config)
