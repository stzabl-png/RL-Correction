# OAK-D W Camera Mount

3D-printed mount for the ego / head-view camera used in the GR00T-WholeBodyControl
data-collection and VLA pipeline. The mount positions the camera that produces the
`observation.images.ego_view` stream, so reproducing this mount reproduces the
camera viewpoint the policies were trained with.

| | |
|---|---|
| **File** | [`oak_d_w_mount.step`](./oak_d_w_mount.step) (STEP AP242) |
| **Version** | v1 |
| **Camera** | [Luxonis OAK-D W](https://shop.luxonis.com/products/oak-d-w) (OV9782 global-shutter sensor) |

> **Note:** The `.step` file is stored directly in Git (it is ~44 KB of plain ASCII
> text), so a normal `git clone` or the GitHub "Download" button gives you the real
> geometry — no Git LFS required.

## Manufacturing

| Parameter | Value |
|---|---|
| Process | FDM 3D printing |
| Material | PLA |
| Layer height | 0.2 mm |
| Infill / supports | Slicer defaults — no special settings required |

Open the STEP file in any CAD tool (FreeCAD, Fusion 360, SolidWorks, or any
online STEP viewer).

## Bill of materials

| Fastener | Spec | Notes |
|---|---|---|
| Camera → mount | 2 × M4 × 8 mm | Threads directly into the printed PLA — no heat-set inserts. The OAK-D W sits flush against the mount face. |
| Mount → G1 | Stock G1 screws | Reuses the default screws at the G1 mounting location. |

## Mounting on the G1

The mount attaches at the **same location as the G1's stock Intel RealSense head
camera** — it screws into the existing RealSense mounting point using the stock
G1 screws, so no new holes or modifications are needed.

The OAK-D W sits flush against the printed face and keeps the **same orientation
plane as the stock RealSense**, angled approximately **40° relative to the head**.
This reproduces the ego-view camera pose the policies were trained with.

## Related documentation

- [Data Collection for VLA](../../docs/source/tutorials/data_collection.md) — camera
  server setup and the `ego_view` image stream this mount provides.
