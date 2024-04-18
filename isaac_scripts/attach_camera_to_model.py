import asyncio
import omni.replicator.core as rep

async def attach_camera_to_model():

    try:
        cam = rep.create.camera(
            position=(0,-0.08,0), 
            look_at=(0,0, 100.0),
            parent="/kinova/end_effector_link",
            clipping_range=(0.10, 1000000.0)
        )
        rp = rep.create.render_product(cam, (1024, 512))

        depth_cam = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        depth_cam.attach(rp)

        data = depth_cam.get_data()
        print(data)
    except Exception as e:
        print(f"Exception: {e}")

asyncio.ensure_future(attach_camera_to_model())