import runpod
import os

def handler(job):
    info = {}
    info["root"] = os.listdir("/")
    info["runpod_volume_exists"] = os.path.exists("/runpod-volume")
    if os.path.exists("/runpod-volume"):
        info["runpod_volume_contents"] = os.listdir("/runpod-volume")
    info["workspace_exists"] = os.path.exists("/workspace")
    if os.path.exists("/workspace"):
        info["workspace_contents"] = os.listdir("/workspace")
    return info

runpod.serverless.start({"handler": handler})
